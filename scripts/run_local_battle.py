#!/usr/bin/env python3
"""Run one local battle using the copied official simulator.

This is a development harness only. Kaggle calls ``agent`` itself; this
script drives the lower-level ``cg`` API so that a submission can be tested
without uploading it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"


def _load_submission(name: str):
    submission = SUBMISSION / name
    main_path = submission / "main.py"
    if not main_path.exists():
        raise ValueError(f"unknown submission directory: {name}")
    module_name = f"local_submission_{name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, main_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load {main_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _available_submissions() -> list[str]:
    return sorted(
        path.name
        for path in SUBMISSION.iterdir()
        if path.is_dir() and (path / "main.py").exists() and (path / "deck.csv").exists()
    )


def _state_summary(observation: dict[str, Any]) -> dict[str, Any]:
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    return {
        "turn": current.get("turn"),
        "yourIndex": current.get("yourIndex"),
        "result": current.get("result"),
        "selectType": select.get("type"),
        "selectContext": select.get("context"),
        "optionCount": len(select.get("option") or []),
        "minCount": select.get("minCount"),
        "maxCount": select.get("maxCount"),
    }


def run(agent0_name: str, agent1_name: str, max_steps: int, output: Path) -> dict[str, Any]:
    try:
        sys.path.insert(0, str(SUBMISSION / agent0_name))
        from cg.game import battle_finish, battle_select, battle_start
    except OSError as exc:
        raise RuntimeError(
            "无法加载官方 cg 模拟器。请查看动态库错误；Linux 通常需要"
            " GLIBCXX_3.4.29 或更新的 libstdc++.so.6。"
        ) from exc

    agent0_module = _load_submission(agent0_name)
    agent1_module = _load_submission(agent1_name)
    deck0 = agent0_module.read_deck_csv()
    deck1 = agent1_module.read_deck_csv()
    observation, start_data = battle_start(deck0, deck1)
    if not start_data.battlePtr or observation is None:
        raise RuntimeError(
            f"BattleStart failed: errorPlayer={start_data.errorPlayer}, "
            f"errorType={start_data.errorType}"
        )

    trace: list[dict[str, Any]] = []
    finished = False
    error: str | None = None
    try:
        for step in range(max_steps):
            summary = _state_summary(observation)
            current = observation.get("current") or {}
            if current.get("result", -1) != -1:
                finished = True
                trace.append({"step": step, "state": summary, "observation": observation})
                break

            select = observation.get("select")
            if select is None:
                raise RuntimeError(
                    "BattleStart returned select=None. The low-level API already receives "
                    "both decks, so an initial deck callback is not expected here."
                )

            your_index = int((observation.get("current") or {}).get("yourIndex", 0))
            selected_agent = agent0_module if your_index == 0 else agent1_module
            action = selected_agent.agent(observation)
            trace.append(
                {
                    "step": step,
                    "state": summary,
                    "action": action,
                    "observation": observation,
                }
            )
            observation = battle_select(action)
        else:
            error = f"step limit reached ({max_steps})"
    except Exception as exc:  # preserve the full trace for debugging
        error = f"{type(exc).__name__}: {exc}"
    finally:
        battle_finish()

    final_state = observation.get("current") if observation else None
    result = {
        "finished": finished,
        "result": None if final_state is None else final_state.get("result"),
        "steps": len(trace),
        "error": error,
        "agent0": agent0_name,
        "agent1": agent1_name,
        "deck0": deck0,
        "deck1": deck1,
        "trace": trace,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    available = _available_submissions()
    parser.add_argument("--agent0", choices=available, default="alakazam_v1")
    parser.add_argument("--agent1", choices=available, default="alakazam_v1")
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "replays" / "local_battle.json",
    )
    args = parser.parse_args()
    result = run(args.agent0, args.agent1, args.max_steps, args.output)
    print(json.dumps({key: result[key] for key in ("finished", "result", "steps", "error")}, ensure_ascii=False))
    print(args.output.resolve())
    if result["error"] or not result["finished"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
