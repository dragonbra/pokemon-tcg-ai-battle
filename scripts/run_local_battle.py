#!/usr/bin/env python3
"""Run one local battle using the copied official simulator.

This is a development harness only. Kaggle calls ``agent`` itself; this
script drives the lower-level ``cg`` API so that a submission can be tested
without uploading it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
sys.path.insert(0, str(SUBMISSION))


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


def run(max_steps: int, output: Path) -> dict[str, Any]:
    try:
        from cg.game import battle_finish, battle_select, battle_start
        from main import agent, read_deck_csv
    except OSError as exc:
        raise RuntimeError(
            "无法加载官方 cg 模拟器。请查看动态库错误；Linux 通常需要"
            " GLIBCXX_3.4.29 或更新的 libstdc++.so.6。"
        ) from exc

    deck = read_deck_csv()
    observation, start_data = battle_start(deck, deck)
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

            action = agent(observation)
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
        "deck": deck,
        "trace": trace,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "replays" / "local_battle.json",
    )
    args = parser.parse_args()
    result = run(args.max_steps, args.output)
    print(json.dumps({key: result[key] for key in ("finished", "result", "steps", "error")}, ensure_ascii=False))
    print(args.output.resolve())
    if result["error"] or not result["finished"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
