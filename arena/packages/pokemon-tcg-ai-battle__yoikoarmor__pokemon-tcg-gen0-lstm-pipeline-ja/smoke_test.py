"""Local smoke test for the stable Kaggle submission package.

Runs the bundled cg engine with the submission agent on both sides and records
legality, latency, fallback-like engine errors, and memory behavior.  The script
is intentionally self-contained so it can be executed from an extracted
submission archive.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import main  # noqa: E402
from src import engine  # noqa: E402


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def _is_legal(choice: Any, obs: dict[str, Any]) -> bool:
    if not isinstance(choice, list) or not all(isinstance(x, int) for x in choice):
        return False
    select = obs.get("select") if isinstance(obs, dict) else None
    options = (select or {}).get("option") or []
    min_count = int((select or {}).get("minCount") or 0)
    max_count = int((select or {}).get("maxCount") or 0)
    return (
        min_count <= len(choice) <= max_count
        and len(set(choice)) == len(choice)
        and all(0 <= x < len(options) for x in choice)
    )


def _readonly_safe_output(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _run_one_game(game_index: int, max_steps: int) -> dict[str, Any]:
    deck = [int(x) for x in main.agent({"select": None})]
    obs, start_data = engine.battle_start(deck, deck)
    stats = {
        "game_index": game_index,
        "start_error_player": int(start_data.errorPlayer),
        "start_error_type": int(start_data.errorType),
        "decisions": 0,
        "illegal": 0,
        "engine_errors": 0,
        "latencies_ms": [],
        "termination": "unknown",
    }
    if obs is None:
        stats["termination"] = "start_failed"
        return stats

    try:
        for _step in range(max_steps):
            if any(int(log.get("type", -1)) == 23 for log in (obs.get("logs") or [])):
                stats["termination"] = "result"
                break
            select = obs.get("select") if isinstance(obs, dict) else None
            if not select:
                stats["termination"] = "no_select"
                break

            t0 = time.perf_counter()
            try:
                choice = main.agent(obs)
            except Exception:
                stats["engine_errors"] += 1
                choice = main._raw_legal_fallback(obs)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            stats["latencies_ms"].append(latency_ms)
            stats["decisions"] += 1

            if not _is_legal(choice, obs):
                stats["illegal"] += 1
                stats["termination"] = "illegal_action"
                break

            try:
                obs = engine.battle_select(choice)
            except Exception:
                stats["engine_errors"] += 1
                stats["termination"] = "engine_select_error"
                break
        else:
            stats["termination"] = "max_steps"
    finally:
        try:
            engine.battle_finish()
        except Exception:
            stats["engine_errors"] += 1
    return stats


def _multi_select_check() -> dict[str, Any]:
    obs = {
        "select": {
            "minCount": 2,
            "maxCount": 3,
            "option": [{"x": 0}, {"x": 1}, {"x": 2}, {"x": 3}],
        }
    }
    choice = main._raw_legal_fallback(obs)
    return {
        "choice": choice,
        "legal": _is_legal(choice, obs),
        "expected_min": 2,
        "expected_max": 3,
    }


def run_smoke(games: int, max_steps: int, min_decisions: int) -> dict[str, Any]:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    tracemalloc.start()
    memory_start = tracemalloc.get_traced_memory()[0]
    game_stats: list[dict[str, Any]] = []
    latencies: list[float] = []
    total_decisions = 0

    for game_index in range(games):
        stats = _run_one_game(game_index, max_steps)
        game_stats.append(stats)
        latencies.extend(float(x) for x in stats["latencies_ms"])
        total_decisions += int(stats["decisions"])
        if total_decisions >= min_decisions and game_index + 1 >= games:
            break

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    illegal = sum(int(s["illegal"]) for s in game_stats)
    engine_errors = sum(int(s["engine_errors"]) for s in game_stats)
    decisions = sum(int(s["decisions"]) for s in game_stats)
    legal_rate = 1.0 if decisions == 0 else (decisions - illegal) / decisions
    submission_stats = main.submission_stats() if hasattr(main, "submission_stats") else {}
    policy_decisions = int(submission_stats.get("generation0_policy_decisions", 0))
    rule_fallbacks = int(submission_stats.get("rule_fallback_decisions", 0))
    first_legal_fallbacks = int(submission_stats.get("first_legal_fallback_decisions", 0))
    policy_usage_rate = 0.0 if decisions <= 0 else policy_decisions / decisions
    rule_fallback_rate = 0.0 if decisions <= 0 else rule_fallbacks / decisions
    multi_select = _multi_select_check()
    terminations: dict[str, int] = {}
    for stats in game_stats:
        key = str(stats["termination"])
        terminations[key] = terminations.get(key, 0) + 1

    return {
        "package_dir": str(BASE_DIR),
        "games": len(game_stats),
        "decisions": decisions,
        "fault": engine_errors,
        "invalid": illegal,
        "generation0_policy_decisions": policy_decisions,
        "rule_fallback_decisions": rule_fallbacks,
        "first_legal_fallback_decisions": first_legal_fallbacks,
        "policy_usage_rate": policy_usage_rate,
        "rule_fallback_rate": rule_fallback_rate,
        "fallback_reasons": submission_stats.get("fallback_reasons", {}),
        "fallback": rule_fallbacks + first_legal_fallbacks,
        "hidden_reset_errors": 0,
        "legal_action_rate": legal_rate,
        "multi_select_fallback": multi_select,
        "avg_select_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
        "p50_select_latency_ms": _percentile(latencies, 50),
        "p95_select_latency_ms": _percentile(latencies, 95),
        "p99_select_latency_ms": _percentile(latencies, 99),
        "max_select_latency_ms": max(latencies) if latencies else 0.0,
        "memory_start_bytes": memory_start,
        "memory_peak_bytes": peak,
        "memory_end_bytes": current,
        "terminations": terminations,
        "success": illegal == 0 and engine_errors == 0 and multi_select["legal"],
    }


def main_cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--min-decisions", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = run_smoke(args.games, args.max_steps, args.min_decisions)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    path = _readonly_safe_output(args.output)
    if path is not None:
        path.write_text(output + "\n", encoding="utf-8")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
