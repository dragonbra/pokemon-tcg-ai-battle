"""Periodic Benchmark V2 cadence and W&B scalar contract."""

from __future__ import annotations

from typing import Any, Mapping


INTERVAL_UPDATES = 10
GAMES = 2048


def is_due(checkpoint_update: int) -> bool:
    if checkpoint_update < 0:
        raise ValueError("checkpoint update must be non-negative")
    return checkpoint_update > 0 and checkpoint_update % INTERVAL_UPDATES == 0


def wandb_metrics(
    report: Mapping[str, Any], *, initial_baseline_update: int | None = None,
) -> dict[str, float | int]:
    """Extract only common-seed Benchmark V2 facts into ``eval/*``."""
    if report.get("status") != "PASS":
        raise RuntimeError("periodic Benchmark V2 report did not PASS")
    if report.get("benchmark_id") != "Benchmark-V2":
        raise RuntimeError("periodic evaluation is not Benchmark V2")
    summary = report.get("summary")
    update = report.get("focal_checkpoint_update")
    if not isinstance(summary, Mapping) or summary.get("games") != GAMES:
        raise RuntimeError("periodic Benchmark V2 report is not CUDA-2048")
    # A new version logs its exact initial checkpoint before collecting any PPO
    # rollout. Later evaluations remain restricted to the ten-update cadence.
    if (
        not isinstance(update, int)
        or (update != initial_baseline_update and not is_due(update))
    ):
        raise RuntimeError("periodic Benchmark V2 report is off the ten-update boundary")
    return {
        "eval/checkpoint_update": update,
        "eval/games": GAMES,
        "eval/wins": int(summary["wins"]),
        "eval/losses": int(summary["losses"]),
        "eval/draws": int(summary["draws"]),
        "eval/win_rate": float(summary["win_rate"]),
        "eval/focal_first_win_rate": float(summary["focal_first_win_rate"]),
        "eval/focal_second_win_rate": float(summary["focal_second_win_rate"]),
        "eval/games_per_second": float(summary["games_per_second"]),
        "eval/official_engine": 1.0,
        "eval/greedy": 1.0,
        "eval/benchmark_v2": 1.0,
        "eval/common_random_numbers": 1.0,
    }


__all__ = ["GAMES", "INTERVAL_UPDATES", "is_due", "wandb_metrics"]
