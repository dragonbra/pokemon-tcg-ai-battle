"""Periodic 0045 Benchmark Tiny V2 cadence and W&B scalar contract."""

from __future__ import annotations

from typing import Any, Mapping


INTERVAL_UPDATES = 5
FORMAL_EVALUATE_INTERVAL_UPDATES = 10
GAMES = 512


def is_due(checkpoint_update: int) -> bool:
    if checkpoint_update < 0:
        raise ValueError("checkpoint update must be non-negative")
    return checkpoint_update > 0 and checkpoint_update % INTERVAL_UPDATES == 0


def wandb_metrics(
    report: Mapping[str, Any], *, initial_baseline_update: int | None = None,
) -> dict[str, float | int]:
    """Extract only deployment-effective Tiny V2 facts into ``eval/*``."""
    if report.get("status") != "PASS":
        raise RuntimeError("periodic Benchmark V2 report did not PASS")
    if report.get("benchmark_id") != "Benchmark-Tiny-V2":
        raise RuntimeError("periodic evaluation is not Benchmark Tiny V2")
    summary = report.get("summary")
    update = report.get("focal_checkpoint_update")
    if not isinstance(summary, Mapping) or summary.get("games") != GAMES:
        raise RuntimeError("periodic Benchmark Tiny V2 report is not CUDA-512")
    # A new version logs its exact initial checkpoint before collecting any PPO
    # rollout. Later evaluations remain restricted to the ten-update cadence.
    if (
        not isinstance(update, int)
        or (update != initial_baseline_update and not is_due(update))
    ):
        raise RuntimeError("periodic Benchmark Tiny V2 report is off the five-update boundary")
    return {
        "eval/checkpoint_update": update,
        "eval/games": GAMES,
        "eval/wins": int(summary["wins"]),
        "eval/losses": int(summary["losses"]),
        "eval/draws": int(summary["draws"]),
        "eval/win_rate": float(summary["win_rate"]),
        "eval/core/win_rate": float(summary["win_rate"]),
        "eval/focal_first_win_rate": float(summary["focal_first_win_rate"]),
        "eval/focal_second_win_rate": float(summary["focal_second_win_rate"]),
        "eval/games_per_second": float(summary["games_per_second"]),
        "eval/official_engine": 1.0,
        "eval/greedy": 1.0,
        "eval/benchmark_tiny_v2": 1.0,
        "eval/formal_ten_update_point": float(update % FORMAL_EVALUATE_INTERVAL_UPDATES == 0),
        "eval/common_random_numbers": 1.0,
    }


__all__ = [
    "FORMAL_EVALUATE_INTERVAL_UPDATES", "GAMES", "INTERVAL_UPDATES",
    "is_due", "wandb_metrics",
]
