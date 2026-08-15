"""Periodic 0045 Benchmark Tiny V2 cadence and W&B scalar contract."""

from __future__ import annotations

from typing import Any, Mapping


INTERVAL_UPDATES = 5
FORMAL_EVALUATE_INTERVAL_UPDATES = 10
GAMES = 512


def is_due(
    checkpoint_update: int, *, interval_updates: int = INTERVAL_UPDATES,
) -> bool:
    if checkpoint_update < 0:
        raise ValueError("checkpoint update must be non-negative")
    if interval_updates <= 0:
        raise ValueError("evaluation interval must be positive")
    return checkpoint_update > 0 and checkpoint_update % interval_updates == 0


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


def wandb_metrics_three_pool(
    report: Mapping[str, Any], *, initial_baseline_update: int | None = None,
    interval_updates: int = INTERVAL_UPDATES,
) -> dict[str, float | int]:
    """Extract three independent Policy-0809 CUDA-512 sentinel curves."""
    if report.get("status") != "PASS":
        raise RuntimeError("three-pool periodic evaluation did not PASS")
    if report.get("benchmark_id") != "Benchmark-Tiny-V2-Policy0809-Three-Pool":
        raise RuntimeError("periodic evaluation is not the Policy-0809 three-pool contract")
    update = report.get("focal_checkpoint_update")
    if (
        not isinstance(update, int)
        or (
            update != initial_baseline_update
            and not is_due(update, interval_updates=interval_updates)
        )
    ):
        raise RuntimeError("three-pool report is off the five-update boundary")
    summaries = report.get("pool_summaries")
    if not isinstance(summaries, Mapping) or set(summaries) != {
        "low_score", "priority", "remaining",
    }:
        raise RuntimeError("three-pool report has the wrong pool inventory")
    metrics: dict[str, float | int] = {
        "eval/checkpoint_update": update,
        "eval/official_engine": 1.0,
        "eval/greedy": 1.0,
        "eval/policy0809_three_pool": 1.0,
        "eval/formal_ten_update_point": float(
            update % FORMAL_EVALUATE_INTERVAL_UPDATES == 0
        ),
        "eval/common_random_numbers": 1.0,
    }
    for pool_name in ("low_score", "priority", "remaining"):
        summary = summaries[pool_name]
        if not isinstance(summary, Mapping) or summary.get("games") != GAMES:
            raise RuntimeError(f"periodic pool {pool_name} is not CUDA-512")
        prefix = f"eval/{pool_name}"
        metrics.update({
            f"{prefix}/games": GAMES,
            f"{prefix}/wins": int(summary["wins"]),
            f"{prefix}/losses": int(summary["losses"]),
            f"{prefix}/draws": int(summary["draws"]),
            f"{prefix}/win_rate": float(summary["win_rate"]),
            f"{prefix}/focal_first_win_rate": float(
                summary["focal_first_win_rate"]
            ),
            f"{prefix}/focal_second_win_rate": float(
                summary["focal_second_win_rate"]
            ),
            f"{prefix}/games_per_second": float(summary["games_per_second"]),
        })
    aggregate = report.get("summary")
    if not isinstance(aggregate, Mapping) or aggregate.get("games") != 3 * GAMES:
        raise RuntimeError("three-pool aggregate is not 1,536 games")
    metrics.update({
        "eval/three_pool_aggregate/games": 3 * GAMES,
        "eval/three_pool_aggregate/wins": int(aggregate["wins"]),
        "eval/three_pool_aggregate/losses": int(aggregate["losses"]),
        "eval/three_pool_aggregate/draws": int(aggregate["draws"]),
        "eval/three_pool_aggregate/win_rate": float(aggregate["win_rate"]),
        "eval/three_pool_aggregate/games_per_second": float(
            aggregate["games_per_second"]
        ),
    })
    return metrics


__all__ = [
    "FORMAL_EVALUATE_INTERVAL_UPDATES", "GAMES", "INTERVAL_UPDATES",
    "is_due", "wandb_metrics", "wandb_metrics_three_pool",
]
