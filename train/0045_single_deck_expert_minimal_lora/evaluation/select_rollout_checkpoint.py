"""Select a healthy deck-007 checkpoint from canonical mirrored rollout metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from ..assets import sha256_file


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _number(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"metric is missing or non-finite: {key}")
    return float(value)


def select(
    metrics: Path, checkpoints: Path, *, allowed_source_updates: set[int] | None = None,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    eligible: list[dict[str, Any]] = []
    history_007: list[float] = []
    history_margin: list[float] = []
    for row in rows:
        source_update = int(_number(row, "rollout/source_policy_update"))
        if allowed_source_updates is not None and source_update not in allowed_source_updates:
            continue
        checkpoint = checkpoints / f"update-{source_update:06d}.pt"
        win007 = _number(row, "rollout/focal_deck/007/win_rate")
        margin007 = _number(row, "rollout/focal_deck/007/prize_margin")
        history_007.append(win007)
        history_margin.append(margin007)
        hard_gate = {
            "checkpoint_exists": checkpoint.is_file(),
            "deck_007_games_128": _number(row, "rollout/focal_deck/007/games") == 128,
            "error_games_zero": _number(row, "rollout/error_games") == 0,
            "unfinished_games_zero": _number(row, "rollout/unfinished_games") == 0,
            "routing_failures_zero": _number(row, "rollout/lane_routing_audit_failures") == 0,
            "routing_pass": _number(row, "rollout/lane_routing_audit_pass") == 1,
            "feature_d2h_zero": _number(row, "rollout/cuda_feature_d2h_bytes") == 0,
            "behavior_guard_clear": _number(row, "ppo/hard_behavior_kl_guard_triggered") == 0,
            "behavior_kl_below_guard": _number(row, "ppo/behavior_kl")
            < _number(row, "ppo/hard_behavior_kl_guard"),
            "reference_kl_finite": math.isfinite(_number(row, "ppo/reference_kl")),
        }
        trailing3 = _mean(history_007[-3:])
        trailing5 = _mean(history_007[-5:])
        margin3 = _mean(history_margin[-3:]) / 6.0
        policy_floor = min(
            _number(row, "rollout/opponent_policy/Champion-G1/win_rate"),
            _number(row, "rollout/opponent_policy/Policy-0809/win_rate"),
        )
        components = {
            "deck_007_current": win007,
            "deck_007_trailing3": trailing3,
            "deck_007_trailing5": trailing5,
            "deck_007_prize_margin_trailing3_normalized": margin3,
            "rolling_500": _number(row, "rollout/rolling_500/win_rate"),
            "rolling_2000": _number(row, "rollout/rolling_2000/win_rate"),
            "opponent_policy_floor": policy_floor,
            "opponent_deck_p10": _number(row, "coverage/deck_p10_win_rate"),
        }
        score = (
            0.30 * components["deck_007_current"]
            + 0.25 * components["deck_007_trailing3"]
            + 0.15 * components["deck_007_trailing5"]
            + 0.08 * components["deck_007_prize_margin_trailing3_normalized"]
            + 0.06 * components["rolling_500"]
            + 0.06 * components["rolling_2000"]
            + 0.05 * components["opponent_policy_floor"]
            + 0.05 * components["opponent_deck_p10"]
        )
        if all(hard_gate.values()):
            eligible.append({
                "source_policy_update": source_update,
                "metrics_record_trainer_update": int(_number(row, "trainer/update")),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "score": score,
                "components": components,
                "health": {
                    "behavior_kl": _number(row, "ppo/behavior_kl"),
                    "reference_kl": _number(row, "ppo/reference_kl"),
                    "clip_fraction": _number(row, "ppo/clip_fraction"),
                    **hard_gate,
                },
                "curriculum_version": row["rollout/curriculum_version"],
            })
    if not eligible:
        raise RuntimeError("no completed rollout checkpoint passes the selection gates")
    ranked = sorted(eligible, key=lambda item: (item["score"], item["source_policy_update"]), reverse=True)
    return {
        "schema_version": "0044_deck007_rollout_checkpoint_selection_v1",
        "method": "70pct_deck007_smoothed_30pct_general_coverage_with_health_hard_gates",
        "chronology": "metrics rollout/source_policy_update selects the same-number checkpoint",
        "metrics_path": str(metrics),
        "metrics_rows_observed": len(rows),
        "latest_complete_source_policy_update": max(item["source_policy_update"] for item in eligible),
        "selected": ranked[0],
        "top_candidates": ranked[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select(args.metrics.resolve(), args.checkpoints.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
