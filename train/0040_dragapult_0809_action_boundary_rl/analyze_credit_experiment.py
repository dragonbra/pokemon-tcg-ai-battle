"""Summarize one completed 0034 credit experiment without strength overclaiming."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROJECT = "0040_dragapult_0809_action_boundary_rl"


def _window(rows: list[dict[str, Any]], key: str, *, first: bool) -> float | None:
    values = [float(row[key]) for row in rows if key in row]
    if not values:
        return None
    selected = values[:5] if first else values[-5:]
    return statistics.fmean(selected)


def _slope(rows: list[dict[str, Any]], key: str) -> float | None:
    points = [
        (float(row["trainer/update"]), float(row[key]))
        for row in rows
        if "trainer/update" in row and key in row
    ]
    if len(points) < 2:
        return None
    mean_x = statistics.fmean(x for x, _ in points)
    mean_y = statistics.fmean(y for _, y in points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def analyze(version: str) -> dict[str, Any]:
    artifact = ROOT / "rl_runs" / PROJECT / "versions" / version / "artifact"
    status = json.loads((artifact / "status.json").read_text(encoding="utf-8"))
    if status.get("state") not in {"complete", "completed"}:
        raise RuntimeError(f"experiment is not complete: {status.get('state')}")
    metrics_path = artifact / "training_metrics.jsonl"
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or int(rows[-1].get("trainer/update", -1)) != 20:
        raise RuntimeError("credit experiment must contain exactly 20 completed updates")
    evaluation = [
        {
            "checkpoint_update": int(row["eval/checkpoint_update"]),
            "wins": int(row["eval/wins"]),
            "losses": int(row["eval/losses"]),
            "draws": int(row["eval/draws"]),
            "win_rate": float(row["eval/win_rate"]),
            "first_win_rate": float(row["eval/first_win_rate"]),
            "second_win_rate": float(row["eval/second_win_rate"]),
        }
        for row in rows
        if "eval/win_rate" in row
    ]
    if not evaluation:
        raise RuntimeError("experiment has no frozen greedy evaluations")
    selected = max(evaluation, key=lambda row: (row["win_rate"], row["checkpoint_update"]))
    tracked = (
        "rollout/win_rate",
        "ppo/explained_variance",
        "ppo/value_loss",
        "ppo/return_std",
        "ppo/first_decision_terminal_credit_mean",
        "ppo/reference_kl",
        "ppo/behavior_kl",
        "ppo/entropy",
        "system/end_to_end/wall_seconds",
    )
    curves = {
        key: {
            "first_5_mean": _window(rows, key, first=True),
            "last_5_mean": _window(rows, key, first=False),
            "slope_per_update": _slope(rows, key),
        }
        for key in tracked
    }
    errors = sum(float(row.get("rollout/error_games", 0.0)) for row in rows)
    clocks = {int(row.get("ppo/credit_clock_turn_config", -1)) for row in rows}
    turn_equal_weights = {
        int(row.get("ppo/loss_weighting_turn_equal_config", 0)) for row in rows
    }
    lambdas = {float(row.get("ppo/gae_lambda", -1.0)) for row in rows}
    payload = {
        "schema": "0034_credit_experiment_analysis_v1",
        "version": version,
        "updates": 20,
        "zero_error": errors == 0.0,
        "rollout_error_games": errors,
        "credit_clock_turn_values": sorted(clocks),
        "loss_weighting_turn_equal_values": sorted(turn_equal_weights),
        "gae_lambda_values": sorted(lambdas),
        "frozen_evaluations": evaluation,
        "selected_checkpoint": selected,
        "selection_rule": "highest comparable frozen greedy win rate; latest checkpoint on ties",
        "curves": curves,
        "strength_boundary": (
            "rollout curves are diagnostics; checkpoint strength requires the formal "
            "official-engine Frozen51 evaluation"
        ),
    }
    output = artifact / "analysis.json"
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.version), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
