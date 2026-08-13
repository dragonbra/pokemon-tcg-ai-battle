"""One-pass aggregate telemetry from the same 256 terminal rollout games."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from statistics import mean
from typing import Any, Iterable, Mapping


REQUIRED_FIELDS = {
    "result", "focal_prizes_taken", "opponent_prizes_taken", "full_turns",
    "opponent_deck_id", "opponent_policy_id", "branch",
}
FORMAL_REQUIRED_FIELDS = REQUIRED_FIELDS | {
    "focal_deck_id", "error", "unfinished", "engine_decisions",
}


def _average(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _p10(values: Iterable[float]) -> float:
    rows = sorted(values)
    if not rows:
        return 0.0
    return float(rows[max(0, math.ceil(0.1 * len(rows)) - 1)])


def _entropy(counts: Mapping[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((value / total) * math.log(value / total) for value in counts.values() if value)


def aggregate_rollout(
    games: Iterable[Mapping[str, Any]], *, curriculum_version: str,
    deck_weights: Mapping[str, float], policy_weights: Mapping[str, float],
    sampling_mode: str = "pfsp_mixture",
    red_threshold: float = 0.35,
) -> dict[str, Any]:
    rows = list(games)
    if len(rows) != 256:
        raise ValueError("0044 rollout telemetry requires exactly 256 games")
    for index, row in enumerate(rows):
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            raise ValueError(f"rollout game {index} missing telemetry fields: {sorted(missing)}")
        if row["result"] not in {"win", "loss", "draw"}:
            raise ValueError(f"rollout game {index} is not terminal")
    wins = [row for row in rows if row["result"] == "win"]
    losses = [row for row in rows if row["result"] == "loss"]
    deck_results: dict[str, list[bool]] = defaultdict(list)
    policy_results: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if row["result"] != "draw":
            deck_results[str(row["opponent_deck_id"])].append(row["result"] == "win")
            policy_results[str(row["opponent_policy_id"])].append(row["result"] == "win")
    deck_wr = {key: sum(values) / len(values) for key, values in deck_results.items()}
    policy_wr = {key: sum(values) / len(values) for key, values in policy_results.items()}
    deck_counts = Counter(str(row["opponent_deck_id"]) for row in rows)
    policy_counts = Counter(str(row["opponent_policy_id"]) for row in rows)
    branch_counts = Counter(str(row["branch"]) for row in rows)
    output = {
        "rollout/raw_win_rate": len(wins) / len(rows),
        "rollout/games": 256,
        "rollout/curriculum_version": curriculum_version,
        "rollout/opponent_deck_distribution": dict(sorted(deck_counts.items())),
        "rollout/opponent_policy_distribution": dict(sorted(policy_counts.items())),
        "rollout/branch_distribution": dict(sorted(branch_counts.items())),
        "strength/terminal_prize_margin": _average([
            float(row["focal_prizes_taken"] - row["opponent_prizes_taken"]) for row in rows
        ]),
        "strength/prizes_taken_on_loss": _average([
            float(row["focal_prizes_taken"]) for row in losses
        ]),
        "strength/opponent_prizes_taken_on_win": _average([
            float(row["opponent_prizes_taken"]) for row in wins
        ]),
        "strength/turns_to_win": _average([float(row["full_turns"]) for row in wins]),
        "strength/turns_to_loss": _average([float(row["full_turns"]) for row in losses]),
        "coverage/deck_p10_win_rate": _p10(deck_wr.values()),
        "coverage/policy_p10_win_rate": _p10(policy_wr.values()),
        "coverage/red_deck_count": sum(value < red_threshold for value in deck_wr.values()),
        "coverage/red_policy_count": sum(value < red_threshold for value in policy_wr.values()),
    }
    sampling = {
        "sampling/entropy": _entropy(deck_counts) + _entropy(policy_counts),
        "sampling/mode_uniform": float(sampling_mode == "uniform_001_067"),
        "sampling/deck_probability": dict(sorted(deck_weights.items())),
        "sampling/policy_probability": dict(sorted(policy_weights.items())),
    }
    output.update(sampling)
    if sampling_mode == "pfsp_mixture":
        output.update({
            "pfsp/sampling_entropy": sampling["sampling/entropy"],
            "pfsp/curriculum_version": curriculum_version,
            "pfsp/deck_difficulty": {
                key: 1.0 - value for key, value in sorted(deck_wr.items())
            },
            "pfsp/policy_difficulty": {
                key: 1.0 - value for key, value in sorted(policy_wr.items())
            },
            "pfsp/deck_sampling_probability": dict(sorted(deck_weights.items())),
            "pfsp/policy_sampling_probability": dict(sorted(policy_weights.items())),
        })
    elif sampling_mode != "uniform_001_067":
        raise ValueError(f"unsupported rollout sampling mode: {sampling_mode}")
    return output


class RolloutHistory:
    """Episode-ordered strength diagnostics for W&B candidate localization."""

    def __init__(self) -> None:
        self._games: list[dict[str, Any]] = []

    def append(self, games: Iterable[Mapping[str, Any]]) -> None:
        self._games.extend(dict(row) for row in games)

    def metrics(self) -> dict[str, float]:
        output: dict[str, float] = {}
        for width in (100, 500, 2000):
            rows = self._games[-width:]
            wins = sum(row["result"] == "win" for row in rows)
            losses = sum(row["result"] == "loss" for row in rows)
            draws = sum(row["result"] == "draw" for row in rows)
            prefix = f"rollout/rolling_{width}"
            output[f"{prefix}/games"] = float(len(rows))
            output[f"{prefix}/wins"] = float(wins)
            output[f"{prefix}/losses"] = float(losses)
            output[f"{prefix}/draws"] = float(draws)
            output[f"{prefix}/win_rate"] = wins / max(1, len(rows))
            output[f"{prefix}/terminal_prize_margin"] = _average([
                float(row["focal_prizes_taken"] - row["opponent_prizes_taken"])
                for row in rows
            ])
        return output


def aggregate_training_rollout(
    games: Iterable[Mapping[str, Any]], *, source_policy_update: int,
    checkpoint_update: int, curriculum_version: str,
    deck_weights: Mapping[str, float], policy_weights: Mapping[str, float],
    history: RolloutHistory, runtime_metrics: Mapping[str, float],
    sampling_mode: str = "pfsp_mixture",
) -> dict[str, Any]:
    """Formal scalar telemetry mirrored unchanged to JSONL/TensorBoard/W&B."""

    rows = list(games)
    if checkpoint_update != source_policy_update + 1:
        raise ValueError("rollout/checkpoint policy chronology mismatch")
    for index, row in enumerate(rows):
        missing = FORMAL_REQUIRED_FIELDS - set(row)
        if missing:
            raise ValueError(f"formal rollout game {index} missing fields: {sorted(missing)}")
    metrics = aggregate_rollout(
        rows, curriculum_version=curriculum_version,
        deck_weights=deck_weights, policy_weights=policy_weights,
        sampling_mode=sampling_mode,
    )
    history.append(rows)
    metrics.update(history.metrics())
    metrics.update({
        "rollout/source_policy_update": int(source_policy_update),
        "checkpoint/update": int(checkpoint_update),
        "rollout/wins": float(sum(row["result"] == "win" for row in rows)),
        "rollout/losses": float(sum(row["result"] == "loss" for row in rows)),
        "rollout/draws": float(sum(row["result"] == "draw" for row in rows)),
        "rollout/error_games": float(sum(bool(row["error"]) for row in rows)),
        "rollout/unfinished_games": float(sum(bool(row["unfinished"]) for row in rows)),
        "rollout/engine_decisions": float(sum(int(row["engine_decisions"]) for row in rows)),
        "rollout/strength_evidence": 0.0,
        "rollout/candidate_localization_only": 1.0,
    })
    for group_field, namespace in (
        ("focal_deck_id", "focal_deck"),
        ("opponent_deck_id", "opponent_deck"),
        ("opponent_policy_id", "opponent_policy"),
        ("branch", "branch"),
    ):
        values = sorted({str(row[group_field]) for row in rows})
        for value in values:
            group = [row for row in rows if str(row[group_field]) == value]
            metrics[f"rollout/{namespace}/{value}/games"] = float(len(group))
            metrics[f"rollout/{namespace}/{value}/win_rate"] = (
                sum(row["result"] == "win" for row in group) / len(group)
            )
            metrics[f"rollout/{namespace}/{value}/prize_margin"] = _average([
                float(row["focal_prizes_taken"] - row["opponent_prizes_taken"])
                for row in group
            ])
    for key, value in runtime_metrics.items():
        if not key.startswith(("rollout/", "system/")):
            raise ValueError(f"runtime metric lacks rollout/system namespace: {key}")
        metrics[key] = float(value)
    required_runtime = {
        "rollout/cuda_games_per_second", "rollout/strategic_decisions_per_second",
        "rollout/cuda_features_device_resident", "rollout/cuda_feature_d2h_bytes",
        "rollout/lane_routing_audit_pass", "rollout/lane_routing_audit_failures",
        "rollout/policy_weight_loads", "rollout/deck_static_cache_hits",
        "rollout/deck_static_cache_misses",
    }
    if required_runtime - set(metrics):
        raise ValueError(f"formal rollout runtime metrics missing: {sorted(required_runtime-set(metrics))}")
    if (
        metrics["rollout/cuda_features_device_resident"] != 1.0
        or metrics["rollout/cuda_feature_d2h_bytes"] != 0.0
        or metrics["rollout/lane_routing_audit_pass"] != 1.0
        or metrics["rollout/lane_routing_audit_failures"] != 0.0
        or metrics["rollout/policy_weight_loads"] != 1.0
    ):
        raise ValueError("formal rollout resident policy/feature health gate failed")
    return metrics


__all__ = ["RolloutHistory", "aggregate_rollout", "aggregate_training_rollout"]
