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
    red_threshold: float = 0.35,
) -> dict[str, Any]:
    rows = list(games)
    if len(rows) != 256:
        raise ValueError("0043 rollout telemetry requires exactly 256 games")
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
    return {
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
        "pfsp/sampling_entropy": _entropy(deck_counts) + _entropy(policy_counts),
        "pfsp/curriculum_version": curriculum_version,
        "pfsp/deck_difficulty": {key: 1.0 - value for key, value in sorted(deck_wr.items())},
        "pfsp/policy_difficulty": {key: 1.0 - value for key, value in sorted(policy_wr.items())},
        "pfsp/deck_sampling_probability": dict(sorted(deck_weights.items())),
        "pfsp/policy_sampling_probability": dict(sorted(policy_weights.items())),
        "coverage/deck_p10_win_rate": _p10(deck_wr.values()),
        "coverage/policy_p10_win_rate": _p10(policy_wr.values()),
        "coverage/red_deck_count": sum(value < red_threshold for value in deck_wr.values()),
        "coverage/red_policy_count": sum(value < red_threshold for value in policy_wr.values()),
    }


__all__ = ["aggregate_rollout"]
