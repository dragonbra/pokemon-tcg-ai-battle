from __future__ import annotations

from typing import Mapping


def has_minimum_evaluation_coverage(
    summary: Mapping[str, object],
    *,
    opponent_count: int = 17,
    games_per_opponent: int = 10,
) -> bool:
    """Require the minimum fixed-pool sample before treating results as promotion evidence."""
    if opponent_count < 1 or games_per_opponent < 1:
        raise ValueError("opponent_count and games_per_opponent must be positive")
    by_opponent = summary.get("by_opponent")
    if not isinstance(by_opponent, Mapping) or len(by_opponent) != opponent_count:
        return False
    return all(
        isinstance(result, Mapping)
        and int(result.get("games", 0)) >= games_per_opponent
        for result in by_opponent.values()
    )


def is_promotable(
    candidate: Mapping[str, object],
    champion: Mapping[str, object],
    *,
    min_win_rate_delta: float = 0.01,
    max_invalid_action_rate: float = 0.0,
    max_error_rate: float = 0.0,
) -> bool:
    """Apply conservative promotion guardrails to a frozen evaluation summary."""
    if not has_minimum_evaluation_coverage(candidate):
        return False
    candidate_win = float(candidate.get("win_rate", 0.0))
    champion_win = float(champion.get("win_rate", 0.0))
    if candidate_win < champion_win + min_win_rate_delta:
        return False
    if float(candidate.get("invalid_action_rate", 0.0)) > max_invalid_action_rate:
        return False
    if float(candidate.get("error_rate", 0.0)) > max_error_rate:
        return False
    return True
