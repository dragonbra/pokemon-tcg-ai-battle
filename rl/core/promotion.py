from __future__ import annotations

from typing import Mapping


def is_promotable(
    candidate: Mapping[str, float],
    champion: Mapping[str, float],
    *,
    min_win_rate_delta: float = 0.01,
    max_invalid_action_rate: float = 0.0,
    max_error_rate: float = 0.0,
) -> bool:
    """Apply conservative promotion guardrails to a frozen evaluation summary."""
    candidate_win = float(candidate.get("win_rate", 0.0))
    champion_win = float(champion.get("win_rate", 0.0))
    if candidate_win < champion_win + min_win_rate_delta:
        return False
    if float(candidate.get("invalid_action_rate", 0.0)) > max_invalid_action_rate:
        return False
    if float(candidate.get("error_rate", 0.0)) > max_error_rate:
        return False
    return True
