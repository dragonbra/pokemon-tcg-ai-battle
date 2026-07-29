from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

from .models import Match, MatchupCell


def hhi(probabilities: Iterable[float]) -> float:
    return sum(value * value for value in probabilities)


def jensen_shannon_divergence(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("distributions must have the same dimensions")
    if not left:
        return 0.0
    left_total = sum(left)
    right_total = sum(right)
    if left_total <= 0 or right_total <= 0:
        raise ValueError("distributions must have positive mass")
    p = [value / left_total for value in left]
    q = [value / right_total for value in right]
    midpoint = [(a + b) / 2 for a, b in zip(p, q, strict=True)]

    def divergence(values: list[float]) -> float:
        return sum(
            value * math.log2(value / center)
            for value, center in zip(values, midpoint, strict=True)
            if value > 0
        )

    return (divergence(p) + divergence(q)) / 2


def wilson_interval(
    effective_wins: float, n: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("Wilson interval requires n > 0")
    proportion = effective_wins / n
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / n + z * z / (4 * n * n)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def evidence_grade(wins: int, losses: int, draws: int) -> str:
    n = wins + losses + draws
    if n < 15:
        return "insufficient"
    if n < 30:
        return "directional"
    low, high = wilson_interval(wins + 0.5 * draws, n)
    if low > 0.5:
        return "strong_advantage"
    if high < 0.5:
        return "strong_disadvantage"
    return "directional"


def aggregate_matchups(matches: Iterable[Match]) -> dict[tuple[str, str], MatchupCell]:
    matrix: dict[tuple[str, str], MatchupCell] = defaultdict(MatchupCell)
    for match in matches:
        forward = matrix[(match.deck1, match.deck2)]
        if match.deck1 == match.deck2:
            if match.winner in {match.player1, match.player2}:
                forward.decisive += 1
            else:
                forward.draws += 1
            continue
        reverse = matrix[(match.deck2, match.deck1)]
        if match.winner == match.player1:
            forward.wins += 1
            reverse.losses += 1
        elif match.winner == match.player2:
            forward.losses += 1
            reverse.wins += 1
        else:
            forward.draws += 1
            reverse.draws += 1
    return dict(matrix)
