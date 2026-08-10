"""Predeclared statistical equivalence calculations for CPU/CUDA RNG audits."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence


Z_95 = 1.959963984540054


@dataclass(frozen=True, slots=True)
class EquivalenceInterval:
    estimate: float
    lower: float
    upper: float
    margin: float

    @property
    def passed(self) -> bool:
        return self.lower >= -self.margin and self.upper <= self.margin


def binary_difference_interval(
    success_a: int,
    total_a: int,
    success_b: int,
    total_b: int,
    *,
    margin: float = 0.01,
) -> EquivalenceInterval:
    if min(total_a, total_b) <= 0:
        raise ValueError("binary totals must be positive")
    if not 0 <= success_a <= total_a or not 0 <= success_b <= total_b:
        raise ValueError("binary successes outside totals")
    pa = success_a / total_a
    pb = success_b / total_b
    standard_error = math.sqrt(pa * (1 - pa) / total_a + pb * (1 - pb) / total_b)
    half_width = Z_95 * standard_error
    estimate = pa - pb
    return EquivalenceInterval(estimate, estimate - half_width, estimate + half_width, margin)


@dataclass(frozen=True, slots=True)
class TotalVariationResult:
    estimate: float
    upper_95: float
    margin: float

    @property
    def passed(self) -> bool:
        return self.upper_95 <= self.margin


def multinomial_total_variation(
    counts_a: Mapping[int | str, int],
    counts_b: Mapping[int | str, int],
    *,
    margin: float = 0.02,
) -> TotalVariationResult:
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())
    if min(total_a, total_b) <= 0 or any(value < 0 for value in (*counts_a.values(), *counts_b.values())):
        raise ValueError("multinomial counts must be nonnegative with positive totals")
    keys = set(counts_a) | set(counts_b)
    tv = 0.5 * sum(abs(counts_a.get(key, 0) / total_a - counts_b.get(key, 0) / total_b) for key in keys)
    # A conservative distribution-free multinomial L1 concentration bound.
    categories = max(1, len(keys))
    epsilon = 0.5 * math.sqrt(2 * categories * math.log(40) * (1 / total_a + 1 / total_b))
    return TotalVariationResult(tv, min(1.0, tv + epsilon), margin)


@dataclass(frozen=True, slots=True)
class CorrelationResult:
    lag1: float
    lower_99: float
    upper_99: float

    @property
    def passed(self) -> bool:
        return self.lower_99 <= self.lag1 <= self.upper_99


def lag1_binary_correlation(values: Sequence[int]) -> CorrelationResult:
    if len(values) < 10 or any(value not in (0, 1) for value in values):
        raise ValueError("lag correlation requires at least ten binary values")
    mean = sum(values) / len(values)
    denominator = sum((value - mean) ** 2 for value in values)
    lag1 = 0.0 if denominator == 0 else sum(
        (values[index] - mean) * (values[index - 1] - mean)
        for index in range(1, len(values))
    ) / denominator
    envelope = 2.5758293035489004 / math.sqrt(len(values))
    return CorrelationResult(lag1, -envelope, envelope)


__all__ = [
    "CorrelationResult", "EquivalenceInterval", "TotalVariationResult",
    "binary_difference_interval", "lag1_binary_correlation",
    "multinomial_total_variation",
]
