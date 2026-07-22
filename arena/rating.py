from __future__ import annotations

from dataclasses import dataclass, replace
from math import erf, sqrt


@dataclass(frozen=True)
class RatingState:
    name: str
    mu: float = 600.0
    sigma: float = 200.0
    elo: float = 600.0
    games: int = 0
    status: str = "active"
    below_threshold_checkpoints: int = 0


@dataclass(frozen=True)
class RatingUpdate:
    before_a: RatingState
    before_b: RatingState
    after_a: RatingState
    after_b: RatingState
    expected_a: float
    delta_a: float
    delta_b: float
    outcome: float
    engine: str
    revision: int


class GaussianRatingEngine:
    method = "official_gaussian_approx"

    def __init__(
        self,
        *,
        initial_mu: float = 600.0,
        initial_sigma: float = 200.0,
        beta: float = 200.0,
        update_scale: float = 80.0,
        min_sigma: float = 40.0,
        revision: int = 1,
    ) -> None:
        self.initial_mu = initial_mu
        self.initial_sigma = initial_sigma
        self.beta = beta
        self.update_scale = update_scale
        self.min_sigma = min_sigma
        self.revision = revision

    def expected(self, a: RatingState, b: RatingState) -> float:
        scale = sqrt(a.sigma**2 + b.sigma**2 + 2 * self.beta**2)
        z = (a.mu - b.mu) / (scale * sqrt(2.0))
        return 0.5 * (1.0 + erf(z))

    def update(self, a: RatingState, b: RatingState, outcome: float) -> RatingUpdate:
        _validate_outcome(outcome)
        expected = self.expected(a, b)
        uncertainty = min(1.0, (a.sigma + b.sigma) / (2.0 * self.initial_sigma))
        delta = self.update_scale * uncertainty * (outcome - expected)
        shrink = 1.0 - 0.05 * abs(outcome - expected)
        after_a = replace(
            a,
            mu=a.mu + delta,
            sigma=max(self.min_sigma, a.sigma * shrink),
            games=a.games + 1,
        )
        after_b = replace(
            b,
            mu=b.mu - delta,
            sigma=max(self.min_sigma, b.sigma * shrink),
            games=b.games + 1,
        )
        return RatingUpdate(
            a,
            b,
            after_a,
            after_b,
            expected,
            delta,
            -delta,
            outcome,
            self.method,
            self.revision,
        )


class EloRatingEngine:
    method = "elo_compat"

    def __init__(self, *, initial: float = 600.0, k_factor: float = 32.0) -> None:
        self.initial = initial
        self.k_factor = k_factor
        self.revision = 1

    def expected(self, a: RatingState, b: RatingState) -> float:
        return 1.0 / (1.0 + 10.0 ** ((b.elo - a.elo) / 400.0))

    def update(self, a: RatingState, b: RatingState, outcome: float) -> RatingUpdate:
        _validate_outcome(outcome)
        expected = self.expected(a, b)
        delta = self.k_factor * (outcome - expected)
        after_a = replace(a, elo=a.elo + delta, games=a.games + 1)
        after_b = replace(b, elo=b.elo - delta, games=b.games + 1)
        return RatingUpdate(
            a,
            b,
            after_a,
            after_b,
            expected,
            delta,
            -delta,
            outcome,
            self.method,
            self.revision,
        )


def eligible_for_demotion(
    state: RatingState,
    threshold: float = 300.0,
    minimum_games: int = 20,
) -> bool:
    return (
        state.games >= minimum_games
        and state.mu < threshold
        and state.below_threshold_checkpoints >= 2
    )


def apply_checkpoint_status(
    states: tuple[RatingState, ...] | list[RatingState],
    *,
    threshold: float = 300.0,
    minimum_games: int = 20,
) -> tuple[RatingState, ...]:
    """在 checkpoint 更新连续低分计数；只改变状态，不删除任何卡组。"""
    updated: list[RatingState] = []
    for state in states:
        if state.games >= minimum_games and state.mu < threshold:
            checkpoints = state.below_threshold_checkpoints + 1
        else:
            checkpoints = 0
        candidate = replace(state, below_threshold_checkpoints=checkpoints)
        if eligible_for_demotion(candidate, threshold, minimum_games):
            candidate = replace(candidate, status="demoted")
        elif candidate.status != "demoted":
            candidate = replace(candidate, status="active")
        updated.append(candidate)
    return tuple(updated)


def _validate_outcome(outcome: float) -> None:
    if outcome not in (0.0, 0.5, 1.0):
        raise ValueError("rating outcome must be 0.0, 0.5 or 1.0")
