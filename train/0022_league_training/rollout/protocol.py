"""Immutable League rollout records."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from pathlib import Path

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any


class LeaguePolicyView(StrEnum):
    FROZEN = "frozen"
    LIVE = "live"


@dataclass(frozen=True)
class RolloutJob:
    game_id: str
    focal_deck_id: str
    opponent_deck_id: str
    opponent_view: LeaguePolicyView
    focal_first: bool
    seed: int
    source_policy_update: int
    focal_deck: tuple[int, ...]
    opponent_deck: tuple[int, ...]
    runtime_root: Path
    max_steps: int = 1_000


@dataclass(frozen=True)
class TrajectoryDecision:
    features: dict[str, Tensor]
    indices: tuple[int, ...]
    stopped: bool
    log_prob: float
    entropy: float
    value: float
    policy_deck_id: str
    policy_update: int
    reward_sign: int


@dataclass
class EpisodeTrajectory:
    job: RolloutJob
    decisions: list[TrajectoryDecision] = field(default_factory=list)
    reward: float | None = None
    turns: int = 0
    valid: bool = False
    error: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def finish(self, reward: float, turns: int) -> None:
        if reward not in (-1.0, 0.0, 1.0):
            raise ValueError(f"invalid terminal reward: {reward}")
        self.reward = reward
        self.turns = turns
        self.valid = True

    def reward_for(self, decision: TrajectoryDecision) -> float:
        if self.reward is None:
            raise ValueError("episode has no terminal reward")
        if decision.reward_sign not in (-1, 1):
            raise ValueError("trajectory reward_sign must be -1 or 1")
        return float(self.reward) * decision.reward_sign

    def policy_updates(self) -> dict[str, set[int]]:
        updates: dict[str, set[int]] = {}
        for decision in self.decisions:
            updates.setdefault(decision.policy_deck_id, set()).add(decision.policy_update)
        return updates
