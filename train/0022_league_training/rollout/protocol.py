"""Immutable League rollout records."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from pathlib import Path

from torch import Tensor


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
