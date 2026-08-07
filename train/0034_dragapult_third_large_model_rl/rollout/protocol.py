"""Immutable trajectory records for decoder-only PPO."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any


@dataclass(frozen=True)
class RolloutJob:
    game_id: str
    opponent_id: str
    focal_first: bool
    seed: int
    source_policy_update: int
    focal_deck: tuple[int, ...]
    opponent_deck: tuple[int, ...]
    runtime_root: Path
    policy_seed: int = 0
    search_seed: int = 0
    engine_library: Path | None = None
    max_steps: int = 1_000
    ability_repeat_limit: int = 8
    full_round_draw_limit: int = 50


@dataclass(frozen=True)
class TrajectoryDecision:
    features: dict[str, Tensor]
    indices: tuple[int, ...]
    stopped: bool
    log_prob: float
    entropy: float
    value: float
    policy_update: int
    turn: int | None = None


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


__all__ = ["EpisodeTrajectory", "RolloutJob", "TrajectoryDecision"]
