from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

from evaluation.packages.loader import SubmissionPackage


@dataclass(frozen=True)
class RolloutJob:
    episode_id: str
    opponent: SubmissionPackage
    candidate_first: bool
    seed: int
    max_steps: int = 1_000
    opponent_remote: bool = False


@dataclass
class TrajectoryDecision:
    features: dict[str, Tensor]
    action: tuple[int, ...]
    stopped: bool
    old_log_prob: float
    old_value: float
    entropy: float
    turn: int
    inference_ms: float = 0.0
    selected_options: tuple[dict[str, Any], ...] = ()


@dataclass
class Episode:
    episode_id: str
    opponent: str
    opponent_hash: str
    candidate_first: bool
    seed: int
    valid: bool
    reward: float | None
    winner: int | None
    status: str
    error: str | None
    engine_selections: int
    final_turn: int | None
    complete_rounds: int | None
    decisions: list[TrajectoryDecision] = field(default_factory=list)
    diagnostics: dict[str, float] = field(default_factory=dict)


__all__ = ["Episode", "RolloutJob", "TrajectoryDecision"]
