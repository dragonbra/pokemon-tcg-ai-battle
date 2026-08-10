"""Immutable trajectory records for decoder-only PPO."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any


DEFAULT_FULL_ROUND_DRAW_LIMIT = 50


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
    opponent_policy_id: str
    policy_seed: int = 0
    search_seed: int = 0
    engine_library: Path | None = None
    max_steps: int = 1_000
    ability_repeat_limit: int = 20
    full_round_draw_limit: int = DEFAULT_FULL_ROUND_DRAW_LIMIT
    action_boundary_mode: str = "shadow"
    trace_policy: str = "full"
    normal_trace_sample_modulus: int = 100
    include_action_prefix: bool = False
    focal_won_toss: bool | None = None

    def __post_init__(self) -> None:
        if self.action_boundary_mode not in {"shadow", "enabled"}:
            raise ValueError("action_boundary_mode must be shadow or enabled")
        if self.trace_policy not in {"full", "compact", "errors_and_sample"}:
            raise ValueError("invalid trace_policy")
        if self.normal_trace_sample_modulus < 1:
            raise ValueError("normal_trace_sample_modulus must be positive")
        if self.full_round_draw_limit < 0:
            raise ValueError("full_round_draw_limit cannot be negative")
        if not self.opponent_policy_id.strip():
            raise ValueError("opponent_policy_id must identify a concrete policy")


def require_opponent_policy_binding(
    jobs: list[RolloutJob], *, materialized_policy_id: str
) -> None:
    requested = {job.opponent_policy_id for job in jobs}
    if requested != {materialized_policy_id}:
        raise RuntimeError(
            "FATAL: rollout opponent policy binding mismatch: "
            f"requested={sorted(requested)!r}, "
            f"materialized={materialized_policy_id!r}"
        )


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
    parameter_log_prob: float = 0.0
    parameter_entropy: float = 0.0
    macro_action: dict[str, Any] | None = None
    engine_event_index: int = -1
    auxiliary_values: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalMacroAction:
    family: str
    root_indices: tuple[int, ...]
    root_stopped: bool
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyTransition:
    """One semi-MDP record per real policy boundary, never per UI callback."""

    pre_action_features: dict[str, Tensor]
    canonical_macro_action: CanonicalMacroAction
    root_old_logprob: float
    parameter_old_logprob: float
    joint_old_logprob: float
    pre_action_value: float
    accumulated_reward: float
    accumulated_discount: float
    post_commit_observation: dict[str, Any] | None
    next_value: float
    done: bool
    engine_event_span: tuple[int, int]
    strategic: bool = True
    valid: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = self.root_old_logprob + self.parameter_old_logprob
        if abs(self.joint_old_logprob - expected) > 1e-6:
            raise ValueError("joint_old_logprob must equal root plus parameter logprob")
        if not 0.0 <= self.accumulated_discount <= 1.0:
            raise ValueError("accumulated_discount must be in [0, 1]")
        if self.done and self.next_value != 0.0:
            raise ValueError("terminal transition must have next_value=0")


@dataclass
class EpisodeTrajectory:
    job: RolloutJob
    decisions: list[TrajectoryDecision] = field(default_factory=list)
    reward: float | None = None
    turns: int = 0
    valid: bool = False
    error: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    policy_transitions: list[PolicyTransition] = field(default_factory=list)

    def finish(self, reward: float, turns: int) -> None:
        if reward not in (-1.0, 0.0, 1.0):
            raise ValueError(f"invalid terminal reward: {reward}")
        self.reward = reward
        self.turns = turns
        self.valid = True


__all__ = [
    "CanonicalMacroAction", "DEFAULT_FULL_ROUND_DRAW_LIMIT", "EpisodeTrajectory",
    "PolicyTransition", "RolloutJob", "TrajectoryDecision",
    "require_opponent_policy_binding",
]
