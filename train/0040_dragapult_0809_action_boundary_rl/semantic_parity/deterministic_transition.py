"""Model-free primitive transition comparison with first-divergence stopping."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

from .canonical_state import CanonicalAuthorityState, FieldDifference, canonical_state_diff


@dataclass(frozen=True, slots=True)
class TransitionDivergence:
    step: int
    stage: str
    field_difference: FieldDifference | None
    left: Any = None
    right: Any = None


@dataclass(frozen=True, slots=True)
class TransitionParityResult:
    status: str
    compared_steps: int
    first_divergence: TransitionDivergence | None


def compare_primitive_trace(
    official_steps: Sequence[Mapping[str, Any]],
    cuda_steps: Sequence[Mapping[str, Any]],
) -> TransitionParityResult:
    if len(official_steps) != len(cuda_steps):
        return TransitionParityResult("FAIL", min(len(official_steps), len(cuda_steps)), TransitionDivergence(min(len(official_steps), len(cuda_steps)), "trace_length", None, len(official_steps), len(cuda_steps)))
    stages = ("primitive_action", "random_outcome", "continuation", "legal_options", "event_trace", "reward_delta")
    for index, (official, cuda) in enumerate(zip(official_steps, cuda_steps)):
        for stage in stages:
            if official.get(stage) != cuda.get(stage):
                return TransitionParityResult("FAIL", index, TransitionDivergence(index, stage, None, official.get(stage), cuda.get(stage)))
        official_state = CanonicalAuthorityState.from_official(official["post_state"])
        cuda_state = CanonicalAuthorityState.from_cuda(cuda["post_state"])
        differences = canonical_state_diff(official_state, cuda_state)
        if differences:
            return TransitionParityResult("FAIL", index, TransitionDivergence(index, "post_state", differences[0]))
    return TransitionParityResult("PASS", len(official_steps), None)


__all__ = ["TransitionDivergence", "TransitionParityResult", "compare_primitive_trace"]
