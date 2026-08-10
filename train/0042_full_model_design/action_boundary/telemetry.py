"""Shadow-only action-boundary telemetry with no behavior side effects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .decision_gate import DecisionGateResult


@dataclass(frozen=True, slots=True)
class ShadowDecisionTelemetry:
    session_id: str
    select_context: Any
    action_family: str
    actor: int
    raw_option_count: int
    legal_completion_count: int
    canonical_completion_count: int
    stop_legal: bool
    cancel_legal: bool
    decline_legal: bool
    pass_legal: bool
    expected_gate: str
    transformer_seconds: float = 0.0
    gru_seconds: float = 0.0
    value_seconds: float = 0.0
    service_roundtrip_seconds: float = 0.0
    written_to_policy_trajectory: bool = False
    value_sample: bool = False
    same_outcome_alias_count: int = 0

    @classmethod
    def from_gate(cls, *, session_id: str, context: Any, action_family: str,
                  actor: int, gate: DecisionGateResult, **timings: Any):
        return cls(
            session_id=session_id, select_context=context, action_family=action_family,
            actor=actor, raw_option_count=gate.raw_option_count,
            legal_completion_count=gate.legal_completion_count,
            canonical_completion_count=gate.canonical_completion_count,
            stop_legal=gate.stop_legal, cancel_legal=gate.cancel_legal,
            decline_legal=gate.decline_legal, pass_legal=gate.pass_legal,
            expected_gate=gate.classification.value, **timings,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["ShadowDecisionTelemetry"]
