"""0040 model-only checkpoint contract; optimizer and rollout state are forbidden."""

from __future__ import annotations

from typing import Any


ACTION_SCHEMA_VERSION = "0038_compound_action_v2_semantic_parity"
DECISION_GATE_VERSION = "0038_decision_gate_v1"
CANONICALIZER_VERSION = "0038_phantom_allocation_v2_area_zero_n8"
TRAJECTORY_SCHEMA_VERSION = "0038_policy_boundary_v1"
OFFICIAL_PROTOCOL_ADAPTER_VERSION = "0038_official_primitive_adapter_v3_fail_closed"
CHECKPOINT_SCHEMA_VERSION = "0040_model_only_checkpoint_v1"
LEGACY_ACTION_CONTRACT = {
    "action_schema_version": "0038_compound_action_v1",
    "decision_gate_version": "0038_decision_gate_v1",
    "canonicalizer_version": "0038_phantom_allocation_v1",
    "trajectory_schema_version": "0038_policy_boundary_v1",
    "official_protocol_adapter_version": "0038_official_primitive_adapter_v2_chance_fallback",
}
FORBIDDEN_FIELDS = frozenset({
    "optimizer", "optimizer_state_dict", "scheduler", "scheduler_state_dict",
    "scaler", "rng_state", "rollout_buffer", "replay_buffer",
})


def checkpoint_metadata(*, source_actor_sha256: str, source_value_sha256: str,
                        version: str = "V1_zero_shot_action_boundary") -> dict[str, Any]:
    return {
        "project_id": "0040_dragapult_0809_action_boundary_rl",
        "version": version,
        "source_actor_sha256": source_actor_sha256,
        "source_value_sha256": source_value_sha256,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "decision_gate_version": DECISION_GATE_VERSION,
        "canonicalizer_version": CANONICALIZER_VERSION,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "official_protocol_adapter_version": OFFICIAL_PROTOCOL_ADAPTER_VERSION,
    }


def validate_model_only_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("0040 checkpoint schema mismatch")
    present = sorted(FORBIDDEN_FIELDS.intersection(payload))
    if present:
        raise ValueError(f"0040 model-only checkpoint contains forbidden state: {present}")
    metadata = payload.get("metadata")
    expected = checkpoint_metadata(source_actor_sha256="", source_value_sha256="")
    if not isinstance(metadata, dict):
        raise ValueError("0040 checkpoint metadata missing")
    keys = (
        "action_schema_version", "decision_gate_version", "canonicalizer_version",
        "trajectory_schema_version", "official_protocol_adapter_version",
    )
    actual_contract = {key: metadata.get(key) for key in keys}
    expected_contract = {key: expected[key] for key in keys}
    if actual_contract not in (expected_contract, LEGACY_ACTION_CONTRACT):
        raise ValueError("0040 checkpoint action contract mismatch")


__all__ = [name for name in globals() if name.isupper()] + [
    "checkpoint_metadata", "validate_model_only_payload",
]
