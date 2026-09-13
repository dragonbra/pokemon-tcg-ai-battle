"""Versioned post-audit Agent/runtime contracts, independent of legacy checkpoints."""

ACTION_BOUNDARY_SCHEMA_VERSION = "0038_compound_action_v2_semantic_parity"
DECISION_GATE_VERSION = "0038_decision_gate_v1"
CANONICALIZER_VERSION = "0038_phantom_allocation_v2_area_zero_n8"
TRAJECTORY_SCHEMA_VERSION = "0038_policy_boundary_v1"
OFFICIAL_PROTOCOL_ADAPTER_VERSION = "0038_official_primitive_adapter_v3_fail_closed"
FEATURE_PREPROCESSING_VERSION = "0038_cpu_authoritative_cuda_feature_parity_v1"


__all__ = [name for name in globals() if name.isupper()]
