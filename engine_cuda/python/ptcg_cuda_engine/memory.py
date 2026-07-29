from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .policy_pool import PolicyPoolManifest, PolicySpec


MIB = 1024**2
GIB = 1024**3


DTYPE_SCALE_FROM_FP32 = {
    "fp32": 1.0,
    "fp16": 0.5,
    "bf16": 0.5,
    "int8": 0.25,
}


@dataclass(frozen=True)
class MemoryEstimate:
    environments: int
    policy_count: int
    frozen_policy_count: int
    cohort_capacity: int
    engine_raw_bytes: int
    engine_operational_bytes: int
    frozen_weight_bytes: int
    learner_training_bytes: int
    shared_inference_workspace_bytes: int
    learner_rollout_bytes: int
    total_bytes: int
    checkpoint_rows: tuple[dict[str, Any], ...]
    assumptions: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for field in (
            "engine_raw_bytes",
            "engine_operational_bytes",
            "frozen_weight_bytes",
            "learner_training_bytes",
            "shared_inference_workspace_bytes",
            "learner_rollout_bytes",
            "total_bytes",
        ):
            value = result[field]
            result[field.replace("_bytes", "_mib")] = round(value / MIB, 3)
            result[field.replace("_bytes", "_gib")] = round(value / GIB, 4)
        return result


def _fp32_parameter_bytes(policy: PolicySpec, workspace_root: Path | None) -> tuple[int, str]:
    if policy.parameter_mib is not None:
        return int(policy.parameter_mib * MIB), "manifest_parameter_mib"
    path = Path(policy.checkpoint)
    if not path.is_absolute() and workspace_root is not None:
        path = workspace_root / path
    if path.is_file():
        return path.stat().st_size, "checkpoint_file_size"
    return 154 * MIB, "fallback_154_mib"


def estimate_memory(
    environments: int,
    manifest: PolicyPoolManifest,
    *,
    workspace_root: str | Path | None = None,
    cohort_capacity: int | None = None,
    state_bytes_per_env: int = 32 * 1024,
    codec_bytes_per_env: int = 22_224,
    scratch_bytes_per_env: int = 8 * 1024,
    control_bytes_per_env: int = 2 * 1024,
    fixed_rule_bytes: int = 32 * MIB,
    allocator_graph_reserve_bytes: int = 64 * MIB,
    engine_headroom: float = 1.20,
    inference_workspace_bytes_per_row: int = 256 * 1024,
    learner_training_multiplier_from_fp32: float = 4.0,
    learner_rollout_bytes_per_env: int = 0,
) -> MemoryEstimate:
    """Estimate operational VRAM, separating engine and model costs.

    The learner multiplier approximates BF16/FP16 forward weights plus FP32
    master weights, gradients, and two FP32 Adam moments. Model activations are
    represented separately as a shared maximum-cohort workspace because frozen
    policies are dispatched sequentially in the first implementation.
    """

    manifest.validate()
    if environments <= 0:
        raise ValueError("environments must be positive")
    if min(
        state_bytes_per_env,
        codec_bytes_per_env,
        scratch_bytes_per_env,
        control_bytes_per_env,
    ) < 0:
        raise ValueError("per-environment byte counts must be non-negative")
    if engine_headroom < 1.0:
        raise ValueError("engine_headroom must be at least 1")
    if cohort_capacity is None:
        balanced = math.ceil(environments / len(manifest.policies))
        cohort_capacity = max(1, math.ceil(balanced * 1.25))
    if cohort_capacity <= 0:
        raise ValueError("cohort_capacity must be positive")

    route_bytes = len(manifest.policies) * cohort_capacity * 4 + len(manifest.policies) * 4
    raw_engine = (
        environments
        * (state_bytes_per_env + codec_bytes_per_env + scratch_bytes_per_env + control_bytes_per_env)
        + route_bytes
        + fixed_rule_bytes
        + allocator_graph_reserve_bytes
    )
    operational_engine = math.ceil(raw_engine * engine_headroom)

    root = None if workspace_root is None else Path(workspace_root)
    frozen_weights = 0
    learner_training = 0
    checkpoint_rows: list[dict[str, Any]] = []
    for policy in manifest.policies:
        fp32_bytes, source = _fp32_parameter_bytes(policy, root)
        if policy.frozen:
            resident_bytes = math.ceil(fp32_bytes * DTYPE_SCALE_FROM_FP32[policy.dtype])
            frozen_weights += resident_bytes
        else:
            resident_bytes = math.ceil(fp32_bytes * learner_training_multiplier_from_fp32)
            learner_training += resident_bytes
        checkpoint_rows.append(
            {
                "policy_id": policy.policy_id,
                "name": policy.name,
                "deck": policy.deck,
                "frozen": policy.frozen,
                "dtype": policy.dtype,
                "fp32_parameter_mib": round(fp32_bytes / MIB, 3),
                "resident_mib": round(resident_bytes / MIB, 3),
                "size_source": source,
            }
        )

    inference_workspace = cohort_capacity * inference_workspace_bytes_per_row
    learner_rollout = environments * learner_rollout_bytes_per_env
    total = (
        operational_engine
        + frozen_weights
        + learner_training
        + inference_workspace
        + learner_rollout
    )
    assumptions = {
        "state_bytes_per_env": state_bytes_per_env,
        "codec_bytes_per_env": codec_bytes_per_env,
        "scratch_bytes_per_env": scratch_bytes_per_env,
        "control_bytes_per_env": control_bytes_per_env,
        "fixed_rule_mib": fixed_rule_bytes / MIB,
        "allocator_graph_reserve_mib": allocator_graph_reserve_bytes / MIB,
        "engine_headroom": engine_headroom,
        "inference_workspace_kib_per_cohort_row": inference_workspace_bytes_per_row / 1024,
        "learner_training_multiplier_from_fp32": learner_training_multiplier_from_fp32,
        "learner_rollout_bytes_per_env": learner_rollout_bytes_per_env,
        "cohort_capacity_requires_reset_quota": True,
        "frozen_activation_workspaces_reused_sequentially": True,
        "optimizer": "Adam-like two FP32 moments",
    }
    return MemoryEstimate(
        environments=environments,
        policy_count=len(manifest.policies),
        frozen_policy_count=manifest.frozen_policy_count,
        cohort_capacity=cohort_capacity,
        engine_raw_bytes=raw_engine,
        engine_operational_bytes=operational_engine,
        frozen_weight_bytes=frozen_weights,
        learner_training_bytes=learner_training,
        shared_inference_workspace_bytes=inference_workspace,
        learner_rollout_bytes=learner_rollout,
        total_bytes=total,
        checkpoint_rows=tuple(checkpoint_rows),
        assumptions=assumptions,
    )
