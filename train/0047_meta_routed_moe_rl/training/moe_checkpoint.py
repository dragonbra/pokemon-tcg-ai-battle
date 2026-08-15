"""Compact FP32 model-only checkpoints for the 0047 Meta-routed MoE."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from ..policy.moe_actor_critic import (
    EFFECTIVE_0814_SHA256,
    POLICY_0814_SHA256,
    VALUE_0814_SHA256,
    MetaRoutedMoEActorCritic,
)


SCHEMA_VERSION = "0047_meta_routed_moe_compact_fp32_delta_v1"
RETENTION_POLICY = "evaluated_nodes_only_after_successful_eval"
FORBIDDEN_FIELDS = {
    "optimizer",
    "optimizer_state",
    "scheduler",
    "grad_scaler",
    "rng_state",
    "rollout_buffer",
    "replay",
}


def _delta_names(model: MetaRoutedMoEActorCritic) -> tuple[str, ...]:
    return tuple(name for name in model.state_dict() if not name.startswith("actor."))


def build_compact_checkpoint(
    model: MetaRoutedMoEActorCritic,
    update: int,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if update < 0:
        raise ValueError("checkpoint update must be nonnegative")
    forbidden = FORBIDDEN_FIELDS.intersection(metadata)
    if forbidden:
        raise ValueError(f"model-only metadata contains forbidden fields: {sorted(forbidden)}")
    state = model.state_dict()
    delta = {
        name: state[name].detach().cpu().clone()
        for name in _delta_names(model)
    }
    non_fp32 = [
        name for name, value in delta.items()
        if value.is_floating_point() and value.dtype != torch.float32
    ]
    if non_fp32:
        raise TypeError(f"compact training checkpoint must remain FP32: {non_fp32[:8]}")
    return {
        "schema_version": SCHEMA_VERSION,
        "update": int(update),
        "base": {
            "policy_id": "Policy-0814",
            "actor_checkpoint_sha256": POLICY_0814_SHA256,
            "value_checkpoint_sha256": VALUE_0814_SHA256,
            "effective_policy_sha256": EFFECTIVE_0814_SHA256,
        },
        "delta_state_dict": delta,
        "metadata": {
            **dict(metadata),
            "checkpoint_contents": "fp32_effective_delta_only",
            "checkpoint_retention": RETENTION_POLICY,
            "optimizer_state_saved": False,
        },
    }


def load_compact_checkpoint(
    model: MetaRoutedMoEActorCritic,
    payload: Mapping[str, Any],
) -> None:
    base_representation_sha256 = model.representation_sha256()
    expected_top = {"schema_version", "update", "base", "delta_state_dict", "metadata"}
    if set(payload) != expected_top or payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("0047 compact checkpoint schema/inventory mismatch")
    if FORBIDDEN_FIELDS.intersection(payload):
        raise RuntimeError("0047 compact checkpoint contains forbidden resume state")
    base = payload.get("base")
    expected_base = {
        "policy_id": "Policy-0814",
        "actor_checkpoint_sha256": POLICY_0814_SHA256,
        "value_checkpoint_sha256": VALUE_0814_SHA256,
        "effective_policy_sha256": EFFECTIVE_0814_SHA256,
    }
    if base != expected_base:
        raise RuntimeError("FATAL: compact checkpoint immutable Policy-0814 base mismatch")
    delta = payload.get("delta_state_dict")
    if not isinstance(delta, Mapping):
        raise RuntimeError("0047 compact checkpoint has no delta state")
    expected_delta = set(_delta_names(model))
    if set(delta) != expected_delta:
        missing = sorted(expected_delta - set(delta))
        unexpected = sorted(set(delta) - expected_delta)
        raise RuntimeError(
            f"0047 compact delta inventory mismatch: missing={missing[:8]} "
            f"unexpected={unexpected[:8]}"
        )
    state = model.state_dict()
    for name, value in delta.items():
        target = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"compact delta entry is not a tensor: {name}")
        if value.shape != target.shape or value.dtype != target.dtype:
            raise RuntimeError(
                f"compact delta tensor contract mismatch for {name}: "
                f"checkpoint={tuple(value.shape)}/{value.dtype} "
                f"model={tuple(target.shape)}/{target.dtype}"
            )
        if value.is_floating_point() and value.dtype != torch.float32:
            raise TypeError(f"compact training checkpoint is not FP32: {name}")
        state[name] = value
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict compact checkpoint load failed: {incompatible}")
    if model.representation_sha256() != base_representation_sha256:
        raise RuntimeError("FATAL: compact checkpoint changed immutable representation")
    model.router_logits.requires_grad_(model.soft_routing)


def atomic_save_compact_checkpoint(
    path: Path,
    model: MetaRoutedMoEActorCritic,
    update: int,
    *,
    metadata: Mapping[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(build_compact_checkpoint(model, update, metadata=metadata), temporary)
    temporary.replace(path)
    return path


def prune_non_eval_checkpoints(
    checkpoint_dir: Path,
    *,
    through_update: int,
    eval_interval: int,
    evaluation_status: str,
) -> tuple[Path, ...]:
    if evaluation_status != "PASS":
        raise RuntimeError("checkpoint pruning requires a successful evaluation")
    if eval_interval <= 0 or through_update <= 0 or through_update % eval_interval:
        raise ValueError("pruning boundary must be a positive evaluated update")
    removed: list[Path] = []
    for path in sorted(checkpoint_dir.glob("update-*.pt")):
        try:
            update = int(path.stem.removeprefix("update-"))
        except ValueError as error:
            raise RuntimeError(f"unrecognized checkpoint filename: {path.name}") from error
        if update <= through_update and update % eval_interval:
            path.unlink()
            removed.append(path)
    return tuple(removed)


__all__ = [
    "RETENTION_POLICY",
    "SCHEMA_VERSION",
    "atomic_save_compact_checkpoint",
    "build_compact_checkpoint",
    "load_compact_checkpoint",
    "prune_non_eval_checkpoints",
]
