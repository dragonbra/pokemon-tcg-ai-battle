"""Complete model-only delta checkpoints for reconstructable 0045 policies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


SCHEMA_VERSION = "0045_complete_delta_model_only_v2"
LEGACY_SCHEMA_VERSION = "0045_minimal_lora_model_only_v1"
FORBIDDEN_RESUME_FIELDS = frozenset((
    "optimizer_state_dict", "scheduler_state_dict", "grad_scaler_state_dict",
    "rng_state", "dataloader_state", "rollout_buffer", "replay",
))


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii")); digest.update(b"\0")
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8")); digest.update(b"\0")
        digest.update(_tensor_sha256(value).encode("ascii")); digest.update(b"\0")
    return digest.hexdigest()


def trainable_parameter_names(model: nn.Module) -> tuple[str, ...]:
    names = tuple(sorted(
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ))
    if not names:
        raise RuntimeError("complete-delta checkpoint has no trainable parameters")
    missing = sorted(set(names) - set(model.state_dict()))
    if missing:
        raise RuntimeError(f"trainable parameters have no state_dict keys: {missing}")
    return names


def build_complete_delta_checkpoint(
    model: nn.Module, envelope: Mapping[str, Any],
) -> dict[str, Any]:
    if FORBIDDEN_RESUME_FIELDS.intersection(envelope):
        raise RuntimeError("complete-delta envelope contains forbidden resume state")
    state = model.state_dict()
    names = trainable_parameter_names(model)
    delta = {name: state[name].detach().cpu() for name in names}
    parameters = dict(model.named_parameters())
    payload = dict(envelope)
    payload.update({
        "schema_version": SCHEMA_VERSION,
        "state_dict": delta,
        "checkpoint_integrity": {
            "schema_version": "0045_complete_delta_integrity_v1",
            "status": "PASS",
            "saved_tensor_policy": "all_and_only_requires_grad_named_parameters",
            "immutable_base_required": True,
            "trainable_parameter_names": list(names),
            "trainable_tensor_count": len(names),
            "total_trainable_params": sum(
                int(parameters[name].numel()) for name in names
            ),
            "tensor_sha256": {
                name: _tensor_sha256(value) for name, value in delta.items()
            },
            "delta_state_sha256": _state_sha256(delta),
            "source_full_state_tensor_count": len(state),
            "source_full_state_sha256": _state_sha256(state),
            "forbidden_resume_fields": sorted(FORBIDDEN_RESUME_FIELDS),
        },
    })
    validate_complete_delta(model, payload, require_model_match=True)
    return payload


def validate_complete_delta(
    model: nn.Module, payload: Mapping[str, Any], *, require_model_match: bool = False,
) -> dict[str, Any]:
    forbidden = sorted(FORBIDDEN_RESUME_FIELDS.intersection(payload))
    if forbidden:
        raise RuntimeError(f"complete-delta checkpoint contains forbidden fields: {forbidden}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("checkpoint is not complete-delta model-only V2")
    state = payload.get("state_dict")
    integrity = payload.get("checkpoint_integrity")
    if not isinstance(state, Mapping) or not isinstance(integrity, Mapping):
        raise RuntimeError("complete-delta checkpoint is missing state/integrity")
    expected_names = trainable_parameter_names(model)
    if (
        tuple(integrity.get("trainable_parameter_names", ())) != expected_names
        or set(state) != set(expected_names)
    ):
        raise RuntimeError("complete-delta trainable tensor inventory mismatch")
    if integrity.get("trainable_tensor_count") != len(expected_names):
        raise RuntimeError("complete-delta trainable tensor count mismatch")
    parameters = dict(model.named_parameters())
    expected_parameters = sum(int(parameters[name].numel()) for name in expected_names)
    if integrity.get("total_trainable_params") != expected_parameters:
        raise RuntimeError("complete-delta trainable parameter count mismatch")
    observed_hashes = {name: _tensor_sha256(state[name]) for name in expected_names}
    if observed_hashes != integrity.get("tensor_sha256"):
        raise RuntimeError("complete-delta per-tensor hash mismatch")
    if _state_sha256(state) != integrity.get("delta_state_sha256"):
        raise RuntimeError("complete-delta aggregate hash mismatch")
    if require_model_match:
        current = model.state_dict()
        mismatched = [
            name for name in expected_names
            if not torch.equal(current[name].detach().cpu(), state[name].detach().cpu())
        ]
        if mismatched:
            raise RuntimeError(f"complete-delta differs from in-memory model: {mismatched}")
        if (
            len(current) != integrity.get("source_full_state_tensor_count")
            or _state_sha256(current) != integrity.get("source_full_state_sha256")
        ):
            raise RuntimeError("complete-delta source full-model hash mismatch")
    return dict(integrity)


def load_complete_delta(model: nn.Module, payload: Mapping[str, Any]) -> Any:
    validate_complete_delta(model, payload)
    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"complete-delta checkpoint has unexpected tensors: {incompatible.unexpected_keys}"
        )
    integrity = payload["checkpoint_integrity"]
    state = model.state_dict()
    if (
        len(state) != integrity.get("source_full_state_tensor_count")
        or _state_sha256(state) != integrity.get("source_full_state_sha256")
    ):
        raise RuntimeError(
            "complete-delta reconstruction does not match saved full-model identity"
        )
    return incompatible


def load_model_only_delta(
    model: nn.Module, payload: Mapping[str, Any], *, allow_legacy: bool,
) -> Any:
    schema = payload.get("schema_version")
    if schema == SCHEMA_VERSION:
        return load_complete_delta(model, payload)
    if schema != LEGACY_SCHEMA_VERSION or not allow_legacy:
        raise RuntimeError(f"unsupported 0045 model-only checkpoint schema: {schema}")
    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"legacy 0045 checkpoint has unexpected tensors: {incompatible.unexpected_keys}"
        )
    return incompatible


def audit_reconstruction(
    source_model: nn.Module, reconstructed_model: nn.Module,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    validate_complete_delta(source_model, payload, require_model_match=True)
    source = source_model.state_dict()
    reconstructed = reconstructed_model.state_dict()
    if set(source) != set(reconstructed):
        raise RuntimeError("reconstructed full-model tensor inventory mismatch")
    mismatched = [
        name for name in source
        if not torch.equal(
            source[name].detach().cpu(), reconstructed[name].detach().cpu()
        )
    ]
    if mismatched:
        raise RuntimeError(f"reconstructed full-model tensors differ: {mismatched}")
    return {
        "schema_version": "0045_complete_delta_reconstruction_audit_v1",
        "status": "PASS",
        "checkpoint_update": payload.get("update"),
        "trainable_tensor_count": len(payload["state_dict"]),
        "full_state_tensor_count": len(source),
        "source_full_state_sha256": _state_sha256(source),
        "reconstructed_full_state_sha256": _state_sha256(reconstructed),
    }


def atomic_save_complete_delta(
    path: Path, payload: Mapping[str, Any], model: nn.Module,
) -> None:
    validate_complete_delta(model, payload, require_model_match=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    stored = torch.load(temporary, map_location="cpu", weights_only=True)
    validate_complete_delta(model, stored, require_model_match=True)
    temporary.replace(path)


__all__ = [
    "LEGACY_SCHEMA_VERSION", "SCHEMA_VERSION", "atomic_save_complete_delta",
    "audit_reconstruction", "build_complete_delta_checkpoint",
    "load_complete_delta", "load_model_only_delta", "trainable_parameter_names",
    "validate_complete_delta",
]
