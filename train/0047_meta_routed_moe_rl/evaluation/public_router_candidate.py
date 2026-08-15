"""Compact, identity-bound 0045 public-router deployment artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from ..assets import sha256_file
from ..semantic_runtime.deployment.public_meta_memory import (
    DEFAULT_UPDATE,
    PUBLIC_POLICY_ID,
    ROUTED_UPDATES,
    RULE_MANIFEST,
)
from .candidate import _expanded_actor_state_dict, _tensor_hash
from .meta_oracle_v1 import ORACLE_POLICY_ID, route_manifest


SCHEMA_VERSION = "0045_public_meta_router_compact_heads_v1"
HEAD_FIELDS = (
    "action_decoder_state_dict",
    "policy_option_lora_state_dict",
    "allocation_head_state_dict",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _tensor_digest(rows: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(rows.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8")); digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii")); digest.update(b"\0")
        digest.update(_canonical_json(list(tensor.shape))); digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _effective(rows: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: value.float() if torch.is_floating_point(value) else value
        for name, value in rows.items()
    }


def _head_states(candidate: Mapping[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    prefix = "action_decoder."
    decoder = {
        name.removeprefix(prefix): value
        for name, value in candidate["actor_state_dict"].items()
        if name.startswith(prefix)
    }
    if not decoder:
        raise RuntimeError("portable candidate has no Action Decoder tensors")
    return {
        "action_decoder_state_dict": decoder,
        "policy_option_lora_state_dict": dict(candidate["policy_option_lora_state_dict"]),
        "allocation_head_state_dict": dict(candidate["allocation_head_state_dict"]),
    }


def _head_effective_hash(states: Mapping[str, Mapping[str, torch.Tensor]]) -> str:
    ordered = (
        states["action_decoder_state_dict"],
        states["policy_option_lora_state_dict"],
        states["allocation_head_state_dict"],
    )
    rows: dict[str, torch.Tensor] = {}
    for index, state in enumerate(ordered):
        rows.update({f"module{index}.{name}": value for name, value in _effective(state).items()})
    return _tensor_digest(rows)


def _shared_effective_state(candidate: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    expanded = _expanded_actor_state_dict(candidate["actor_state_dict"])
    return _effective({
        name: value for name, value in expanded.items()
        if not name.startswith("action_decoder.")
    })


def _floating_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        if torch.is_floating_point(value):
            yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _floating_tensors(item)


def materialize_compact_heads(
    *, portable_candidates: Mapping[int, Path], qualifying_report: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Write only non-default routed heads and bind them to V12 CUDA identity."""
    if set(portable_candidates) != set(ROUTED_UPDATES):
        raise ValueError(f"public router requires exactly {ROUTED_UPDATES}")
    audit = qualifying_report.get("focal_public_router_identity_audit") or {}
    if (
        qualifying_report.get("status") != "PASS"
        or audit.get("status") != "PASS"
        or audit.get("policy_id") != PUBLIC_POLICY_ID
        or audit.get("rules") != RULE_MANIFEST
        or audit.get("default_checkpoint_update") != DEFAULT_UPDATE
    ):
        raise RuntimeError("qualifying public-router deployment identity is not V12 PASS")

    candidates: dict[int, dict[str, Any]] = {}
    effective_ids: dict[str, str] = {}
    head_ids: dict[str, str] = {}
    shared_reference: dict[str, torch.Tensor] | None = None
    for update in ROUTED_UPDATES:
        path = portable_candidates[update]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != "0045_minimal_lora_candidate_v1":
            raise RuntimeError(f"U{update} is not an 0045 portable candidate")
        metadata = payload.get("metadata") or {}
        if metadata.get("checkpoint_update") != update:
            raise RuntimeError(f"U{update} portable checkpoint update mismatch")
        expected = audit["candidate_materializations"][str(update)]
        if sha256_file(path) != expected["portable_checkpoint_sha256"]:
            raise RuntimeError(f"U{update} portable file differs from qualifying CUDA artifact")
        effective_id = _tensor_hash(payload, expected["focal_exact_deck_sha256"])
        if effective_id != expected["effective_candidate_sha256"]:
            raise RuntimeError(f"U{update} deployment-effective identity mismatch")
        states = _head_states(payload)
        head_id = _head_effective_hash(states)
        if head_id != audit["source_head_identity"][str(update)]:
            raise RuntimeError(f"U{update} routed-head identity mismatch")
        shared = _shared_effective_state(payload)
        if shared_reference is None:
            shared_reference = shared
        elif set(shared) != set(shared_reference) or any(
            not torch.equal(shared[name], shared_reference[name]) for name in shared
        ):
            raise RuntimeError(f"U{update} shared Actor tensors are not exact-equal")
        candidates[update] = payload
        effective_ids[str(update)] = effective_id
        head_ids[str(update)] = head_id

    assert shared_reference is not None
    shared_id = _tensor_digest(shared_reference)
    if shared_id != audit["shared_actor"]["shared_effective_sha256"]:
        raise RuntimeError("compact shared Actor identity differs from qualifying CUDA run")
    source_identity_payload = {
        "policy_id": ORACLE_POLICY_ID,
        "route": route_manifest(),
        "shared_effective_sha256": shared_id,
        "head_effective_sha256": head_ids,
        "candidate_effective_sha256": effective_ids,
    }
    source_composite = _json_hash(source_identity_payload)
    public_identity_payload = {
        "policy_id": PUBLIC_POLICY_ID,
        "rules": RULE_MANIFEST,
        "source_composite": source_composite,
    }
    composite = _json_hash(public_identity_payload)
    if composite != audit["composite_effective_sha256"]:
        raise RuntimeError("compact public-router composite differs from qualifying CUDA run")

    routed = {
        str(update): _head_states(candidates[update])
        for update in ROUTED_UPDATES if update != DEFAULT_UPDATE
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "routed_head_state_dicts": routed,
        "metadata": {
            "policy_id": PUBLIC_POLICY_ID,
            "default_checkpoint_update": DEFAULT_UPDATE,
            "routed_updates": list(ROUTED_UPDATES),
            "rules": RULE_MANIFEST,
            "rules_sha256": _json_hash(RULE_MANIFEST),
            "storage_dtype": "fp16",
            "runtime_dtype": "fp32",
            "deployment_contract": "kaggle_fp16_storage_fp32_runtime_v1",
            "candidate_effective_sha256": effective_ids,
            "head_effective_sha256": head_ids,
            "shared_effective_sha256": shared_id,
            "source_identity_payload": source_identity_payload,
            "source_composite_sha256": source_composite,
            "composite_effective_sha256": composite,
            "qualifying_report_schema": qualifying_report.get("schema_version"),
            "qualifying_benchmark_id": qualifying_report.get("benchmark_id"),
        },
    }
    floating = list(_floating_tensors(payload["routed_head_state_dicts"]))
    if not floating or any(tensor.dtype != torch.float16 for tensor in floating):
        raise RuntimeError("compact routed heads are not wholly FP16 stored")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    strict = torch.load(output, map_location="cpu", weights_only=True)
    if strict.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("compact routed-head schema changed during storage")
    return {
        "schema_version": "0045_public_meta_router_compact_audit_v1",
        "status": "PASS",
        "policy_id": PUBLIC_POLICY_ID,
        "default_portable_checkpoint_sha256": sha256_file(portable_candidates[DEFAULT_UPDATE]),
        "compact_heads_sha256": sha256_file(output),
        "stored_head_updates": sorted(map(int, routed)),
        "shared_effective_sha256": shared_id,
        "head_effective_sha256": head_ids,
        "candidate_effective_sha256": effective_ids,
        "rules_sha256": _json_hash(RULE_MANIFEST),
        "composite_effective_sha256": composite,
        "storage_dtype": "fp16",
        "runtime_dtype": "fp32",
        "deployment_contract": "kaggle_fp16_storage_fp32_runtime_v1",
    }


__all__ = ["HEAD_FIELDS", "SCHEMA_VERSION", "materialize_compact_heads"]
