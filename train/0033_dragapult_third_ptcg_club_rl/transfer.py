"""Audited, explicit 0031-to-0033 initialization transfer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn


SOURCE_CHECKPOINT_SHA256 = "285b88f5e30c40ad07ad025b429c5bd1594f0359cc0d44849c368056222d63ae"
SOURCE_SCHEMA = "0031_model_only_checkpoint_v1"
SOURCE_PROJECT = "0031_rule_faithful_semantic_foundation_pretraining"
SOURCE_VERSION = "V1_starter_daily_topn_20260729_20260803"
SOURCE_ARM = "rule_faithful_semantic"
SOURCE_EPOCH = 13
SOURCE_GLOBAL_STEP = 109135

EXACT_PREFIX_MAP = (
    ("action_decoder.", "action_decoder."),
    ("state_encoder.board_encoder.", "state_encoder."),
    ("option_encoder.cross_attention_transformer.", "option_encoder."),
)
EXACT_TENSOR_MAP = (
    ("state_encoder.summary_norm.weight", "summary_norm.weight"),
    ("state_encoder.summary_norm.bias", "summary_norm.bias"),
    ("state_encoder.card_parent.weight", "entity_parent_relation.weight"),
    ("state_encoder.card_parent.weight", "entity_child_relation.weight"),
    ("option_encoder.source_relation.weight", "option_source_relation.weight"),
    ("option_encoder.target_relation.weight", "option_target_relation.weight"),
    ("option_encoder.input_norm.weight", "option_norm.weight"),
    ("option_encoder.input_norm.bias", "option_norm.bias"),
)
ROW_SLICE_MAP = (
    ("prototype_encoder.card_identity.weight", "entity_cat_embeddings.fields.0.weight"),
    ("prototype_encoder.card_identity.weight", "option_cat_embeddings.fields.4.weight"),
    ("prototype_encoder.card_identity.weight", "option_cat_embeddings.fields.5.weight"),
    ("prototype_encoder.attack_identity.weight", "option_cat_embeddings.fields.6.weight"),
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expanded_exact_map(source: Mapping[str, Tensor]) -> list[tuple[str, str]]:
    pairs = list(EXACT_TENSOR_MAP)
    for source_prefix, destination_prefix in EXACT_PREFIX_MAP:
        for source_name in sorted(name for name in source if name.startswith(source_prefix)):
            destination_name = destination_prefix + source_name[len(source_prefix) :]
            pairs.append((source_name, destination_name))
    return pairs


def transfer_state_dict(
    model: nn.Module,
    source: Mapping[str, Tensor],
    *,
    source_checkpoint: str,
    source_sha256: str,
) -> dict[str, Any]:
    destination = model.state_dict()
    updated = {name: value.clone() for name, value in destination.items()}
    copied: list[dict[str, Any]] = []
    used_destinations: set[str] = set()

    def reserve(source_name: str, destination_name: str) -> tuple[Tensor, Tensor]:
        if source_name not in source:
            raise KeyError(f"required source tensor is missing: {source_name}")
        if destination_name not in destination:
            raise KeyError(f"required destination tensor is missing: {destination_name}")
        if destination_name in used_destinations:
            raise ValueError(f"duplicate destination transfer: {destination_name}")
        used_destinations.add(destination_name)
        return source[source_name], updated[destination_name]

    for source_name, destination_name in _expanded_exact_map(source):
        source_tensor, destination_tensor = reserve(source_name, destination_name)
        if source_tensor.shape != destination_tensor.shape:
            raise ValueError(
                f"shape mismatch for {source_name} -> {destination_name}: "
                f"{tuple(source_tensor.shape)} != {tuple(destination_tensor.shape)}"
            )
        destination_tensor.copy_(source_tensor.to(dtype=destination_tensor.dtype))
        copied.append(
            {
                "source": source_name,
                "destination": destination_name,
                "mode": "exact",
                "shape": list(destination_tensor.shape),
                "parameters": destination_tensor.numel(),
            }
        )

    for source_name, destination_name in ROW_SLICE_MAP:
        source_tensor, destination_tensor = reserve(source_name, destination_name)
        if source_tensor.ndim != 2 or destination_tensor.ndim != 2:
            raise ValueError(f"row transfer requires rank-2 tensors: {source_name}")
        if source_tensor.shape[1] != destination_tensor.shape[1]:
            raise ValueError(
                f"embedding width mismatch for {source_name} -> {destination_name}"
            )
        rows = min(source_tensor.shape[0], destination_tensor.shape[0])
        destination_tensor[:rows].copy_(source_tensor[:rows].to(dtype=destination_tensor.dtype))
        copied.append(
            {
                "source": source_name,
                "destination": destination_name,
                "mode": "leading_rows",
                "source_shape": list(source_tensor.shape),
                "destination_shape": list(destination_tensor.shape),
                "rows": rows,
                "parameters": rows * destination_tensor.shape[1],
            }
        )

    model.load_state_dict(updated, strict=True)
    total_parameters = sum(value.numel() for value in destination.values())
    copied_parameters = sum(int(row["parameters"]) for row in copied)
    newly_initialized = [
        {
            "destination": name,
            "shape": list(value.shape),
            "parameters": value.numel(),
        }
        for name, value in destination.items()
        if name not in used_destinations
    ]
    return {
        "schema": "0033_weight_transfer_audit_v1",
        "source_checkpoint": source_checkpoint,
        "source_sha256": source_sha256,
        "source_schema": SOURCE_SCHEMA,
        "strict_feature_parity_claimed": False,
        "mapping_policy": "explicit_semantic_map_only_no_shape_matching",
        "copied_tensors": copied,
        "newly_initialized_tensors": newly_initialized,
        "destination_parameter_count": total_parameters,
        "transferred_parameter_count": copied_parameters,
        "transferred_parameter_fraction": copied_parameters / total_parameters,
    }


def load_0031_initialization(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str = SOURCE_CHECKPOINT_SHA256,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"0031 checkpoint hash mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError(f"unsupported source checkpoint schema: {payload.get('schema_version')}")
    metadata = payload.get("metadata")
    expected_metadata = {
        "project_id": SOURCE_PROJECT,
        "version": SOURCE_VERSION,
        "arm": SOURCE_ARM,
        "epoch": SOURCE_EPOCH,
        "global_step": SOURCE_GLOBAL_STEP,
    }
    if not isinstance(metadata, Mapping) or any(
        metadata.get(key) != value for key, value in expected_metadata.items()
    ):
        raise ValueError("0031 checkpoint metadata does not match PT0805 Epoch 13")
    source = payload.get("state_dict")
    if not isinstance(source, Mapping):
        raise ValueError("source checkpoint has no state_dict mapping")
    report = transfer_state_dict(
        model,
        source,
        source_checkpoint=str(checkpoint_path),
        source_sha256=actual_sha256,
    )
    report["source_metadata"] = metadata
    return report


def write_transfer_report(report: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


__all__ = [
    "SOURCE_CHECKPOINT_SHA256",
    "load_0031_initialization",
    "sha256_file",
    "transfer_state_dict",
    "write_transfer_report",
]
