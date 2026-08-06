"""Atomic model-only checkpoint storage for 0034."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


SCHEMA = "0034_pod_cuda_ppo_model_only_v1"
TRAINABLE_HEADS_SCHEMA = "0034_pod_cuda_ppo_trainable_heads_v1"
FORBIDDEN = {"optimizer", "scheduler", "scaler", "rng", "dataloader", "rollout", "buffer"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_model_only(
    model: nn.Module,
    path: Path,
    *,
    update: int,
    metadata: Mapping[str, Any],
) -> str:
    payload = {
        "schema_version": SCHEMA,
        "update": int(update),
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "metadata": dict(metadata),
    }
    if FORBIDDEN & {key.lower() for key in payload}:
        raise ValueError("checkpoint contains forbidden recovery state")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    digest = _sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def load_model_only(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("unsupported 0034 checkpoint schema")
    lowered = {str(key).lower() for key in payload}
    if any(any(token in key for token in FORBIDDEN) for key in lowered):
        raise ValueError("checkpoint contains forbidden recovery state")
    return payload


def save_trainable_heads(
    model: nn.Module,
    path: Path,
    *,
    update: int,
    metadata: Mapping[str, Any],
) -> str:
    state_dict = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.startswith("action_decoder.") or name.startswith("value_head.")
    }
    if not state_dict or not any(name.startswith("action_decoder.") for name in state_dict):
        raise ValueError("model does not expose an action_decoder state")
    if not any(name.startswith("value_head.") for name in state_dict):
        raise ValueError("model does not expose a value_head state")
    payload = {
        "schema_version": TRAINABLE_HEADS_SCHEMA,
        "update": int(update),
        "state_dict": state_dict,
        "metadata": dict(metadata),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    digest = _sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def load_trainable_heads(model: nn.Module, path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != TRAINABLE_HEADS_SCHEMA:
        raise ValueError("unsupported 0034 trainable-head checkpoint schema")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("trainable-head checkpoint has no state_dict")
    if any(
        not (name.startswith("action_decoder.") or name.startswith("value_head."))
        for name in state_dict
    ):
        raise ValueError("trainable-head checkpoint contains a frozen model tensor")
    result = model.load_state_dict(state_dict, strict=False)
    unexpected = list(result.unexpected_keys)
    missing_trainable = [
        name
        for name in result.missing_keys
        if name.startswith("action_decoder.") or name.startswith("value_head.")
    ]
    if unexpected or missing_trainable:
        raise ValueError(
            f"trainable-head checkpoint mismatch: missing={missing_trainable} "
            f"unexpected={unexpected}"
        )
    return payload


__all__ = [
    "SCHEMA",
    "TRAINABLE_HEADS_SCHEMA",
    "load_model_only",
    "load_trainable_heads",
    "save_model_only",
    "save_trainable_heads",
]
