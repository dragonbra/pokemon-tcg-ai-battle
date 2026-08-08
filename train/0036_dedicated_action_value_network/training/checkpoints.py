"""Atomic model-only Value checkpoint storage."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = "0036_value_model_only_checkpoint_v1"
FORBIDDEN_KEYS = frozenset({"optimizer", "scheduler", "scaler", "rng", "rollout", "replay"})


def save_value_checkpoint(path: Path, value_head: torch.nn.Module, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "value_head_state_dict": {key: value.detach().cpu() for key, value in value_head.state_dict().items()},
        "metadata": metadata,
    }
    if FORBIDDEN_KEYS.intersection(payload) or FORBIDDEN_KEYS.intersection(metadata):
        raise ValueError("0036 model-only checkpoint contains recovery state")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_value_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if set(payload) != {"schema_version", "value_head_state_dict", "metadata"} or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("invalid 0036 model-only checkpoint")
    if FORBIDDEN_KEYS.intersection(payload) or FORBIDDEN_KEYS.intersection(payload["metadata"]):
        raise ValueError("0036 checkpoint contains forbidden recovery state")
    return payload


__all__ = ["SCHEMA_VERSION", "load_value_checkpoint", "save_value_checkpoint"]
