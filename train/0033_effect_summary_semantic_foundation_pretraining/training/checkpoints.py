"""Finite-retention model-only checkpoints for 0033 BC training."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Any

import torch
from torch import nn


SCHEMA_VERSION = "0033_model_only_checkpoint_v1"
_NAME = re.compile(r"[a-z0-9][a-z0-9_]*")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    root: Path,
    *,
    name: str,
    model: nn.Module,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Atomically publish one replaceable retention slot without optimizer state."""
    if _NAME.fullmatch(name) is None:
        raise ValueError(f"invalid checkpoint name: {name}")
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"{name}.pt"
    temporary = root / f".{name}.{uuid.uuid4().hex}.tmp"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "metadata": dict(metadata),
    }
    try:
        torch.save(payload, temporary)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(final)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": final.name,
        "sha256": _sha256(final),
        "bytes": final.stat().st_size,
        "name": name,
        "schema_version": SCHEMA_VERSION,
        "optimizer_state_saved": False,
        "resumable_training_state_saved": False,
    }


__all__ = ["SCHEMA_VERSION", "save_checkpoint"]
