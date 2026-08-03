"""Finite model-only retention plus one exact epoch-resume slot for 0031."""

from __future__ import annotations

import hashlib
import os
import random
import re
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


SCHEMA_VERSION = "0031_model_only_checkpoint_v1"
TRAINING_STATE_SCHEMA_VERSION = "0031_exact_training_state_v1"
_NAME = re.compile(r"[a-z0-9][a-z0-9_]*")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _original_model(model: nn.Module) -> nn.Module:
    """Remove torch.compile wrappers without coupling checkpoints to Dynamo."""
    seen: set[int] = set()
    while isinstance(getattr(model, "_orig_mod", None), nn.Module):
        if id(model) in seen:
            raise ValueError("cyclic compiled model wrapper")
        seen.add(id(model))
        model = model._orig_mod
    return model


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


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
    original = _original_model(model)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "state_dict": {
            key: value.detach().cpu() for key, value in original.state_dict().items()
        },
        "metadata": dict(metadata),
    }
    _atomic_torch_save(final, payload)
    return {
        "path": final.name,
        "sha256": _sha256(final),
        "bytes": final.stat().st_size,
        "name": name,
        "schema_version": SCHEMA_VERSION,
        "optimizer_state_saved": False,
        "resumable_training_state_saved": False,
    }


def save_training_state(
    root: Path,
    *,
    models: dict[str, nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    trainer_state: dict[str, Any],
    compatibility: dict[str, Any],
    schedulers: dict[str, Any] | None = None,
    grad_scaler: Any | None = None,
) -> dict[str, Any]:
    """Atomically replace the single exact-resume slot at an epoch boundary."""
    if not models or set(models) != set(optimizers):
        raise ValueError("resume models and optimizers must have identical arms")
    if schedulers is not None and set(schedulers) != set(models):
        raise ValueError("resume schedulers must have identical arms")
    root.mkdir(parents=True, exist_ok=True)
    final = root / "latest_resume.pt"
    numpy_rng = np.random.get_state()
    payload = {
        "schema_version": TRAINING_STATE_SCHEMA_VERSION,
        "model_state_dicts": {
            arm: {
                key: value.detach().cpu()
                for key, value in _original_model(model).state_dict().items()
            }
            for arm, model in models.items()
        },
        "optimizer_state_dicts": {
            arm: optimizer.state_dict() for arm, optimizer in optimizers.items()
        },
        "scheduler_state_dicts": None
        if schedulers is None
        else {arm: scheduler.state_dict() for arm, scheduler in schedulers.items()},
        "grad_scaler_state_dict": None
        if grad_scaler is None
        else grad_scaler.state_dict(),
        "trainer_state": dict(trainer_state),
        "compatibility": dict(compatibility),
        "rng_state": {
            "python": random.getstate(),
            "numpy": {
                "bit_generator": numpy_rng[0],
                "state": numpy_rng[1].tolist(),
                "position": numpy_rng[2],
                "has_gauss": numpy_rng[3],
                "cached_gaussian": numpy_rng[4],
            },
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        },
    }
    _atomic_torch_save(final, payload)
    return {
        "path": final.name,
        "sha256": _sha256(final),
        "bytes": final.stat().st_size,
        "schema_version": TRAINING_STATE_SCHEMA_VERSION,
        "optimizer_state_saved": True,
        "resumable_training_state_saved": True,
    }


def load_training_state(
    path: Path,
    *,
    models: dict[str, nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    expected_compatibility: dict[str, Any],
    schedulers: dict[str, Any] | None = None,
    grad_scaler: Any | None = None,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Restore an exact-resume slot only under an identical training contract."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema_version") != TRAINING_STATE_SCHEMA_VERSION:
        raise ValueError("unsupported 0031 training-state checkpoint schema")
    if payload.get("compatibility") != expected_compatibility:
        raise ValueError("training-state checkpoint compatibility mismatch")
    if set(models) != set(optimizers) or set(models) != set(payload["model_state_dicts"]):
        raise ValueError("training-state checkpoint arm mismatch")
    if set(models) != set(payload["optimizer_state_dicts"]):
        raise ValueError("training-state optimizer arm mismatch")
    saved_schedulers = payload.get("scheduler_state_dicts")
    if (schedulers is None) != (saved_schedulers is None):
        raise ValueError("training-state scheduler contract mismatch")
    if schedulers is not None and set(schedulers) != set(saved_schedulers):
        raise ValueError("training-state scheduler arm mismatch")
    saved_scaler = payload.get("grad_scaler_state_dict")
    if (grad_scaler is None) != (saved_scaler is None):
        raise ValueError("training-state GradScaler contract mismatch")

    for arm, model in models.items():
        _original_model(model).load_state_dict(payload["model_state_dicts"][arm], strict=True)
        optimizers[arm].load_state_dict(payload["optimizer_state_dicts"][arm])
    if schedulers is not None:
        for arm, scheduler in schedulers.items():
            scheduler.load_state_dict(saved_schedulers[arm])
    if grad_scaler is not None:
        grad_scaler.load_state_dict(saved_scaler)
    random.setstate(payload["rng_state"]["python"])
    numpy_state = payload["rng_state"]["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(payload["rng_state"]["torch_cpu"])
    cuda_state = payload["rng_state"].get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG state cannot be restored without CUDA")
        torch.cuda.set_rng_state_all(cuda_state)
    return dict(payload["trainer_state"])


__all__ = [
    "SCHEMA_VERSION",
    "TRAINING_STATE_SCHEMA_VERSION",
    "load_training_state",
    "save_checkpoint",
    "save_training_state",
]
