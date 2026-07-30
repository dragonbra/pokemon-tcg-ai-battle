"""Atomic, bounded, exact epoch-boundary recovery checkpoints for 0021 BC."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


SCHEMA_VERSION = "0021_resumable_bc_checkpoint_v1"
MAX_RETAINED_CHECKPOINTS = 8
REQUIRED_METADATA = (
    "project_id",
    "version",
    "dataset_content_sha256",
    "feature_compiler_sha256",
    "model_config_sha256",
    "training_config_sha256",
)


@dataclass(frozen=True)
class ResumeState:
    completed_epoch: int
    global_step: int
    best: dict[str, float]
    no_loss_improvement: int
    progress_iteration: int
    gpu_training_seconds: float
    history: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_METADATA if not metadata.get(field)]
    if missing:
        raise ValueError(f"checkpoint metadata missing required fields: {missing}")


def _numpy_rng_state() -> dict[str, Any]:
    bit_generator, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "bit_generator": bit_generator,
        "keys": torch.from_numpy(keys.astype(np.int64, copy=True)),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": _numpy_rng_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _restore_rng_state(payload: Mapping[str, Any]) -> None:
    if set(payload) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("checkpoint RNG payload is incomplete")
    numpy_state = payload["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise ValueError("checkpoint NumPy RNG payload is invalid")
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_state["keys"].cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    random.setstate(payload["python"])
    torch.set_rng_state(payload["torch_cpu"].cpu())
    cuda_states = payload["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG device count mismatch")
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])


def _trainer_payload(
    *,
    completed_epoch: int,
    global_step: int,
    trainer_state: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "best", "no_loss_improvement", "progress_iteration",
        "gpu_training_seconds", "history",
    }
    if not required.issubset(trainer_state):
        raise ValueError("checkpoint trainer state is incomplete")
    if completed_epoch < 1 or global_step < 0:
        raise ValueError("checkpoint must describe a completed epoch boundary")
    return {
        "epoch_boundary": True,
        "completed_epoch": int(completed_epoch),
        "global_step": int(global_step),
        "best": dict(trainer_state["best"]),
        "no_loss_improvement": int(trainer_state["no_loss_improvement"]),
        "progress_iteration": int(trainer_state["progress_iteration"]),
        "gpu_training_seconds": float(trainer_state["gpu_training_seconds"]),
        "history": [dict(record) for record in trainer_state["history"]],
    }


def _prune_unreferenced(root: Path, criteria_root: Path, *, current: str) -> None:
    retained = {current}
    for criterion in criteria_root.glob("*.json"):
        manifest = json.loads(criterion.read_text(encoding="utf-8"))
        retained.add(str(manifest["path"]))
    if len(retained) > MAX_RETAINED_CHECKPOINTS:
        raise RuntimeError("checkpoint criteria exceed the bounded retention contract")
    for checkpoint in root.glob("epoch-*.pt"):
        if checkpoint.name not in retained:
            checkpoint.unlink()
            checkpoint.with_suffix(".json").unlink(missing_ok=True)


def retained_checkpoint_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.glob("epoch-*.pt"))


def save_checkpoint(
    root: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    grad_scaler: Any | None,
    completed_epoch: int,
    global_step: int,
    trainer_state: Mapping[str, Any],
    metadata: dict[str, Any],
    criteria: Sequence[str],
) -> dict[str, Any]:
    _validate_metadata(metadata)
    root.mkdir(parents=True, exist_ok=True)
    criteria_root = root / "criteria"
    criteria_root.mkdir(exist_ok=True)
    temporary = root / f".epoch-{completed_epoch:04d}.{os.getpid()}.tmp"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "grad_scaler": grad_scaler.state_dict() if grad_scaler is not None else None,
        "trainer": _trainer_payload(
            completed_epoch=completed_epoch,
            global_step=global_step,
            trainer_state=trainer_state,
        ),
        "rng": _capture_rng_state(),
        "metadata": dict(metadata),
    }
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        digest = _sha256(temporary)
        final = root / f"epoch-{completed_epoch:04d}-{digest[:16]}.pt"
        if final.exists():
            raise FileExistsError(f"checkpoint already exists: {final}")
        temporary.replace(final)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "path": final.name,
        "sha256": digest,
        "completed_epoch": completed_epoch,
        "global_step": global_step,
        "metadata": metadata,
        "criteria": list(criteria),
        "optimizer_state_saved": True,
        "scheduler_state_saved": scheduler is not None,
        "grad_scaler_state_saved": grad_scaler is not None,
        "rng_state_saved": True,
        "resumable_training_state_saved": True,
        "resume_boundary": "completed_epoch",
    }
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    _write_exclusive(final.with_suffix(".json"), manifest_bytes)
    for criterion in criteria:
        _write_atomic(criteria_root / f"{criterion}.json", manifest_bytes)
    _prune_unreferenced(root, criteria_root, current=final.name)
    manifest["retained_checkpoint_bytes"] = retained_checkpoint_bytes(root)
    return manifest


def _validate_model_state(model: nn.Module, state: Mapping[str, Any]) -> None:
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("checkpoint model keys mismatch")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape:
            raise ValueError(f"checkpoint model tensor mismatch: {name}")


def _validate_optimizer_state(
    optimizer: torch.optim.Optimizer, state: Mapping[str, Any]
) -> None:
    current_groups = optimizer.state_dict()["param_groups"]
    saved_groups = state.get("param_groups")
    if not isinstance(saved_groups, list) or len(saved_groups) != len(current_groups):
        raise ValueError("checkpoint optimizer parameter-group mismatch")
    if any(
        len(saved["params"]) != len(current["params"])
        for saved, current in zip(saved_groups, current_groups, strict=True)
    ):
        raise ValueError("checkpoint optimizer parameter-group mismatch")


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    grad_scaler: Any | None,
    expected_metadata: Mapping[str, Any],
) -> ResumeState:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "schema_version", "model", "optimizer", "scheduler", "grad_scaler",
        "trainer", "rng", "metadata",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("checkpoint payload fields are invalid")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("checkpoint schema mismatch")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata is invalid")
    _validate_metadata(metadata)
    mismatches = {
        field: (metadata.get(field), expected_metadata.get(field))
        for field in REQUIRED_METADATA
        if metadata.get(field) != expected_metadata.get(field)
    }
    if mismatches:
        raise ValueError(f"checkpoint metadata mismatch: {mismatches}")
    trainer = payload["trainer"]
    if not isinstance(trainer, dict) or trainer.get("epoch_boundary") is not True:
        raise ValueError("checkpoint is not a completed epoch boundary")
    if (payload["scheduler"] is None) != (scheduler is None):
        raise ValueError("checkpoint scheduler presence mismatch")
    if (payload["grad_scaler"] is None) != (grad_scaler is None):
        raise ValueError("checkpoint GradScaler presence mismatch")
    _validate_model_state(model, payload["model"])
    _validate_optimizer_state(optimizer, payload["optimizer"])
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if grad_scaler is not None:
        grad_scaler.load_state_dict(payload["grad_scaler"])
    _restore_rng_state(payload["rng"])
    return ResumeState(
        completed_epoch=int(trainer["completed_epoch"]),
        global_step=int(trainer["global_step"]),
        best={str(key): float(value) for key, value in trainer["best"].items()},
        no_loss_improvement=int(trainer["no_loss_improvement"]),
        progress_iteration=int(trainer["progress_iteration"]),
        gpu_training_seconds=float(trainer["gpu_training_seconds"]),
        history=tuple(dict(record) for record in trainer["history"]),
        metadata=metadata,
    )


__all__ = [
    "MAX_RETAINED_CHECKPOINTS",
    "REQUIRED_METADATA",
    "ResumeState",
    "SCHEMA_VERSION",
    "load_checkpoint",
    "retained_checkpoint_bytes",
    "save_checkpoint",
]
