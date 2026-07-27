from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .policy.actor_critic import DragapultActorCritic


def save_model_only(
    path: Path,
    model: DragapultActorCritic,
    *,
    update: int,
    metadata: dict[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "0017_dragapult_actor_critic_model_only_v1",
        "update": int(update),
        "model": model.state_dict(),
        "metadata": metadata,
    }
    forbidden = {"optimizer", "scheduler", "scaler", "rng_state", "rollout"}
    if forbidden.intersection(payload):
        raise RuntimeError("model-only checkpoint contains resumable training state")
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def prune_model_checkpoints(directory: Path, keep: int = 8) -> list[Path]:
    if keep < 1:
        raise ValueError("checkpoint retention must be positive")
    checkpoints = sorted(
        directory.glob("update-*.pt"), key=lambda path: path.stat().st_mtime
    )
    removed = checkpoints[:-keep]
    for path in removed:
        path.unlink()
    return removed


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if any(key in payload for key in ("optimizer", "scheduler", "scaler", "rng_state")):
        raise ValueError("checkpoint violates model-only contract")
    return {
        "schema_version": payload["schema_version"],
        "update": payload["update"],
        "metadata": json.loads(json.dumps(payload["metadata"])),
    }


__all__ = ["checkpoint_metadata", "prune_model_checkpoints", "save_model_only"]
