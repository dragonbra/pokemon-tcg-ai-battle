from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    state["torch"] = torch.get_rng_state()
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


class CheckpointManager:
    """Save resumable training state and immutable phase checkpoints."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        name: str,
        model: torch.nn.Module,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        step: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        payload: dict[str, Any] = {
            "step": int(step),
            "model": model.state_dict(),
            "metadata": metadata or {},
            "rng_state": _rng_state(),
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        path = self.directory / (name if name.endswith(".pt") else f"{name}.pt")
        torch.save(payload, path)
        return path

    def load(
        self,
        path: str | Path,
        model: torch.nn.Module,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> dict[str, Any]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(payload["model"], strict=strict)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and "scheduler" in payload:
            scheduler.load_state_dict(payload["scheduler"])
        return payload
