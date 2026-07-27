"""Immutable content-addressed training checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload);handle.flush();os.fsync(handle.fileno())


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


def save_checkpoint(root: Path, *, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, global_step: int, metadata: dict[str, Any], criteria: Sequence[str]) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True);criteria_root=root/"criteria";criteria_root.mkdir(exist_ok=True)
    temporary=root/f".epoch-{epoch:04d}.tmp"
    torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,"global_step":global_step,"metadata":metadata,"rng_state":_rng_state()},temporary)
    digest=hashlib.sha256(temporary.read_bytes()).hexdigest();final=root/f"epoch-{epoch:04d}-{digest[:16]}.pt"
    if final.exists():temporary.unlink();raise FileExistsError(f"checkpoint already exists: {final}")
    temporary.replace(final)
    manifest={"schema_version":"semantic_goal_checkpoint_v1","path":final.name,"sha256":digest,"epoch":epoch,"global_step":global_step,"metadata":metadata,"criteria":list(criteria)}
    manifest_bytes=(json.dumps(manifest,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode();_write_exclusive(final.with_suffix(".json"),manifest_bytes)
    for criterion in criteria:_write_atomic(criteria_root/f"{criterion}.json",manifest_bytes)
    return manifest


__all__=["save_checkpoint"]
