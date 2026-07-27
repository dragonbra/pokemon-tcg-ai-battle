"""Strict recovery of one interrupted 0016 training version."""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class ResumeContext:
    payload: dict[str, Any]
    checkpoint_sha256: str
    start_epoch: int
    global_step: int
    best: dict[str, float]
    no_loss_improvement: int
    progress_iteration: int
    history: list[dict[str, Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_records(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if b"\0" in raw or (raw and not raw.endswith(b"\n")):
        raise ValueError("canonical metrics contain an interrupted trailing write")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"metrics line {line_number} is not a mapping")
        records.append(value)
    return records


def load_resume_context(
    checkpoint: Path,
    *,
    checkpoint_root: Path,
    metrics_path: Path,
    expected_version: str,
    expected_dataset_sha256: str,
    early_stopping_min_delta: float,
) -> ResumeContext:
    resolved = checkpoint.resolve()
    if resolved.parent != checkpoint_root.resolve() or not resolved.is_file():
        raise ValueError("resume checkpoint must belong to the requested version")
    manifest_path = resolved.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_sha256 = _sha256(resolved)
    if manifest.get("sha256") != actual_sha256 or manifest.get("path") != resolved.name:
        raise ValueError("resume checkpoint manifest commitment mismatch")
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("resume checkpoint payload is not a mapping")
    start_epoch = int(payload.get("epoch", 0))
    global_step = int(payload.get("global_step", 0))
    if start_epoch < 1 or global_step < 1:
        raise ValueError("resume checkpoint lacks a completed epoch/global step")
    if manifest.get("epoch") != start_epoch or manifest.get("global_step") != global_step:
        raise ValueError("resume checkpoint manifest epoch/global step mismatch")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("resume checkpoint metadata is absent")
    if metadata.get("version") != expected_version:
        raise ValueError("resume checkpoint belongs to another version")
    if metadata.get("dataset_content_sha256") != expected_dataset_sha256:
        raise ValueError("resume checkpoint belongs to another dataset")

    records = _metric_records(metrics_path)
    completed = {
        int(record["trainer/epoch"]): record
        for record in records
        if "bc/validation/loss" in record
        and int(record.get("trainer/epoch", 0)) <= start_epoch
    }
    if sorted(completed) != list(range(1, start_epoch + 1)):
        raise ValueError("canonical metrics do not contain every completed epoch")
    if int(completed[start_epoch].get("trainer/update", -1)) != global_step:
        raise ValueError("checkpoint global step disagrees with canonical metrics")

    best = {"loss": math.inf, "teacher_exact": -math.inf, "greedy_exact": -math.inf}
    no_loss_improvement = 0
    for epoch in range(1, start_epoch + 1):
        record = completed[epoch]
        current_loss = float(record["bc/validation/loss"])
        previous_loss = best["loss"]
        best["loss"] = min(best["loss"], current_loss)
        best["teacher_exact"] = max(
            best["teacher_exact"], float(record["bc/validation/teacher_exact_action"])
        )
        best["greedy_exact"] = max(
            best["greedy_exact"], float(record["bc/validation/exact_action"])
        )
        if current_loss <= previous_loss - early_stopping_min_delta:
            no_loss_improvement = 0
        else:
            no_loss_improvement += 1

    manifests: dict[int, dict[str, Any]] = {}
    for path in checkpoint_root.glob("epoch-*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        epoch = int(value.get("epoch", 0))
        if 1 <= epoch <= start_epoch:
            if epoch in manifests:
                raise ValueError(f"multiple checkpoint manifests for epoch {epoch}")
            manifests[epoch] = value
    if sorted(manifests) != list(range(1, start_epoch + 1)):
        raise ValueError("checkpoint history is incomplete before resume")
    history = [
        {**completed[epoch], "checkpoint": manifests[epoch]}
        for epoch in range(1, start_epoch + 1)
    ]
    progress_iteration = max(
        (
            int(record["progress/iteration"])
            for record in records
            if int(record.get("trainer/epoch", 0)) <= start_epoch
            and "progress/iteration" in record
        ),
        default=0,
    )
    return ResumeContext(
        payload=payload,
        checkpoint_sha256=actual_sha256,
        start_epoch=start_epoch,
        global_step=global_step,
        best=best,
        no_loss_improvement=no_loss_improvement,
        progress_iteration=progress_iteration,
        history=history,
    )


def restore_rng_state(payload: dict[str, Any]) -> bool:
    state = payload.get("rng_state")
    if not isinstance(state, dict) or not state:
        return False
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
    return True


__all__ = ["ResumeContext", "load_resume_context", "restore_rng_state"]
