"""Auditable BC trainer for the 0032 audited semantic policy."""

from __future__ import annotations

import json
import hashlib
import math
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from rl_environment.logging import TrainingLogger
from rl_environment.runs import VersionPaths, write_version_status

from .checkpoints import save_checkpoint
from .objective import evaluate_audited_batches, evaluate_batches, teacher_batch
from .prefetch import PrefetchIterator


class DecisionBatchDataset(Protocol):
    manifest: dict[str, Any]
    manifest_sha256: str
    split_counts: dict[str, int]

    def batch_count(self, split: str, batch_size: int) -> int: ...

    def iter_batches(
        self, split: str, batch_size: int, *, seed: int
    ) -> Any: ...

    def iter_audited_batches(
        self, split: str, batch_size: int, *, seed: int
    ) -> Any: ...


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_jsonl(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _device_batch(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def _train_epoch(
    arm: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: DecisionBatchDataset,
    *,
    epoch: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    amp: bool,
    grad_clip: float,
    maximum_batches: int | None,
    prefetch_depth: int,
) -> tuple[dict[str, float], int]:
    model.train()
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    tokens = torch.zeros((), dtype=torch.float64, device=device)
    correct = torch.zeros((), dtype=torch.float64, device=device)
    exact = torch.zeros((), dtype=torch.float64, device=device)
    decisions = 0
    updates = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total = dataset.batch_count("train", batch_size)
    if maximum_batches is not None:
        total = min(total, maximum_batches)
    source_batches = dataset.iter_batches("train", batch_size, seed=seed + epoch)
    prefetch = (
        PrefetchIterator(source_batches, depth=prefetch_depth)
        if prefetch_depth > 0
        else None
    )
    iterator = tqdm(
        prefetch if prefetch is not None else source_batches,
        total=total,
        desc=f"0032 {arm} epoch {epoch} train",
        unit="batch",
        dynamic_ncols=True,
        mininterval=5.0,
        leave=False,
    )
    try:
        for batch_index, source in enumerate(iterator, 1):
            if maximum_batches is not None and batch_index > maximum_batches:
                break
            batch = _device_batch(source, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=amp and device.type == "cuda", dtype=torch.bfloat16
            ):
                result = teacher_batch(model, batch)
            if not torch.isfinite(result.loss):
                raise FloatingPointError(f"nonfinite training loss in arm {arm}")
            result.loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"nonfinite gradient norm in arm {arm}")
            optimizer.step()
            updates += 1
            batch_decisions = int(batch["targets"].size(0))
            decisions += batch_decisions
            loss_sum += result.loss.detach().double() * result.tokens.double()
            tokens += result.tokens.double()
            correct += result.correct.double()
            exact += result.exact.double()
            iterator.set_postfix(
                loss=f"{float(result.loss.detach()):.4f}", refresh=False
            )
    finally:
        iterator.close()
        if prefetch is not None:
            prefetch.close()
    elapsed = time.perf_counter() - started
    if updates == 0 or tokens.item() == 0:
        raise ValueError(f"training arm produced no updates: {arm}")
    data_wait_seconds = prefetch.wait_seconds if prefetch is not None else elapsed
    metrics = {
            "loss": float((loss_sum / tokens).item()),
            "token_accuracy": float((correct / tokens).item()),
            "teacher_exact_action": float((exact / decisions).item()),
            "decisions": decisions,
            "tokens": int(tokens.item()),
            "seconds": elapsed,
            "decisions_per_second": decisions / elapsed,
            "data_wait_seconds": data_wait_seconds,
            "data_wait_fraction": data_wait_seconds / elapsed,
            "prefetch_depth": prefetch_depth,
            "producer_batches": prefetch.produced if prefetch is not None else updates,
        }
    if device.type == "cuda":
        metrics["cuda_peak_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        metrics["cuda_peak_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(device)
        )
    return metrics, updates


def train_ablation(
    *,
    paths: VersionPaths,
    models: Mapping[str, nn.Module],
    optimizers: Mapping[str, torch.optim.Optimizer],
    dataset: DecisionBatchDataset,
    config: dict[str, Any],
    device: torch.device,
    epochs: int,
    batch_size: int,
    validation_batch_size: int,
    seed: int,
    amp: bool,
    grad_clip: float,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    maximum_train_batches: int | None = None,
    maximum_validation_batches: int | None = None,
    prefetch_depth: int = 2,
) -> dict[str, Any]:
    if set(models) != set(optimizers) or not models:
        raise ValueError("models and optimizers must have identical non-empty arms")
    _atomic_json(paths.config, config)
    training_config_sha256 = _sha256(paths.config)
    model_contract_sha256 = (
        _sha256(paths.model_contract) if paths.model_contract.is_file() else None
    )
    write_version_status(
        paths,
        {
            "state": "training",
            "started_at": time.time(),
            "device": str(device),
            "wandb_expected_mode": config["wandb"]["mode"],
        },
    )
    for model in models.values():
        model.to(device)
    best = {
        arm: {"loss": math.inf, "teacher_exact": -math.inf, "greedy_exact": -math.inf}
        for arm in models
    }
    no_improvement = {arm: 0 for arm in models}
    active = {arm: True for arm in models}
    global_step = 0
    history: list[dict[str, Any]] = []
    checkpoint_records: dict[str, dict[str, Any]] = {arm: {} for arm in models}
    try:
        with TrainingLogger(paths.metrics, paths.tensorboard) as logger:
            logger.log(
                0,
                {
                    "trainer/epoch": 0,
                    "trainer/update": 0,
                    "system/run_started": 1,
                    **{
                        f"system/parameter_count/{arm}": sum(
                            parameter.numel() for parameter in model.parameters()
                        )
                        for arm, model in models.items()
                    },
                },
            )
            for epoch in range(1, epochs + 1):
                record: dict[str, Any] = {
                    "trainer/epoch": epoch,
                    "trainer/update": global_step,
                }
                for arm, model in models.items():
                    if not active[arm]:
                        continue
                    train_metrics, updates = _train_epoch(
                        arm,
                        model,
                        optimizers[arm],
                        dataset,
                        epoch=epoch,
                        batch_size=batch_size,
                        seed=seed,
                        device=device,
                        amp=amp,
                        grad_clip=grad_clip,
                        maximum_batches=maximum_train_batches,
                        prefetch_depth=prefetch_depth,
                    )
                    global_step += updates
                    if hasattr(dataset, "iter_audited_batches"):
                        validation_source = dataset.iter_audited_batches(
                            "validation", validation_batch_size, seed=seed
                        )
                        if prefetch_depth > 0:
                            with PrefetchIterator(
                                validation_source, depth=prefetch_depth
                            ) as validation_batches:
                                validation, breakdown = evaluate_audited_batches(
                                    model,
                                    validation_batches,
                                    device=device,
                                    amp=amp,
                                    maximum_batches=maximum_validation_batches,
                                )
                        else:
                            validation, breakdown = evaluate_audited_batches(
                                model,
                                validation_source,
                                device=device,
                                amp=amp,
                                maximum_batches=maximum_validation_batches,
                            )
                    else:
                        validation_source = dataset.iter_batches(
                            "validation", validation_batch_size, seed=seed
                        )
                        if prefetch_depth > 0:
                            with PrefetchIterator(
                                validation_source, depth=prefetch_depth
                            ) as validation_batches:
                                validation = evaluate_batches(
                                    model,
                                    validation_batches,
                                    device=device,
                                    amp=amp,
                                    maximum_batches=maximum_validation_batches,
                                )
                        else:
                            validation = evaluate_batches(
                                model,
                                validation_source,
                                device=device,
                                amp=amp,
                                maximum_batches=maximum_validation_batches,
                            )
                        breakdown = {"by_source": {}, "by_deck": {}}
                    for name, value in train_metrics.items():
                        record[f"bc/{arm}/optimization/{name}"] = value
                    for name, value in validation.items():
                        record[f"bc/{arm}/validation/{name}"] = value
                    criteria = ["latest"]
                    previous_loss = best[arm]["loss"]
                    if validation["loss"] < previous_loss:
                        best[arm]["loss"] = validation["loss"]
                        criteria.append("best_validation_loss")
                    if validation["teacher_exact_action"] > best[arm]["teacher_exact"]:
                        best[arm]["teacher_exact"] = validation["teacher_exact_action"]
                        criteria.append("best_teacher_exact")
                    if validation["exact_action"] > best[arm]["greedy_exact"]:
                        best[arm]["greedy_exact"] = validation["exact_action"]
                        criteria.append("best_greedy_exact")
                    if validation["loss"] <= previous_loss - early_stopping_min_delta:
                        no_improvement[arm] = 0
                    else:
                        no_improvement[arm] += 1
                    metadata = {
                        "project_id": paths.project_id,
                        "version": paths.version_name,
                        "arm": arm,
                        "epoch": epoch,
                        "global_step": global_step,
                        "dataset_manifest_sha256": dataset.manifest_sha256,
                        "training_config_sha256": training_config_sha256,
                        "model_contract_sha256": model_contract_sha256,
                        "implementation_sha256": config.get("implementation_sha256"),
                        "initialized_from_checkpoint": config.get(
                            "initialized_from_checkpoint"
                        ),
                        "model_config": config["arms"][arm],
                        "validation": validation,
                    }
                    arm_root = paths.checkpoints / arm
                    for criterion in criteria:
                        checkpoint_records[arm][criterion] = save_checkpoint(
                            arm_root,
                            name=criterion,
                            model=model,
                            metadata=metadata,
                        )
                    if (
                        early_stopping_patience > 0
                        and no_improvement[arm] >= early_stopping_patience
                    ):
                        active[arm] = False
                        record[f"bc/{arm}/early_stopped"] = 1
                record["trainer/update"] = global_step
                record["system/active_arms"] = sum(active.values())
                logger.log(epoch, record)
                _append_jsonl(
                    paths.artifact / "validation_breakdown.jsonl",
                    {
                        "trainer/epoch": epoch,
                        "trainer/update": global_step,
                        "arms": {arm: breakdown},
                    },
                )
                history.append(record)
                _atomic_json(
                    paths.checkpoint_selection,
                    {
                        "schema_version": "0032_checkpoint_selection_v1",
                        "retention_slots": [
                            "latest",
                            "best_validation_loss",
                            "best_teacher_exact",
                            "best_greedy_exact",
                        ],
                        "arms": checkpoint_records,
                    },
                )
                print(
                    json.dumps({"event": "0032_bc_epoch", **record}, sort_keys=True),
                    flush=True,
                )
                if not any(active.values()):
                    break
        summary = {
            "state": "complete",
            "epochs_completed": int(history[-1]["trainer/epoch"]) if history else 0,
            "epochs_requested": epochs,
            "global_step": global_step,
            "best": best,
            "active_at_completion": active,
            "history": history,
            "checkpoint_retention": checkpoint_records,
        }
        _atomic_json(paths.summary, summary)
        write_version_status(
            paths,
            {
                "state": "completed",
                "completed_at": time.time(),
                "global_step": global_step,
                "best": best,
            },
        )
        return summary
    except BaseException as error:
        write_version_status(
            paths,
            {
                "state": "failed",
                "failed_at": time.time(),
                "reason": f"{type(error).__name__}: {error}",
            },
        )
        raise


__all__ = ["train_ablation"]
