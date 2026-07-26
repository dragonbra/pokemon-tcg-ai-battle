"""Formal epoch trainer with online train metrics and snapshot validation."""
from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from rl_environment.logging import TrainingLogger
from rl_environment.runs import VersionPaths, write_version_status
from .checkpoints import save_checkpoint
from .metrics import evaluate_full_pass, teacher_forced_batch_metrics
from .runtime_batch import trim_target_padding

Batch = Mapping[str, Tensor]
_STAGE_CODE = {"train": 1, "validation_eval": 3}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _device_batch(batch: Batch, device: torch.device) -> dict[str, Tensor]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in trim_target_padding(batch).items()
    }


class _PhaseProgress:
    def __init__(
        self,
        *,
        logger: TrainingLogger,
        epoch: int,
        stage: str,
        total_batches: int,
        progress_iteration: list[int],
        trainer_update: Callable[[], int],
        log_every: int,
        device: torch.device,
    ) -> None:
        self.logger = logger
        self.epoch = epoch
        self.stage = stage
        self.total_batches = total_batches
        self.progress_iteration = progress_iteration
        self.trainer_update = trainer_update
        self.log_every = log_every
        self.device = device
        self.started = time.perf_counter()
        self.last_decisions = 0
        self.last_loss: float | None = None
        self.bar = tqdm(
            total=total_batches,
            desc=f"epoch {epoch} {stage}",
            unit="batch",
            dynamic_ncols=True,
            mininterval=0.5,
        )

    def update(
        self,
        batch_index: int,
        decisions: int,
        *,
        loss: float | Tensor | None = None,
    ) -> None:
        increment = batch_index - self.bar.n
        if increment > 0:
            self.bar.update(increment)
        self.last_decisions = decisions
        should_log = batch_index % self.log_every == 0 or batch_index == self.total_batches
        if should_log and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if should_log and loss is not None:
            self.last_loss = float(loss.detach().item()) if isinstance(loss, Tensor) else loss
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        postfix: dict[str, str] = {
            "iter/s": f"{batch_index / elapsed:.2f}",
            "dec/s": f"{decisions / elapsed:.1f}",
        }
        if self.last_loss is not None:
            postfix["loss"] = f"{self.last_loss:.4f}"
        self.bar.set_postfix(postfix, refresh=False)
        if should_log:
            self.progress_iteration[0] += 1
            metrics: dict[str, Any] = {
                "trainer/epoch": self.epoch,
                "trainer/update": self.trainer_update(),
                "progress/iteration": self.progress_iteration[0],
                "progress/stage_code": _STAGE_CODE[self.stage],
                "progress/completed_batches": batch_index,
                "progress/total_batches": self.total_batches,
                "progress/fraction": min(batch_index / self.total_batches, 1.0),
                "progress/iterations_per_second": batch_index / elapsed,
                "progress/decisions_per_second": decisions / elapsed,
                "progress/elapsed_seconds": elapsed,
            }
            if self.last_loss is not None:
                metrics["progress/running_loss"] = self.last_loss
            self.logger.log(self.progress_iteration[0], metrics)

    def close(self) -> None:
        self.bar.close()


def train_version(
    paths: VersionPaths,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_batches: Callable[[int], Iterable[Batch]],
    validation_batches: Callable[[int], Iterable[Batch]],
    batch_counts: Mapping[str, int],
    epochs: int,
    config: dict[str, Any],
    device: torch.device,
    max_grad_norm: float = 1.0,
    amp: bool = True,
    progress_log_every: int = 20,
) -> dict[str, Any]:
    if paths.config.exists() or paths.metrics.exists() or any(paths.checkpoints.iterdir()):
        raise FileExistsError("formal version paths already occupied")
    if epochs <= 0 or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("invalid training epochs or gradient norm")
    if set(batch_counts) != {"train", "validation"} or min(batch_counts.values()) <= 0:
        raise ValueError("complete train and validation batch counts are required")
    _atomic_json(paths.config, config)
    write_version_status(paths, {
        "state": "training",
        "started_at": time.time(),
        "device": str(device),
        "wandb_expected_mode": "online",
    })
    best_loss = math.inf
    best_exact = -math.inf
    global_step = 0
    progress_iteration = [0]
    history: list[dict[str, Any]] = []
    model.to(device)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    try:
        with TrainingLogger(paths.metrics, paths.tensorboard) as logger:
            logger.log(0, {
                "trainer/epoch": 0,
                "trainer/update": 0,
                "progress/iteration": 0,
                "system/run_started": 1,
                "system/parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
            })
            for epoch in range(1, epochs + 1):
                started = time.perf_counter()
                model.train()
                optimization_loss = torch.zeros((), dtype=torch.float64, device=device)
                online_totals = torch.zeros(4, dtype=torch.float64, device=device)
                optimization_batches = 0
                decisions = 0
                train_progress = _PhaseProgress(
                    logger=logger,
                    epoch=epoch,
                    stage="train",
                    total_batches=batch_counts["train"],
                    progress_iteration=progress_iteration,
                    trainer_update=lambda: global_step,
                    log_every=progress_log_every,
                    device=device,
                )
                try:
                    for source_batch in train_batches(epoch):
                        batch = _device_batch(source_batch, device)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.amp.autocast(
                            autocast_device,
                            enabled=amp and device.type == "cuda",
                            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                        ):
                            batch_metrics = teacher_forced_batch_metrics(model, batch)
                            loss = batch_metrics.loss
                        if not torch.isfinite(loss):
                            raise FloatingPointError("nonfinite training loss")
                        loss.backward()
                        gradient_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_grad_norm
                        )
                        if not torch.isfinite(gradient_norm):
                            raise FloatingPointError("nonfinite gradient norm")
                        optimizer.step()
                        optimization_loss += loss.detach().to(torch.float64)
                        online_totals += torch.stack(
                            (
                                batch_metrics.tokens,
                                batch_metrics.token_correct,
                                batch_metrics.actions,
                                batch_metrics.exact_actions,
                            )
                        ).to(torch.float64)
                        optimization_batches += 1
                        decisions += int(batch["state_num"].size(0))
                        global_step += 1
                        train_progress.update(
                            optimization_batches,
                            decisions,
                            loss=optimization_loss / optimization_batches,
                        )
                finally:
                    train_progress.close()
                if optimization_batches != batch_counts["train"]:
                    raise ValueError("training epoch batch count mismatch")

                def evaluate_validation() -> dict[str, float]:
                    progress = _PhaseProgress(
                        logger=logger,
                        epoch=epoch,
                        stage="validation_eval",
                        total_batches=batch_counts["validation"],
                        progress_iteration=progress_iteration,
                        trainer_update=lambda: global_step,
                        log_every=progress_log_every,
                        device=device,
                    )
                    try:
                        return evaluate_full_pass(
                            model,
                            validation_batches(epoch),
                            device=device,
                            namespace="validation",
                            amp=amp,
                            progress=lambda batch_index, count: progress.update(
                                batch_index, count
                            ),
                        )
                    finally:
                        progress.close()

                (
                    online_tokens,
                    online_token_correct,
                    online_actions,
                    online_exact_actions,
                ) = online_totals.cpu().tolist()
                mean_optimization_loss = float(
                    (optimization_loss / optimization_batches).cpu()
                )
                validation_metrics = evaluate_validation()
                metrics = {
                    "trainer/epoch": epoch,
                    "trainer/update": global_step,
                    "progress/iteration": progress_iteration[0],
                    "bc/optimization/loss": mean_optimization_loss,
                    "bc/optimization/token_accuracy": (
                        online_token_correct / online_tokens
                    ),
                    "bc/optimization/teacher_exact_action": (
                        online_exact_actions / online_actions
                    ),
                    "bc/optimization/decisions": online_actions,
                    "bc/optimization/tokens": online_tokens,
                    **validation_metrics,
                    "system/epoch_seconds": time.perf_counter() - started,
                    "system/parameter_count": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "system/global_step": global_step,
                    "system/amp_scale": 1.0,
                }
                logger.log(epoch, metrics)
                criteria = ["latest"]
                validation_loss = metrics["bc/validation/loss"]
                validation_exact = metrics["bc/validation/exact_action"]
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    criteria.append("best_validation_loss")
                if validation_exact > best_exact:
                    best_exact = validation_exact
                    criteria.append("best_validation_exact")
                manifest = save_checkpoint(
                    paths.checkpoints,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    metadata={**config, "metrics": metrics},
                    criteria=criteria,
                )
                history.append({**metrics, "checkpoint": manifest})
        summary = {
            "state": "complete",
            "epochs": epochs,
            "global_step": global_step,
            "best_validation_loss": best_loss,
            "best_validation_exact": best_exact,
            "history": history,
        }
        _atomic_json(paths.summary, summary)
        write_version_status(paths, {
            "state": "complete",
            "completed_at": time.time(),
            "best_validation_loss": best_loss,
            "best_validation_exact": best_exact,
        })
        return summary
    except BaseException as error:
        write_version_status(paths, {
            "state": "failed",
            "reason": f"{type(error).__name__}: {error}",
            "failed_at": time.time(),
        })
        raise


__all__ = ["train_version"]
