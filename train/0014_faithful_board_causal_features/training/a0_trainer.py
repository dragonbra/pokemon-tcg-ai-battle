"""Efficient faithful-0010 BC training with online train and full validation metrics."""
from __future__ import annotations

import importlib
import json
import math
import os
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from rl_environment.logging import TrainingLogger
from rl_environment.runs import VersionPaths, write_version_status

from ..model import Faithful0010PointerPolicy

_checkpoints = importlib.import_module(
    "train.0013_semantic_goal_policy.training.checkpoints"
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    if path.stat().st_size != len(payload.encode("utf-8")):
        raise OSError(f"atomic JSON publication size mismatch: {path}")


def _device_batch(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        name: value.pin_memory().to(device, non_blocking=True)
        for name, value in batch.items()
    }


def _teacher_metrics(
    model: Faithful0010PointerPolicy,
    batch: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    logits = model.teacher_logits(batch)
    mask = batch["targets"] != -100
    loss = F.cross_entropy(logits[mask], batch["targets"][mask])
    correct = logits.argmax(-1).eq(batch["targets"]) & mask
    exact = (correct | ~mask).all(1)
    return loss, mask.sum(), correct.sum(), exact.sum(), logits


def evaluate(
    model: Faithful0010PointerPolicy,
    batches: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    amp: bool,
    maximum_batches: int | None = None,
    progress: Callable[[int, int, float], None] | None = None,
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(8, dtype=torch.float64, device=device)
    with torch.inference_mode():
        for batch_index, source in enumerate(batches, 1):
            if maximum_batches is not None and batch_index > maximum_batches:
                break
            batch = _device_batch(source, device)
            with torch.amp.autocast(
                "cuda",
                enabled=amp and device.type == "cuda",
                dtype=torch.bfloat16,
            ):
                encoded = model.encode(batch)
                logits = model.teacher_logits_from_encoding(batch, encoded)
                mask = batch["targets"] != -100
                loss_sum = F.cross_entropy(
                    logits[mask], batch["targets"][mask], reduction="sum"
                )
                correct = logits.argmax(-1).eq(batch["targets"]) & mask
                teacher_exact = (correct | ~mask).all(1)
                decoded = model.deterministic_action_tensors(batch, encoded=encoded)
                option_width = batch["option_mask"].size(1)
                expected_mask = mask & batch["targets"].lt(option_width)
                expected_lengths = expected_mask.sum(1)
                comparison_width = min(
                    decoded.sequences.size(1), batch["targets"].size(1)
                )
                sequence_equal = (
                    decoded.sequences[:, :comparison_width].eq(
                        batch["targets"][:, :comparison_width]
                    )
                    | ~expected_mask[:, :comparison_width]
                ).all(1)
                greedy_exact = (
                    decoded.legal
                    & decoded.lengths.eq(expected_lengths)
                    & sequence_equal
                )
            totals += torch.stack(
                (
                    loss_sum.double(),
                    mask.sum().double(),
                    correct.sum().double(),
                    torch.tensor(batch["targets"].size(0), device=device).double(),
                    teacher_exact.sum().double(),
                    greedy_exact.sum().double(),
                    decoded.legal.sum().double(),
                    decoded.lengths.eq(expected_lengths).sum().double(),
                )
            )
            if progress is not None:
                progress(batch_index, int(totals[3].item()), float(loss_sum / mask.sum()))
    values = totals.cpu().tolist()
    loss_sum, tokens, token_correct, decisions, teacher_exact, greedy_exact, legal, length = values
    if not tokens or not decisions:
        raise ValueError("validation produced no decisions")
    return {
        "bc/validation/loss": loss_sum / tokens,
        "bc/validation/token_accuracy": token_correct / tokens,
        "bc/validation/teacher_exact_action": teacher_exact / decisions,
        "bc/validation/exact_action": greedy_exact / decisions,
        "bc/validation/legal_action": legal / decisions,
        "bc/validation/action_length_accuracy": length / decisions,
        "bc/validation/decisions": decisions,
        "bc/validation/tokens": tokens,
    }


def train(
    *,
    paths: VersionPaths,
    model: Faithful0010PointerPolicy,
    optimizer: torch.optim.Optimizer,
    train_batches: Callable[[int], Iterable[Mapping[str, Tensor]]],
    validation_batches: Callable[[int], Iterable[Mapping[str, Tensor]]],
    batch_counts: Mapping[str, int],
    epochs: int,
    config: dict[str, Any],
    device: torch.device,
    amp: bool = True,
    grad_clip: float = 1.0,
    maximum_train_batches: int | None = None,
    maximum_validation_batches: int | None = None,
    start_epoch: int = 0,
    initial_global_step: int = 0,
    initial_best: Mapping[str, float] | None = None,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
) -> dict[str, Any]:
    _atomic_json(paths.config, config)
    write_version_status(
        paths,
        {
            "state": "training",
            "started_at": time.time(),
            "device": str(device),
            "wandb_expected_mode": "online",
        },
    )
    model.to(device)
    best = {
        "loss": float((initial_best or {}).get("loss", math.inf)),
        "teacher_exact": float((initial_best or {}).get("teacher_exact", -math.inf)),
        "greedy_exact": float((initial_best or {}).get("greedy_exact", -math.inf)),
    }
    global_step = initial_global_step
    history: list[dict[str, Any]] = []
    no_greedy_improvement = 0
    progress_iteration = 0
    try:
        with TrainingLogger(paths.metrics, paths.tensorboard) as logger:
            logger.log(
                start_epoch,
                {
                    "trainer/epoch": start_epoch,
                    "trainer/update": global_step,
                    "system/run_started": 1,
                    "system/parameter_count": sum(p.numel() for p in model.parameters()),
                },
            )
            for epoch in range(start_epoch + 1, epochs + 1):
                started = time.perf_counter()
                model.train()
                online = torch.zeros(4, dtype=torch.float64, device=device)
                loss_sum = torch.zeros((), dtype=torch.float64, device=device)
                batches_done = 0
                decisions_done = 0
                train_iterator = tqdm(
                    train_batches(epoch),
                    total=(
                        min(batch_counts["train"], maximum_train_batches)
                        if maximum_train_batches is not None
                        else batch_counts["train"]
                    ),
                    desc=f"0014 A0 epoch {epoch}/{epochs} train",
                    unit="batch",
                    dynamic_ncols=True,
                    mininterval=5.0,
                    leave=False,
                )
                for batch_index, source in enumerate(train_iterator, 1):
                    if maximum_train_batches is not None and batch_index > maximum_train_batches:
                        break
                    batch = _device_batch(source, device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(
                        "cuda",
                        enabled=amp and device.type == "cuda",
                        dtype=torch.bfloat16,
                    ):
                        loss, tokens, correct, exact, _ = _teacher_metrics(model, batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite training loss")
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("nonfinite gradient norm")
                    optimizer.step()
                    loss_sum += loss.detach().double()
                    online += torch.stack((tokens, correct, exact, torch.tensor(batch["targets"].size(0), device=device))).double()
                    global_step += 1
                    batches_done += 1
                    decisions_done += batch["targets"].size(0)
                    progress_iteration += 1
                    # Progress is diagnostic rather than a second training
                    # pass.  Keep enough points for a useful W&B progress
                    # curve without flushing TensorBoard/W&B on every batch.
                    if batch_index == 1 or batch_index % 100 == 0:
                        elapsed = max(time.perf_counter() - started, 1e-9)
                        logger.log(
                            progress_iteration,
                            {
                                "trainer/epoch": epoch,
                                "trainer/update": global_step,
                                "progress/iteration": progress_iteration,
                                "progress/stage_code": 1,
                                "progress/completed_batches": batch_index,
                                "progress/total_batches": batch_counts["train"],
                                "progress/fraction": min(
                                    (
                                        (epoch - 1)
                                        + batch_index / max(batch_counts["train"], 1)
                                    )
                                    / max(epochs, 1),
                                    1.0,
                                ),
                                "progress/epoch_fraction": min(
                                    batch_index / max(batch_counts["train"], 1), 1.0
                                ),
                                "progress/decisions_per_second": decisions_done / elapsed,
                                "progress/iterations_per_second": batch_index / elapsed,
                                "progress/elapsed_seconds": elapsed,
                                "progress/running_loss": float(loss_sum / batches_done),
                            },
                            mirror_wandb=False,
                        )
                    if batch_index == 1 or batch_index % 100 == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "0014_a0_train",
                                    "epoch": epoch,
                                    "batch": batch_index,
                                    "batches": batch_counts["train"],
                                    "decisions": decisions_done,
                                    "loss": float(loss.detach()),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    if batch_index == 1 or batch_index % 20 == 0:
                        elapsed = max(time.perf_counter() - started, 1e-9)
                        train_iterator.set_postfix(
                            decisions_s=f"{decisions_done / elapsed:.0f}",
                            loss=f"{float(loss.detach()):.4f}",
                        )
                train_iterator.close()
                training_elapsed = max(time.perf_counter() - started, 1e-9)
                tokens, correct, exact, decisions = online.cpu().tolist()
                validation_started = time.perf_counter()
                def log_validation_progress(
                    completed: int, decisions: int, running_loss: float
                ) -> None:
                    if (
                        completed != 1
                        and completed % 8 != 0
                        and completed != batch_counts["validation"]
                    ):
                        return
                    elapsed = max(time.perf_counter() - validation_started, 1e-9)
                    logger.log(
                        progress_iteration + completed,
                        {
                            "trainer/epoch": epoch,
                            "trainer/update": global_step,
                            "progress/iteration": progress_iteration + completed,
                            "progress/stage_code": 3,
                            "progress/completed_batches": completed,
                            "progress/total_batches": batch_counts["validation"],
                            "progress/fraction": min(
                                (
                                    (epoch - 1)
                                    + completed / max(batch_counts["validation"], 1)
                                )
                                / max(epochs, 1),
                                1.0,
                            ),
                            "progress/epoch_fraction": min(
                                completed / max(batch_counts["validation"], 1), 1.0
                            ),
                            "progress/decisions_per_second": decisions / elapsed,
                            "progress/iterations_per_second": completed / elapsed,
                            "progress/elapsed_seconds": elapsed,
                            "progress/running_loss": running_loss,
                        },
                        mirror_wandb=False,
                    )

                validation = evaluate(
                    model,
                    validation_batches(epoch),
                    device=device,
                    amp=amp,
                    maximum_batches=maximum_validation_batches,
                    progress=(
                        log_validation_progress
                        if maximum_validation_batches is None
                        else None
                    ),
                )
                progress_iteration += batch_counts["validation"]
                metrics = {
                    "trainer/epoch": epoch,
                    "trainer/update": global_step,
                    "bc/optimization/loss": float((loss_sum / batches_done).cpu()),
                    "bc/optimization/token_accuracy": correct / tokens,
                    "bc/optimization/teacher_exact_action": exact / decisions,
                    "bc/optimization/decisions": decisions,
                    "bc/optimization/tokens": tokens,
                    **validation,
                    "system/epoch_seconds": time.perf_counter() - started,
                    "system/train_decisions_per_second": decisions / training_elapsed,
                    "system/train_iterations_per_second": batches_done / training_elapsed,
                    "system/parameter_count": sum(p.numel() for p in model.parameters()),
                    "system/global_step": global_step,
                    "system/amp_scale": 1.0,
                }
                logger.log(epoch, metrics)
                criteria = ["latest"]
                if metrics["bc/validation/loss"] < best["loss"]:
                    best["loss"] = metrics["bc/validation/loss"]
                    criteria.append("best_validation_loss")
                if metrics["bc/validation/teacher_exact_action"] > best["teacher_exact"]:
                    best["teacher_exact"] = metrics["bc/validation/teacher_exact_action"]
                    criteria.append("best_teacher_exact")
                current_greedy = metrics["bc/validation/exact_action"]
                previous_greedy = best["greedy_exact"]
                if current_greedy > previous_greedy:
                    best["greedy_exact"] = current_greedy
                    criteria.append("best_greedy_exact")
                if current_greedy >= previous_greedy + early_stopping_min_delta:
                    no_greedy_improvement = 0
                else:
                    no_greedy_improvement += 1
                checkpoint = _checkpoints.save_checkpoint(
                    paths.checkpoints,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    metadata={**config, "metrics": metrics},
                    criteria=criteria,
                )
                history.append({**metrics, "checkpoint": checkpoint})
                if (
                    early_stopping_patience > 0
                    and no_greedy_improvement >= early_stopping_patience
                ):
                    break
                print(json.dumps({"event": "0014_a0_epoch", **metrics}, sort_keys=True), flush=True)
        summary = {
            "state": "complete",
            "epochs": epochs,
            "global_step": global_step,
            "best_validation_loss": best["loss"],
            "best_teacher_exact": best["teacher_exact"],
            "best_greedy_exact": best["greedy_exact"],
            "history": history,
        }
        _atomic_json(paths.summary, summary)
        _atomic_json(
            paths.checkpoint_selection,
            {
                "best_validation_loss": "minimum validation token cross-entropy",
                "best_teacher_exact": "maximum teacher-forced exact action",
                "best_greedy_exact": "maximum free greedy exact action",
            },
        )
        write_version_status(paths, {"state": "complete", "completed_at": time.time(), **best})
        return summary
    except BaseException as error:
        write_version_status(
            paths,
            {"state": "failed", "failed_at": time.time(), "reason": f"{type(error).__name__}: {error}"},
        )
        raise


__all__ = ["evaluate", "train"]
