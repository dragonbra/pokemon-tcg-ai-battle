"""Efficient faithful-0010 BC training with online train and full validation metrics."""
from __future__ import annotations

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
from . import checkpoints as _checkpoints


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
    neutral_totals = torch.zeros(8, dtype=torch.float64, device=device)

    def batch_totals(
        batch: dict[str, Tensor], encoded: tuple[Tensor, Tensor]
    ) -> Tensor:
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
        comparison_width = min(decoded.sequences.size(1), batch["targets"].size(1))
        sequence_equal = (
            decoded.sequences[:, :comparison_width].eq(
                batch["targets"][:, :comparison_width]
            )
            | ~expected_mask[:, :comparison_width]
        ).all(1)
        greedy_exact = (
            decoded.legal & decoded.lengths.eq(expected_lengths) & sequence_equal
        )
        return torch.stack(
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
                encode_backbone = getattr(model, "encode_backbone", None)
                condition_encoding = getattr(model, "condition_encoding", None)
                if callable(encode_backbone) and callable(condition_encoding):
                    backbone = encode_backbone(batch)
                    encoded = condition_encoding(backbone, batch["source_id"])
                    neutral = condition_encoding(
                        backbone, torch.zeros_like(batch["source_id"])
                    )
                else:
                    encoded = model.encode(batch)
                    neutral = encoded
                current = batch_totals(batch, encoded)
                neutral_current = batch_totals(batch, neutral)
            totals += current
            neutral_totals += neutral_current
            if progress is not None:
                progress(
                    batch_index,
                    int(totals[3].item()),
                    float(current[0] / current[1]),
                )
    values = totals.cpu().tolist()
    loss_sum, tokens, token_correct, decisions, teacher_exact, greedy_exact, legal, length = values
    if not tokens or not decisions:
        raise ValueError("validation produced no decisions")
    neutral_values = neutral_totals.cpu().tolist()
    neutral_loss, neutral_tokens, neutral_correct, neutral_decisions = neutral_values[:4]
    neutral_teacher, neutral_greedy, neutral_legal, neutral_length = neutral_values[4:]
    return {
        "bc/validation/loss": loss_sum / tokens,
        "bc/validation/token_accuracy": token_correct / tokens,
        "bc/validation/teacher_exact_action": teacher_exact / decisions,
        "bc/validation/exact_action": greedy_exact / decisions,
        "bc/validation/legal_action": legal / decisions,
        "bc/validation/action_length_accuracy": length / decisions,
        "bc/validation/decisions": decisions,
        "bc/validation/tokens": tokens,
        "bc/validation_neutral/loss": neutral_loss / neutral_tokens,
        "bc/validation_neutral/token_accuracy": neutral_correct / neutral_tokens,
        "bc/validation_neutral/teacher_exact_action": neutral_teacher / neutral_decisions,
        "bc/validation_neutral/exact_action": neutral_greedy / neutral_decisions,
        "bc/validation_neutral/legal_action": neutral_legal / neutral_decisions,
        "bc/validation_neutral/action_length_accuracy": neutral_length / neutral_decisions,
        "bc/validation_neutral/decisions": neutral_decisions,
        "bc/validation_neutral/tokens": neutral_tokens,
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
    initial_no_loss_improvement: int = 0,
    initial_progress_iteration: int = 0,
    initial_history: Iterable[Mapping[str, Any]] = (),
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
    optimization_step: Callable[
        [Faithful0010PointerPolicy, dict[str, Tensor]],
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Mapping[str, Tensor]],
    ]
    | None = None,
) -> dict[str, Any]:
    _atomic_json(paths.config, config)
    lifecycle = {
        "state": "training",
        "device": str(device),
        "wandb_expected_mode": "online",
    }
    if start_epoch:
        lifecycle.update({"resumed_at": time.time(), "resume_start_epoch": start_epoch})
    else:
        lifecycle["started_at"] = time.time()
    write_version_status(paths, lifecycle)
    model.to(device)
    best = {
        "loss": float((initial_best or {}).get("loss", math.inf)),
        "teacher_exact": float((initial_best or {}).get("teacher_exact", -math.inf)),
        "greedy_exact": float((initial_best or {}).get("greedy_exact", -math.inf)),
    }
    global_step = initial_global_step
    history = [dict(record) for record in initial_history]
    no_loss_improvement = initial_no_loss_improvement
    progress_iteration = initial_progress_iteration
    try:
        with TrainingLogger(paths.metrics, paths.tensorboard) as logger:
            logger.log(
                start_epoch,
                {
                    "trainer/epoch": start_epoch,
                    "trainer/update": global_step,
                    "system/run_resumed" if start_epoch else "system/run_started": 1,
                    "system/parameter_count": sum(p.numel() for p in model.parameters()),
                },
            )
            for epoch in range(start_epoch + 1, epochs + 1):
                started = time.perf_counter()
                model.train()
                online = torch.zeros(4, dtype=torch.float64, device=device)
                loss_sum = torch.zeros((), dtype=torch.float64, device=device)
                optimization_sums: dict[str, Tensor] = {}
                batches_done = 0
                decisions_done = 0
                train_iterator = tqdm(
                    train_batches(epoch),
                    total=(
                        min(batch_counts["train"], maximum_train_batches)
                        if maximum_train_batches is not None
                        else batch_counts["train"]
                    ),
                    desc=f"0019 BC epoch {epoch}/{epochs} train",
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
                        if optimization_step is None:
                            loss, tokens, correct, exact, _ = _teacher_metrics(model, batch)
                            batch_optimization: Mapping[str, Tensor] = {}
                        else:
                            (
                                loss,
                                tokens,
                                correct,
                                exact,
                                _,
                                batch_optimization,
                            ) = optimization_step(model, batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite training loss")
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("nonfinite gradient norm")
                    optimizer.step()
                    loss_sum += loss.detach().double()
                    for name, value in batch_optimization.items():
                        detached = value.detach().double()
                        optimization_sums[name] = optimization_sums.get(
                            name, torch.zeros((), dtype=torch.float64, device=device)
                        ) + detached
                    batch_size = torch.tensor(batch["targets"].size(0), device=device)
                    online += torch.stack((tokens, correct, exact, batch_size)).double()
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
                                    "event": "0019_bc_train",
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
                optimization_loss = float((loss_sum / batches_done).cpu())
                weighted_numerator = optimization_sums.get("weighted_loss_numerator")
                weighted_denominator = optimization_sums.get("weighted_loss_denominator")
                if weighted_numerator is not None and weighted_denominator is not None:
                    optimization_loss = float(
                        (weighted_numerator / weighted_denominator.clamp_min(1e-8)).cpu()
                    )
                extra_optimization_metrics: dict[str, float] = {}
                weight_sum = optimization_sums.get("decision_weight_sum")
                if weight_sum is not None:
                    extra_optimization_metrics["bc/optimization/mean_decision_weight"] = float(
                        (weight_sum / decisions).cpu()
                    )
                for outcome in ("win", "loss", "draw"):
                    count = optimization_sums.get(f"{outcome}_decisions")
                    if count is not None:
                        extra_optimization_metrics[
                            f"bc/optimization/{outcome}_decisions"
                        ] = float(count.cpu())
                metrics = {
                    "trainer/epoch": epoch,
                    "trainer/update": global_step,
                    "bc/optimization/loss": optimization_loss,
                    **extra_optimization_metrics,
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
                current_loss = metrics["bc/validation/loss"]
                previous_loss = best["loss"]
                if current_loss < previous_loss:
                    best["loss"] = current_loss
                    criteria.append("best_validation_loss")
                if metrics["bc/validation/teacher_exact_action"] > best["teacher_exact"]:
                    best["teacher_exact"] = metrics["bc/validation/teacher_exact_action"]
                    criteria.append("best_teacher_exact")
                current_greedy = metrics["bc/validation/exact_action"]
                previous_greedy = best["greedy_exact"]
                if current_greedy > previous_greedy:
                    best["greedy_exact"] = current_greedy
                    criteria.append("best_greedy_exact")
                if current_loss <= previous_loss - early_stopping_min_delta:
                    no_loss_improvement = 0
                else:
                    no_loss_improvement += 1
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
                    and no_loss_improvement >= early_stopping_patience
                ):
                    break
                print(json.dumps({"event": "0019_bc_epoch", **metrics}, sort_keys=True), flush=True)
        summary = {
            "state": "complete",
            "epochs": int(history[-1]["trainer/epoch"]) if history else start_epoch,
            "requested_epochs": epochs,
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
                "latest": "final completed epoch",
                "best_validation_loss": "minimum validation token cross-entropy",
                "best_teacher_exact": "maximum teacher-forced exact action",
                "best_greedy_exact": "maximum free greedy exact action",
            },
        )
        write_version_status(paths, {"state": "completed", "completed_at": time.time(), **best})
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


__all__ = ["evaluate", "train"]
