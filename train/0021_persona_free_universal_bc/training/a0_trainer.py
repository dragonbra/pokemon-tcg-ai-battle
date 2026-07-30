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


def per_decision_sequence_nll(token_nll: Tensor, mask: Tensor) -> Tensor:
    """Average token NLL within each decision before the batch reduction."""
    if token_nll.shape != mask.shape:
        raise ValueError("token NLL and target mask shapes disagree")
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("decision has no supervised target")
    return (token_nll * mask).sum(dim=1) / counts


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
    mask = batch["target_mask"]
    token_nll = F.cross_entropy(
        logits.transpose(1, 2),
        batch["targets"],
        ignore_index=-100,
        reduction="none",
    )
    loss = per_decision_sequence_nll(token_nll, mask).mean()
    correct = logits.argmax(-1).eq(batch["targets"]) & mask
    exact = (correct | ~mask).all(1)
    return loss, mask.sum(), correct.sum(), exact.sum(), logits


def evaluate(
    model: Faithful0010PointerPolicy,
    batches: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    amp: bool,
    namespace: str,
    maximum_batches: int | None = None,
    progress: Callable[[int, int, float], None] | None = None,
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(8, dtype=torch.float64, device=device)
    def batch_totals(
        batch: dict[str, Tensor], encoded: tuple[Tensor, Tensor]
    ) -> Tensor:
        logits = model.teacher_logits_from_encoding(batch, encoded)
        mask = batch["target_mask"]
        token_nll = F.cross_entropy(
            logits.transpose(1, 2),
            batch["targets"],
            ignore_index=-100,
            reduction="none",
        )
        loss_sum = per_decision_sequence_nll(token_nll, mask).sum()
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
                encoded = model.encode(batch)
                current = batch_totals(batch, encoded)
            totals += current
            if progress is not None:
                progress(
                    batch_index,
                    int(totals[3].item()),
                    float(current[0] / current[3]),
                )
    values = totals.cpu().tolist()
    loss_sum, tokens, token_correct, decisions, teacher_exact, greedy_exact, legal, length = values
    if not tokens or not decisions:
        raise ValueError("validation produced no decisions")
    return {
        f"{namespace}/loss": loss_sum / decisions,
        f"{namespace}/token_accuracy": token_correct / tokens,
        f"{namespace}/teacher_exact_action": teacher_exact / decisions,
        f"{namespace}/exact_action": greedy_exact / decisions,
        f"{namespace}/legal_action": legal / decisions,
        f"{namespace}/action_length_accuracy": length / decisions,
        f"{namespace}/decisions": decisions,
        f"{namespace}/tokens": tokens,
    }


def train(
    *,
    paths: VersionPaths,
    model: Faithful0010PointerPolicy,
    optimizer: torch.optim.Optimizer,
    train_batches: Callable[[int], Iterable[Mapping[str, Tensor]]],
    validation_batches: Callable[[int], Iterable[Mapping[str, Tensor]]],
    deck_ood_validation_batches: Callable[[int], Iterable[Mapping[str, Tensor]]],
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
    initial_gpu_training_seconds: float = 0.0,
    initial_history: Iterable[Mapping[str, Any]] = (),
    scheduler: Any | None = None,
    grad_scaler: Any | None = None,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
    minimum_gpu_training_seconds: float = 0.0,
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
        "deck_ood_greedy_exact": float(
            (initial_best or {}).get("deck_ood_greedy_exact", -math.inf)
        ),
    }
    global_step = initial_global_step
    history = [dict(record) for record in initial_history]
    no_loss_improvement = initial_no_loss_improvement
    progress_iteration = initial_progress_iteration
    gpu_training_seconds = initial_gpu_training_seconds
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
                gradient_norm_sum = torch.zeros((), dtype=torch.float64, device=device)
                batches_done = 0
                decisions_done = 0
                train_iterator = tqdm(
                    train_batches(epoch),
                    total=(
                        min(batch_counts["train"], maximum_train_batches)
                        if maximum_train_batches is not None
                        else batch_counts["train"]
                    ),
                    desc=f"0021 BC epoch {epoch}/{epochs} train",
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
                    gradient_norm_sum += norm.detach().double()
                    optimizer.step()
                    batch_size = torch.tensor(batch["targets"].size(0), device=device)
                    loss_sum += loss.detach().double() * batch_size
                    for name, value in batch_optimization.items():
                        detached = value.detach().double()
                        optimization_sums[name] = optimization_sums.get(
                            name, torch.zeros((), dtype=torch.float64, device=device)
                        ) + detached
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
                                "progress/gradient_norm": float(norm.detach()),
                                "progress/learning_rate": optimizer.param_groups[0]["lr"],
                                "system/gpu/memory_allocated_bytes": (
                                    torch.cuda.memory_allocated(device)
                                    if device.type == "cuda"
                                    else 0
                                ),
                                "system/gpu/memory_reserved_bytes": (
                                    torch.cuda.memory_reserved(device)
                                    if device.type == "cuda"
                                    else 0
                                ),
                            },
                        )
                    if batch_index == 1 or batch_index % 100 == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "0021_bc_train",
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
                        and completed != batch_counts["validation_iid"]
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
                            "progress/total_batches": batch_counts["validation_iid"],
                            "progress/fraction": min(
                                (
                                    (epoch - 1)
                                    + completed / max(batch_counts["validation_iid"], 1)
                                )
                                / max(epochs, 1),
                                1.0,
                            ),
                            "progress/epoch_fraction": min(
                                completed / max(batch_counts["validation_iid"], 1), 1.0
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
                    namespace="bc/validation_iid",
                    maximum_batches=maximum_validation_batches,
                    progress=(
                        log_validation_progress
                        if maximum_validation_batches is None
                        else None
                    ),
                )
                validation_iid_elapsed = time.perf_counter() - validation_started
                deck_ood_started = time.perf_counter()
                deck_ood_validation = evaluate(
                    model,
                    deck_ood_validation_batches(epoch),
                    device=device,
                    amp=amp,
                    namespace="bc/validation_deck_ood",
                    maximum_batches=maximum_validation_batches,
                )
                validation_deck_ood_elapsed = time.perf_counter() - deck_ood_started
                progress_iteration += (
                    batch_counts["validation_iid"]
                    + batch_counts["validation_deck_ood"]
                )
                optimization_loss = float((loss_sum / decisions).cpu())
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
                epoch_gpu_seconds = time.perf_counter() - started
                gpu_training_seconds += epoch_gpu_seconds
                metrics = {
                    "trainer/epoch": epoch,
                    "trainer/update": global_step,
                    "bc/optimization/loss": optimization_loss,
                    **extra_optimization_metrics,
                    "bc/optimization/token_accuracy": correct / tokens,
                    "bc/optimization/teacher_exact_action": exact / decisions,
                    "bc/optimization/mean_gradient_norm": float(
                        (gradient_norm_sum / max(batches_done, 1)).cpu()
                    ),
                    "bc/optimization/learning_rate": optimizer.param_groups[0]["lr"],
                    "bc/optimization/decisions": decisions,
                    "bc/optimization/tokens": tokens,
                    **validation,
                    **deck_ood_validation,
                    "system/epoch_seconds": epoch_gpu_seconds,
                    "system/gpu/training_seconds_cumulative": gpu_training_seconds,
                    "system/gpu/training_hours_cumulative": gpu_training_seconds / 3600.0,
                    "system/gpu/minimum_training_seconds": minimum_gpu_training_seconds,
                    "system/gpu/minimum_training_fraction": min(
                        gpu_training_seconds / max(minimum_gpu_training_seconds, 1.0), 1.0
                    ),
                    "system/train_decisions_per_second": decisions / training_elapsed,
                    "system/train_iterations_per_second": batches_done / training_elapsed,
                    "system/train_batches": batches_done,
                    "system/validation_iid_seconds": validation_iid_elapsed,
                    "system/validation_deck_ood_seconds": validation_deck_ood_elapsed,
                    "system/parameter_count": sum(p.numel() for p in model.parameters()),
                    "system/global_step": global_step,
                    "system/amp_scale": 1.0,
                    "system/gpu/memory_allocated_bytes": (
                        torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
                    ),
                    "system/gpu/max_memory_allocated_bytes": (
                        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                    ),
                    "system/gpu/memory_reserved_bytes": (
                        torch.cuda.memory_reserved(device) if device.type == "cuda" else 0
                    ),
                }
                logger.log(epoch, metrics)
                criteria = ["latest"]
                current_loss = metrics["bc/validation_iid/loss"]
                previous_loss = best["loss"]
                if current_loss < previous_loss:
                    best["loss"] = current_loss
                    criteria.append("best_validation_loss")
                if metrics["bc/validation_iid/teacher_exact_action"] > best["teacher_exact"]:
                    best["teacher_exact"] = metrics[
                        "bc/validation_iid/teacher_exact_action"
                    ]
                    criteria.append("best_teacher_exact")
                current_greedy = metrics["bc/validation_iid/exact_action"]
                previous_greedy = best["greedy_exact"]
                if current_greedy > previous_greedy:
                    best["greedy_exact"] = current_greedy
                    criteria.append("best_greedy_exact")
                current_ood_greedy = metrics["bc/validation_deck_ood/exact_action"]
                if current_ood_greedy > best["deck_ood_greedy_exact"]:
                    best["deck_ood_greedy_exact"] = current_ood_greedy
                    criteria.append("best_deck_ood_greedy_exact")
                if current_loss <= previous_loss - early_stopping_min_delta:
                    no_loss_improvement = 0
                else:
                    no_loss_improvement += 1
                checkpoint = _checkpoints.save_checkpoint(
                    paths.checkpoints,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    grad_scaler=grad_scaler,
                    completed_epoch=epoch,
                    global_step=global_step,
                    trainer_state={
                        "best": best,
                        "no_loss_improvement": no_loss_improvement,
                        "progress_iteration": progress_iteration,
                        "gpu_training_seconds": gpu_training_seconds,
                        "history": [*history, metrics],
                    },
                    metadata={
                        field: config[field]
                        for field in _checkpoints.REQUIRED_METADATA
                    },
                    criteria=criteria,
                )
                history.append({**metrics, "checkpoint": checkpoint})
                logger.log(
                    epoch,
                    {
                        "trainer/epoch": epoch,
                        "trainer/update": global_step,
                        "system/checkpoint/retained_bytes": checkpoint[
                            "retained_checkpoint_bytes"
                        ],
                        "system/checkpoint/resumable": 1,
                        "system/checkpoint/optimizer_state_saved": 1,
                    },
                )
                if (
                    early_stopping_patience > 0
                    and no_loss_improvement >= early_stopping_patience
                    and gpu_training_seconds >= minimum_gpu_training_seconds
                ):
                    break
                print(json.dumps({"event": "0021_bc_epoch", **metrics}, sort_keys=True), flush=True)
        summary = {
            "state": "complete",
            "epochs": int(history[-1]["trainer/epoch"]) if history else start_epoch,
            "requested_epochs": epochs,
            "global_step": global_step,
            "best_validation_loss": best["loss"],
            "best_teacher_exact": best["teacher_exact"],
            "best_greedy_exact": best["greedy_exact"],
            "best_deck_ood_greedy_exact": best["deck_ood_greedy_exact"],
            "retained_checkpoint_bytes": _checkpoints.retained_checkpoint_bytes(
                paths.checkpoints
            ),
            "gpu_training_seconds": gpu_training_seconds,
            "minimum_gpu_training_seconds": minimum_gpu_training_seconds,
            "history": history,
        }
        _atomic_json(paths.summary, summary)
        _atomic_json(
            paths.checkpoint_selection,
            {
                "latest": "final completed epoch",
                "best_validation_loss": "minimum IID mean per-decision sequence NLL",
                "best_teacher_exact": "maximum IID teacher-forced exact action",
                "best_greedy_exact": "maximum IID free greedy exact action",
                "best_deck_ood_greedy_exact": "maximum deck-OOD free greedy exact action",
            },
        )
        write_version_status(
            paths,
            {
                "state": "completed",
                "completed_at": time.time(),
                "resumable_training_state_saved": True,
                "optimizer_state_saved": True,
                "retained_checkpoint_bytes": summary["retained_checkpoint_bytes"],
                "gpu_training_seconds": gpu_training_seconds,
                "minimum_gpu_training_seconds": minimum_gpu_training_seconds,
                **best,
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


__all__ = ["evaluate", "per_decision_sequence_nll", "train"]
