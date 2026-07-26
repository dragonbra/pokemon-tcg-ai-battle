"""BC optimization and fixed-snapshot validation metrics."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .runtime_batch import trim_target_padding


@dataclass(frozen=True, slots=True)
class TeacherForcedBatchMetrics:
    loss: Tensor
    tokens: Tensor
    token_correct: Tensor
    actions: Tensor
    exact_actions: Tensor


def teacher_forced_batch_metrics(
    model: nn.Module,
    batch: Mapping[str, Tensor],
) -> TeacherForcedBatchMetrics:
    """Use one training forward for both the BC loss and online diagnostics."""
    logits = model.teacher_logits(dict(batch), batch["targets"])
    mask = batch["target_mask"]
    tokens = mask.sum()
    loss = torch.nn.functional.cross_entropy(logits[mask], batch["targets"][mask])
    correct = logits.argmax(-1).eq(batch["targets"]) & mask
    exact = (correct | ~mask).all(dim=1)
    return TeacherForcedBatchMetrics(
        loss=loss,
        tokens=tokens.detach(),
        token_correct=correct.sum().detach(),
        actions=torch.as_tensor(
            batch["targets"].size(0),
            dtype=torch.long,
            device=batch["targets"].device,
        ),
        exact_actions=exact.sum().detach(),
    )


def sequence_loss(model: nn.Module, batch: Mapping[str, Tensor]) -> Tensor:
    return teacher_forced_batch_metrics(model, batch).loss


def evaluate_full_pass(
    model: nn.Module,
    batches: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    namespace: str,
    amp: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(9, dtype=torch.float64, device=device)
    select_counts = torch.zeros(4096, dtype=torch.float64, device=device)
    select_exact = torch.zeros(4096, dtype=torch.float64, device=device)
    actions = 0
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.inference_mode():
        for batch_index, source_batch in enumerate(batches, 1):
            source_batch = trim_target_padding(source_batch)
            batch_size = int(source_batch["state_num"].size(0))
            maximum_steps = int(source_batch["max_count"].max().item()) if batch_size else 0
            batch = {
                key: value.to(device, non_blocking=device.type == "cuda")
                for key, value in source_batch.items()
            }
            with torch.amp.autocast(
                autocast_device,
                enabled=amp and device.type == "cuda",
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            ):
                encoded = model.encode(batch, include_value=False)
                logits = model.teacher_logits_from_encoding(
                    batch, batch["targets"], encoded
                )
                mask = batch["target_mask"]
                losses = torch.nn.functional.cross_entropy(
                    logits[mask], batch["targets"][mask], reduction="sum"
                )
                predictions = logits.argmax(-1)
                correct = predictions.eq(batch["targets"]) & mask
                teacher_ok = (correct | ~mask).all(dim=1)

                option_width = batch["option_mask"].size(1)
                decoded = model.deterministic_action_tensors(
                    batch,
                    encoded=encoded,
                    maximum_steps=maximum_steps,
                )
                target_option_mask = mask & batch["targets"].lt(option_width)
                target_lengths = target_option_mask.sum(dim=1)
                expected_forced = mask.any(dim=1) & target_lengths.eq(mask.sum(dim=1))
                expected_sequences = torch.full_like(decoded.sequences, -1)
                expected_sequence_mask = torch.zeros_like(
                    decoded.sequences, dtype=torch.bool
                )
                comparison_width = min(maximum_steps, batch["targets"].size(1))
                expected_sequences[:, :comparison_width] = batch["targets"][
                    :, :comparison_width
                ]
                expected_sequence_mask[:, :comparison_width] = target_option_mask[
                    :, :comparison_width
                ]
                sequence_matches = (
                    decoded.sequences.eq(expected_sequences) | ~expected_sequence_mask
                ).all(dim=1)
                free_ok = (
                    decoded.legal
                    & decoded.lengths.eq(target_lengths)
                    & sequence_matches
                )
                length_ok = decoded.lengths.eq(target_lengths)
                termination_ok = decoded.forced_terminal.eq(expected_forced)

                batch_totals = torch.stack(
                    (
                        losses.to(torch.float64),
                        mask.sum().to(torch.float64),
                        correct.sum().to(torch.float64),
                        torch.as_tensor(batch_size, dtype=torch.float64, device=device),
                        teacher_ok.sum().to(torch.float64),
                        free_ok.sum().to(torch.float64),
                        decoded.legal.sum().to(torch.float64),
                        length_ok.sum().to(torch.float64),
                        termination_ok.sum().to(torch.float64),
                    )
                )
                totals += batch_totals
                select_type = batch["state_cat"][:, 2]
                ones = torch.ones(batch_size, dtype=torch.float64, device=device)
                select_counts.scatter_add_(0, select_type, ones)
                select_exact.scatter_add_(0, select_type, free_ok.to(torch.float64))
            actions += batch_size
            if progress is not None:
                progress(batch_index, actions)
    values = totals.cpu().tolist()
    (
        loss_sum,
        tokens,
        token_correct,
        total_actions,
        teacher_exact,
        free_exact,
        free_legal,
        length_correct,
        termination_correct,
    ) = values
    if total_actions == 0 or tokens == 0:
        raise ValueError(f"{namespace} evaluation produced no decisions")
    prefix = f"bc/{namespace}"
    metrics: dict[str, float] = {
        f"{prefix}/loss": loss_sum / tokens,
        f"{prefix}/token_accuracy": token_correct / tokens,
        f"{prefix}/teacher_exact_action": teacher_exact / total_actions,
        f"{prefix}/exact_action": free_exact / total_actions,
        f"{prefix}/legal_action": free_legal / total_actions,
        f"{prefix}/action_length_accuracy": length_correct / total_actions,
        f"{prefix}/termination_accuracy": termination_correct / total_actions,
        f"{prefix}/decisions": total_actions,
        f"{prefix}/tokens": tokens,
    }
    counts = select_counts.cpu().tolist()
    exact = select_exact.cpu().tolist()
    for select_type, count in enumerate(counts):
        if count:
            metrics[f"{prefix}/select_type_{select_type}_exact_action"] = (
                exact[select_type] / count
            )
    return metrics


__all__ = [
    "TeacherForcedBatchMetrics",
    "evaluate_full_pass",
    "sequence_loss",
    "teacher_forced_batch_metrics",
]
