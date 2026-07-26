"""Full-pass BC metrics, including free-running decoder health."""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


def sequence_loss(model: nn.Module, batch: Mapping[str, Tensor]) -> Tensor:
    logits = model.teacher_logits(dict(batch), batch["targets"])
    mask = batch["target_mask"]
    return torch.nn.functional.cross_entropy(logits[mask], batch["targets"][mask])


@dataclass(slots=True)
class _Accumulator:
    loss_sum: float = 0.0
    tokens: int = 0
    token_correct: int = 0
    actions: int = 0
    teacher_exact: int = 0
    free_exact: int = 0
    free_legal: int = 0
    length_correct: int = 0
    termination_correct: int = 0
    select_counts: Counter[int] = None  # type: ignore[assignment]
    select_exact: Counter[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.select_counts = Counter()
        self.select_exact = Counter()


def evaluate_full_pass(
    model: nn.Module,
    batches: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    namespace: str,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, float]:
    model.eval()
    totals = _Accumulator()
    with torch.no_grad():
        for batch_index, source_batch in enumerate(batches, 1):
            batch = {key: value.to(device) for key, value in source_batch.items()}
            logits = model.teacher_logits(batch, batch["targets"])
            mask = batch["target_mask"]
            losses = torch.nn.functional.cross_entropy(
                logits[mask], batch["targets"][mask], reduction="sum"
            )
            predictions = logits.argmax(-1)
            correct = predictions.eq(batch["targets"]) & mask
            token_count = int(mask.sum().item())
            totals.loss_sum += float(losses.item())
            totals.tokens += token_count
            totals.token_correct += int(correct.sum().item())
            option_width = batch["option_mask"].size(1)
            decoded = model.deterministic_actions(batch)
            for row in range(batch["state_num"].size(0)):
                row_mask = mask[row]
                teacher_ok = bool(correct[row, row_mask].all().item())
                target_values = batch["targets"][row, row_mask].tolist()
                target_sequence = tuple(int(value) for value in target_values if value < option_width)
                expected_forced = bool(target_values and target_values[-1] < option_width)
                legal = decoded.legal[row]
                predicted = decoded.sequences[row]
                free_ok = legal and predicted == target_sequence
                totals.actions += 1
                totals.teacher_exact += int(teacher_ok)
                totals.free_exact += int(free_ok)
                totals.free_legal += int(legal)
                totals.length_correct += int(len(predicted) == len(target_sequence))
                totals.termination_correct += int(
                    decoded.forced_terminal[row] == expected_forced
                )
                select_type = int(batch["state_cat"][row, 2].item())
                totals.select_counts[select_type] += 1
                totals.select_exact[select_type] += int(free_ok)
            if progress is not None:
                progress(batch_index, totals.actions)
    if totals.actions == 0 or totals.tokens == 0:
        raise ValueError(f"{namespace} evaluation produced no decisions")
    prefix = f"bc/{namespace}"
    metrics: dict[str, float] = {
        f"{prefix}/loss": totals.loss_sum / totals.tokens,
        f"{prefix}/token_accuracy": totals.token_correct / totals.tokens,
        f"{prefix}/teacher_exact_action": totals.teacher_exact / totals.actions,
        f"{prefix}/exact_action": totals.free_exact / totals.actions,
        f"{prefix}/legal_action": totals.free_legal / totals.actions,
        f"{prefix}/action_length_accuracy": totals.length_correct / totals.actions,
        f"{prefix}/termination_accuracy": totals.termination_correct / totals.actions,
        f"{prefix}/decisions": float(totals.actions),
        f"{prefix}/tokens": float(totals.tokens),
    }
    for select_type, count in sorted(totals.select_counts.items()):
        metrics[f"{prefix}/select_type_{select_type}_exact_action"] = (
            totals.select_exact[select_type] / count
        )
    return metrics


__all__ = ["evaluate_full_pass", "sequence_loss"]
