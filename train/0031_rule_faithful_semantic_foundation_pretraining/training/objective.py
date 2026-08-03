"""Teacher-forced and greedy metrics for 0031 rule-faithful BC."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TeacherBatch:
    loss: Tensor
    tokens: Tensor
    correct: Tensor
    exact: Tensor
    logits: Tensor


@dataclass(frozen=True)
class DecodedActions:
    sequences: Tensor
    lengths: Tensor
    legal: Tensor


def teacher_batch(model: nn.Module, batch: dict[str, Tensor]) -> TeacherBatch:
    logits = model.teacher_logits(batch)
    mask = batch["targets"].ne(-100)
    loss = F.cross_entropy(logits[mask], batch["targets"][mask])
    correct = logits.argmax(-1).eq(batch["targets"]) & mask
    exact = (correct | ~mask).all(1)
    return TeacherBatch(loss, mask.sum(), correct.sum(), exact.sum(), logits)


def deterministic_decode(model: nn.Module, batch: dict[str, Tensor]) -> DecodedActions:
    """Greedily decode a batch while enforcing unique options and count bounds."""
    public_decode = getattr(model, "deterministic_action_tensors", None)
    if callable(public_decode):
        result = public_decode(batch)
        return DecodedActions(result.sequences, result.lengths, result.legal)
    state, options = model.encode(batch)
    batch_size, option_count, width = options.shape
    hidden = torch.tanh(model.decoder_init(state))
    keys = model.pointer_key(options)
    chosen = torch.zeros(
        (batch_size, option_count), dtype=torch.bool, device=options.device
    )
    lengths = torch.zeros(batch_size, dtype=torch.long, device=options.device)
    legal = torch.ones(batch_size, dtype=torch.bool, device=options.device)
    active = torch.ones(batch_size, dtype=torch.bool, device=options.device)
    maximum_steps = min(option_count, int(batch["max_count"].max().item()))
    sequences = torch.full(
        (batch_size, maximum_steps), -1, dtype=torch.long, device=options.device
    )
    rows = torch.arange(batch_size, device=options.device)
    for step in range(maximum_steps):
        pointer = (
            model.pointer_query(hidden).unsqueeze(1) * keys
        ).sum(-1) / math.sqrt(int(model.config.d_model))
        pointer = pointer + model.option_bias(options).squeeze(-1)
        available = batch["option_mask"] & ~chosen
        pointer = pointer.masked_fill(~available, torch.finfo(pointer.dtype).min)
        best_score, best_index = pointer.max(1)
        stop_score = model.stop(hidden).squeeze(1)
        at_maximum = active & lengths.ge(batch["max_count"])
        may_stop = lengths.ge(batch["min_count"])
        chose_stop = active & may_stop & stop_score.ge(best_score)
        no_option = active & ~available.any(1)
        illegal = no_option & ~may_stop
        legal &= ~illegal
        active &= ~(at_maximum | chose_stop | no_option)
        selecting = active.clone()
        if not selecting.any():
            break
        sequences[selecting, step] = best_index[selecting]
        chosen[rows[selecting], best_index[selecting]] = True
        selected = options[rows, best_index]
        updated = model.decoder(selected, hidden)
        hidden = torch.where(selecting.unsqueeze(1), updated, hidden)
        lengths += selecting.long()
    legal &= lengths.ge(batch["min_count"]) & lengths.le(batch["max_count"])
    return DecodedActions(sequences, lengths, legal)


def _to_device(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def evaluate_batches(
    model: nn.Module,
    batches: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    amp: bool = False,
    maximum_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(8, dtype=torch.float64, device=device)
    with torch.inference_mode():
        for batch_index, source in enumerate(batches, 1):
            if maximum_batches is not None and batch_index > maximum_batches:
                break
            batch = _to_device(source, device)
            with torch.amp.autocast(
                "cuda",
                enabled=amp and device.type == "cuda",
                dtype=torch.bfloat16,
            ):
                teacher = teacher_batch(model, batch)
                decoded = deterministic_decode(model, batch)
                mask = batch["targets"].ne(-100)
                loss_sum = F.cross_entropy(
                    teacher.logits[mask], batch["targets"][mask], reduction="sum"
                )
            expected_mask = mask & batch["targets"].lt(batch["option_mask"].size(1))
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
                    teacher.tokens.double(),
                    teacher.correct.double(),
                    torch.tensor(batch["targets"].size(0), device=device).double(),
                    teacher.exact.double(),
                    greedy_exact.sum().double(),
                    decoded.legal.sum().double(),
                    decoded.lengths.eq(expected_lengths).sum().double(),
                )
            )
    values = totals.cpu().tolist()
    loss_sum, tokens, correct, decisions, teacher_exact, greedy_exact, legal, lengths = values
    if tokens == 0 or decisions == 0:
        raise ValueError("evaluation produced no decisions")
    return {
        "loss": loss_sum / tokens,
        "token_accuracy": correct / tokens,
        "teacher_exact_action": teacher_exact / decisions,
        "exact_action": greedy_exact / decisions,
        "legal_action": legal / decisions,
        "action_length_accuracy": lengths / decisions,
        "decisions": int(decisions),
        "tokens": int(tokens),
    }


def evaluate_audited_batches(
    model: nn.Module,
    batches: Iterable[tuple[Mapping[str, Tensor], list[Mapping[str, object]]]],
    *,
    device: torch.device,
    amp: bool = False,
    maximum_batches: int | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, dict[str, float]]]]:
    """Evaluate once while aggregating provenance outside actor forward."""
    model.eval()
    totals = torch.zeros(8, dtype=torch.float64)
    grouped: dict[str, dict[str, Tensor]] = {
        "by_source": defaultdict(lambda: torch.zeros(8, dtype=torch.float64)),
        "by_deck": defaultdict(lambda: torch.zeros(8, dtype=torch.float64)),
    }
    with torch.inference_mode():
        for batch_index, (source, audits) in enumerate(batches, 1):
            if maximum_batches is not None and batch_index > maximum_batches:
                break
            batch = _to_device(source, device)
            if len(audits) != batch["targets"].size(0):
                raise ValueError("validation provenance does not align with actor batch")
            with torch.amp.autocast(
                "cuda", enabled=amp and device.type == "cuda", dtype=torch.bfloat16
            ):
                teacher = teacher_batch(model, batch)
                decoded = deterministic_decode(model, batch)
                mask = batch["targets"].ne(-100)
                token_loss = F.cross_entropy(
                    teacher.logits.transpose(1, 2),
                    batch["targets"],
                    ignore_index=-100,
                    reduction="none",
                )
            expected_mask = mask & batch["targets"].lt(batch["option_mask"].size(1))
            expected_lengths = expected_mask.sum(1)
            comparison_width = min(decoded.sequences.size(1), batch["targets"].size(1))
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
            correct = teacher.logits.argmax(-1).eq(batch["targets"]) & mask
            rows = torch.stack(
                (
                    token_loss.sum(1),
                    mask.sum(1),
                    correct.sum(1),
                    torch.ones(mask.size(0), device=device),
                    (correct | ~mask).all(1),
                    greedy_exact,
                    decoded.legal,
                    decoded.lengths.eq(expected_lengths),
                ),
                dim=1,
            ).double().cpu()
            totals += rows.sum(0)
            for row, audit in zip(rows, audits, strict=True):
                source_key = f"{audit.get('source_id')}:{audit.get('source_team_name', '')}"
                deck_key = str(audit.get("deck_sha256") or "unknown")
                grouped["by_source"][source_key] += row
                grouped["by_deck"][deck_key] += row

    def summarize(values: Tensor) -> dict[str, float]:
        loss_sum, tokens, correct, decisions, teacher_exact, greedy_exact, legal, lengths = values.tolist()
        if tokens == 0 or decisions == 0:
            raise ValueError("evaluation group produced no decisions")
        return {
            "loss": loss_sum / tokens,
            "token_accuracy": correct / tokens,
            "teacher_exact_action": teacher_exact / decisions,
            "exact_action": greedy_exact / decisions,
            "legal_action": legal / decisions,
            "action_length_accuracy": lengths / decisions,
            "decisions": int(decisions),
            "tokens": int(tokens),
        }

    return summarize(totals), {
        family: {key: summarize(values) for key, values in sorted(groups.items())}
        for family, groups in grouped.items()
    }


__all__ = [
    "DecodedActions",
    "TeacherBatch",
    "deterministic_decode",
    "evaluate_batches",
    "evaluate_audited_batches",
    "teacher_batch",
]
