"""Categorical distribution over ordered unique legal options plus STOP."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Categorical

from .actor_critic import CanonicalActorCritic, DecoderPolicyHead


@dataclass(frozen=True)
class SampledAction:
    indices: tuple[int, ...]
    stopped: bool
    log_prob: float
    entropy: float
    value: float


@dataclass(frozen=True)
class ActionEvaluation:
    log_prob: Tensor
    entropy: Tensor
    value: Tensor


def _sample_encoded(
    head: DecoderPolicyHead,
    batch: dict[str, Tensor],
    summary: Tensor,
    options: Tensor,
    values: Tensor,
    *,
    greedy: bool,
) -> list[SampledAction]:
    decoder = head.action_decoder
    batch_size, option_count, _ = options.shape
    hidden = decoder.initial_hidden(summary)
    available = batch["option_mask"].clone()
    lengths = torch.zeros(batch_size, dtype=torch.long, device=options.device)
    maximum = batch["max_count"].clamp_max(decoder.config.max_action_steps)
    stopped = torch.zeros(batch_size, dtype=torch.bool, device=options.device)
    finished = maximum.eq(0)
    sequences = torch.full(
        (batch_size, decoder.config.max_action_steps),
        -1,
        dtype=torch.long,
        device=options.device,
    )
    total_log_prob = torch.zeros(batch_size, device=options.device)
    total_entropy = torch.zeros(batch_size, device=options.device)
    for step in range(int(maximum.max().item())):
        active = ~finished & lengths.lt(maximum)
        logits = decoder.logits(batch, options, hidden, available, lengths)
        distribution = Categorical(logits=logits.float())
        token = logits.argmax(dim=1) if greedy else distribution.sample()
        total_log_prob += torch.where(active, distribution.log_prob(token), 0.0)
        total_entropy += torch.where(active, distribution.entropy(), 0.0)
        choosing_stop = active & token.eq(option_count)
        choosing_option = active & ~choosing_stop
        safe = token.clamp_max(option_count - 1)
        sequences[:, step] = torch.where(choosing_option, safe, -1)
        hidden, available, updated_count = decoder.consume(
            options, hidden, available, lengths, torch.where(choosing_option, safe, -1)
        )
        lengths = torch.where(choosing_option, updated_count, lengths)
        stopped |= choosing_stop
        finished |= choosing_stop | lengths.ge(maximum)
    if lengths.lt(batch["min_count"]).any():
        raise RuntimeError("canonical sampled action violated minCount")
    return [
        SampledAction(
            tuple(int(value) for value in sequences[row, : lengths[row]].tolist()),
            bool(stopped[row]),
            float(total_log_prob[row]),
            float(total_entropy[row]),
            float(values[row]),
        )
        for row in range(batch_size)
    ]


def sample_actions(
    model: CanonicalActorCritic, batch: dict[str, Tensor]
) -> list[SampledAction]:
    with torch.no_grad():
        summary, options, values = model.encode(batch)
        return _sample_encoded(model.head, batch, summary, options, values, greedy=False)


def greedy_actions(
    model: CanonicalActorCritic, batch: dict[str, Tensor]
) -> list[SampledAction]:
    with torch.no_grad():
        summary, options, values = model.encode(batch)
        return _sample_encoded(model.head, batch, summary, options, values, greedy=True)


def evaluate_actions_encoded(
    head: DecoderPolicyHead,
    batch: dict[str, Tensor],
    summary: Tensor,
    options: Tensor,
    sequences: Tensor,
    lengths: Tensor,
    stopped: Tensor,
) -> ActionEvaluation:
    decoder = head.action_decoder
    batch_size, option_count, _ = options.shape
    hidden = decoder.initial_hidden(summary)
    available = batch["option_mask"].clone()
    selected_count = torch.zeros(batch_size, dtype=torch.long, device=options.device)
    total_log_prob = torch.zeros(batch_size, device=options.device)
    total_entropy = torch.zeros(batch_size, device=options.device)
    maximum_steps = max(sequences.size(1), int(lengths.max().item()) + int(stopped.any()))
    for step in range(maximum_steps):
        choosing_option = lengths.gt(step)
        choosing_stop = stopped & lengths.eq(step)
        active = choosing_option | choosing_stop
        if not active.any():
            continue
        logits = decoder.logits(batch, options, hidden, available, selected_count)
        raw = sequences[:, step] if step < sequences.size(1) else torch.full_like(lengths, -1)
        token = torch.where(choosing_option, raw, option_count)
        if (choosing_option & (raw.lt(0) | raw.ge(option_count))).any():
            raise ValueError("trajectory contains an invalid option index")
        distribution = Categorical(logits=logits.float())
        total_log_prob += torch.where(active, distribution.log_prob(token), 0.0)
        total_entropy += torch.where(active, distribution.entropy(), 0.0)
        hidden, available, updated_count = decoder.consume(
            options,
            hidden,
            available,
            selected_count,
            torch.where(choosing_option, raw, -1),
        )
        selected_count = torch.where(choosing_option, updated_count, selected_count)
    return ActionEvaluation(
        total_log_prob,
        total_entropy,
        head.value_head(summary).squeeze(-1),
    )


def evaluate_actions(
    model: CanonicalActorCritic,
    batch: dict[str, Tensor],
    sequences: Tensor,
    lengths: Tensor,
    stopped: Tensor,
) -> ActionEvaluation:
    summary, options, _ = model.encode(batch)
    return evaluate_actions_encoded(model.head, batch, summary, options, sequences, lengths, stopped)


__all__ = [
    "ActionEvaluation",
    "SampledAction",
    "evaluate_actions",
    "evaluate_actions_encoded",
    "greedy_actions",
    "sample_actions",
]
