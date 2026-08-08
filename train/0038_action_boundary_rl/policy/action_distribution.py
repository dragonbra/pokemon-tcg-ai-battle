"""Ordered unique-option distribution over the original 0031 decoder."""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from torch import Tensor
from torch.distributions import Categorical

from .actor_critic import DecoderPolicyHead, SemanticActorCritic


@dataclass(frozen=True, slots=True)
class SampledAction:
    indices: tuple[int, ...]
    stopped: bool
    log_prob: float
    entropy: float
    value: float


@dataclass(frozen=True, slots=True)
class ActionEvaluation:
    log_prob: Tensor
    entropy: Tensor
    value: Tensor


def _decode(
    head: DecoderPolicyHead,
    batch,
    summary: Tensor,
    options: Tensor,
    values: Tensor,
    *,
    greedy: bool,
    generators: list[torch.Generator] | None = None,
) -> list[SampledAction]:
    decoder = head.action_decoder
    batch_size, option_count, _ = options.shape
    state = decoder.initialize(batch, summary)
    lengths = torch.zeros(batch_size, dtype=torch.long, device=options.device)
    maximum = batch.max_count.clamp_max(decoder.config.max_action_steps)
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
        logits = decoder.logits(batch, options, state)
        distribution = Categorical(logits=logits.float())
        if greedy:
            token = logits.argmax(dim=1)
        elif generators is None:
            token = distribution.sample()
        else:
            if len(generators) != batch_size:
                raise ValueError("one sampling generator is required per batch row")
            token = torch.stack(
                [
                    torch.multinomial(
                        distribution.probs[row],
                        1,
                        generator=generators[row],
                    ).squeeze(0)
                    for row in range(batch_size)
                ]
            )
        total_log_prob += torch.where(active, distribution.log_prob(token), 0.0)
        total_entropy += torch.where(active, distribution.entropy(), 0.0)
        choosing_stop = active & token.eq(option_count)
        choosing_option = active & ~choosing_stop
        safe = token.clamp_max(option_count - 1)
        sequences[:, step] = torch.where(choosing_option, safe, -1)
        state = decoder.consume(options, state, torch.where(choosing_option, safe, -1))
        lengths = state.selected_count
        stopped |= choosing_stop
        finished |= choosing_stop | lengths.ge(maximum)
    if lengths.lt(batch.min_count).any():
        raise RuntimeError("0031 sampled action violated minCount")
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
    model: SemanticActorCritic,
    features: dict[str, Tensor],
    *,
    generators: list[torch.Generator] | None = None,
) -> list[SampledAction]:
    with torch.inference_mode():
        batch, state, options, values = model.encode(features)
        return _decode(
            model.head,
            batch,
            model.actor_summary(state),
            options,
            values,
            greedy=False,
            generators=generators,
        )


def sample_actions_with_encoding(
    model: SemanticActorCritic,
    features: dict[str, Tensor],
    *,
    greedy: bool,
    generators: list[torch.Generator] | None = None,
):
    """Expose the one shared encoding for conditional compound-action heads."""
    def synchronize() -> None:
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)

    with torch.inference_mode():
        synchronize()
        started = time.perf_counter()
        batch, state, options = model.actor.encode(features)
        synchronize()
        encoded = time.perf_counter()
        values, auxiliary = model.value_and_aux_from_encoded(batch, state, options)
        synchronize()
        valued = time.perf_counter()
        actions = _decode(
            model.head, batch, model.actor_summary(state), options, values,
            greedy=greedy, generators=generators,
        )
        synchronize()
        decoded = time.perf_counter()
        return actions, batch, state, options, values, auxiliary, {
            "transformer_seconds": encoded - started,
            "value_seconds": valued - encoded,
            "gru_seconds": decoded - valued,
        }


def greedy_actions(model: SemanticActorCritic, features: dict[str, Tensor]) -> list[SampledAction]:
    with torch.inference_mode():
        batch, state, options, values = model.encode(features)
        return _decode(model.head, batch, model.actor_summary(state), options, values, greedy=True)


def evaluate_actions_encoded(
    head: DecoderPolicyHead,
    batch,
    summary: Tensor,
    options: Tensor,
    sequences: Tensor,
    lengths: Tensor,
    stopped: Tensor,
    value: Tensor,
) -> ActionEvaluation:
    decoder = head.action_decoder
    batch_size, option_count, _ = options.shape
    state = decoder.initialize(batch, summary)
    total_log_prob = torch.zeros(batch_size, device=options.device)
    total_entropy = torch.zeros(batch_size, device=options.device)
    maximum_steps = max(sequences.size(1), int(lengths.max().item()) + int(stopped.any()))
    for step in range(maximum_steps):
        choosing_option = lengths.gt(step)
        choosing_stop = stopped & lengths.eq(step)
        active = choosing_option | choosing_stop
        if not active.any():
            continue
        logits = decoder.logits(batch, options, state)
        raw = sequences[:, step] if step < sequences.size(1) else torch.full_like(lengths, -1)
        token = torch.where(choosing_option, raw, option_count)
        if (choosing_option & (raw.lt(0) | raw.ge(option_count))).any():
            raise ValueError("trajectory contains an invalid option index")
        distribution = Categorical(logits=logits.float())
        total_log_prob += torch.where(active, distribution.log_prob(token), 0.0)
        total_entropy += torch.where(active, distribution.entropy(), 0.0)
        state = decoder.consume(options, state, torch.where(choosing_option, raw, -1))
    return ActionEvaluation(
        total_log_prob,
        total_entropy,
        value,
    )


def evaluate_actions(
    model: SemanticActorCritic,
    features: dict[str, Tensor],
    sequences: Tensor,
    lengths: Tensor,
    stopped: Tensor,
) -> ActionEvaluation:
    batch, state, options, value = model.encode(features)
    return evaluate_actions_encoded(
        model.head, batch, model.actor_summary(state), options, sequences, lengths, stopped, value
    )


__all__ = [
    "ActionEvaluation",
    "SampledAction",
    "evaluate_actions",
    "evaluate_actions_encoded",
    "greedy_actions",
    "sample_actions",
    "sample_actions_with_encoding",
]
