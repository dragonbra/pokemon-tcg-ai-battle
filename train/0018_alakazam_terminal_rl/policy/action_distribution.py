from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Categorical

from .actor_critic import AlakazamActorCritic


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


def sample_actions(
    model: AlakazamActorCritic,
    batch: dict[str, Tensor],
) -> list[SampledAction]:
    """Sample a batch of variable-length legal engine selections."""
    with torch.no_grad():
        state, options, values = model.encode(batch)
        actor = model.actor
        batch_size, option_count, _ = options.shape
        keys = actor.pointer_key(options)
        hidden = torch.tanh(actor.decoder_init(state))
        chosen = torch.zeros_like(batch["option_mask"], dtype=torch.bool)
        minimum = batch["min_count"]
        maximum = batch["max_count"].clamp_max(actor.config.max_action_steps)
        sequences = torch.full(
            (batch_size, actor.config.max_action_steps),
            -1,
            dtype=torch.long,
            device=options.device,
        )
        lengths = torch.zeros(batch_size, dtype=torch.long, device=options.device)
        stopped = torch.zeros(batch_size, dtype=torch.bool, device=options.device)
        finished = maximum == 0
        log_prob = torch.zeros(batch_size, device=options.device)
        entropy = torch.zeros(batch_size, device=options.device)
        rows = torch.arange(batch_size, device=options.device)
        for step in range(int(maximum.max().item())):
            active = ~finished & (lengths < maximum)
            logits = _step_logits(
                model,
                options,
                keys,
                hidden,
                batch["option_mask"],
                chosen,
                lengths >= minimum,
            )
            probabilities = torch.softmax(logits.float(), dim=-1)
            token = torch.multinomial(probabilities, 1).squeeze(1)
            distribution = Categorical(probs=probabilities)
            log_prob = log_prob + torch.where(active, distribution.log_prob(token), 0.0)
            entropy = entropy + torch.where(active, distribution.entropy(), 0.0)
            choosing_stop = active & (token == option_count)
            choosing_option = active & ~choosing_stop
            safe_option = token.clamp_max(option_count - 1)
            sequences[:, step] = torch.where(choosing_option, safe_option, -1)
            selected = options[rows, safe_option]
            advanced = actor.decoder(selected, hidden)
            hidden = torch.where(choosing_option.unsqueeze(1), advanced, hidden)
            chosen.scatter_(
                1,
                safe_option.unsqueeze(1),
                chosen.gather(1, safe_option.unsqueeze(1)) | choosing_option.unsqueeze(1),
            )
            lengths += choosing_option
            stopped |= choosing_stop
            finished |= choosing_stop | (lengths >= maximum)
        if (lengths < minimum).any():
            raise RuntimeError("sampled action violated minCount")
        return [
            SampledAction(
                tuple(int(value) for value in sequences[index, : lengths[index]].tolist()),
                bool(stopped[index].item()),
                float(log_prob[index].item()),
                float(entropy[index].item()),
                float(values[index].item()),
            )
            for index in range(batch_size)
        ]


def greedy_actions(
    model: AlakazamActorCritic,
    batch: dict[str, Tensor],
) -> list[SampledAction]:
    """Decode a greedy evaluation batch while retaining comparable log-prob/value."""
    with torch.no_grad():
        state, options, _ = model.encode(batch)
        decoded = model.actor.deterministic_action_tensors(
            batch, encoded=(state, options)
        )
        maximum_width = max(1, decoded.sequences.size(1))
        sequences = torch.full(
            (decoded.sequences.size(0), maximum_width),
            -1,
            dtype=torch.long,
            device=decoded.sequences.device,
        )
        if decoded.sequences.numel():
            sequences[:, : decoded.sequences.size(1)] = decoded.sequences
        stopped = ~decoded.forced_terminal
        evaluated = evaluate_actions(
            model, batch, sequences, decoded.lengths, stopped
        )
        return [
            SampledAction(
                tuple(
                    int(value)
                    for value in sequences[index, : decoded.lengths[index]].tolist()
                ),
                bool(stopped[index].item()),
                float(evaluated.log_prob[index].item()),
                float(evaluated.entropy[index].item()),
                float(evaluated.value[index].item()),
            )
            for index in range(sequences.size(0))
        ]


def _step_logits(
    model: AlakazamActorCritic,
    options: Tensor,
    keys: Tensor,
    hidden: Tensor,
    option_mask: Tensor,
    chosen: Tensor,
    can_stop: Tensor,
) -> Tensor:
    actor = model.actor
    pointer = (
        actor.pointer_query(hidden).unsqueeze(1) * keys
    ).sum(-1) / math.sqrt(actor.config.d_model)
    pointer = pointer + actor.option_bias(options).squeeze(-1)
    pointer = pointer.masked_fill(
        ~option_mask | chosen, torch.finfo(pointer.dtype).min
    )
    stop = actor.stop(hidden).masked_fill(
        ~can_stop.unsqueeze(1), torch.finfo(pointer.dtype).min
    )
    return torch.cat((pointer, stop), dim=1)


def sample_action(
    model: AlakazamActorCritic,
    batch: dict[str, Tensor],
    *,
    generator: torch.Generator | None = None,
) -> SampledAction:
    if batch["global_cat"].size(0) != 1:
        raise ValueError("sample_action requires one observation")
    with torch.no_grad():
        state, options, value = model.encode(batch)
        actor = model.actor
        keys = actor.pointer_key(options)
        hidden = torch.tanh(actor.decoder_init(state))
        chosen = torch.zeros_like(batch["option_mask"], dtype=torch.bool)
        minimum = int(batch["min_count"].item())
        maximum = min(int(batch["max_count"].item()), actor.config.max_action_steps)
        selected: list[int] = []
        total_log_prob = 0.0
        total_entropy = 0.0
        stopped = False
        for step in range(maximum):
            logits = _step_logits(
                model,
                options,
                keys,
                hidden,
                batch["option_mask"],
                chosen,
                torch.tensor([step >= minimum], device=options.device),
            )
            probabilities = torch.softmax(logits.float(), dim=-1)
            token = int(torch.multinomial(probabilities, 1, generator=generator).item())
            distribution = Categorical(probs=probabilities)
            token_tensor = torch.tensor([token], device=options.device)
            total_log_prob += float(distribution.log_prob(token_tensor).item())
            total_entropy += float(distribution.entropy().item())
            if token == options.size(1):
                stopped = True
                break
            selected.append(token)
            chosen[0, token] = True
            hidden = actor.decoder(options[:, token], hidden)
        if len(selected) < minimum:
            raise RuntimeError("sampled action violated minCount")
        return SampledAction(
            tuple(selected), stopped, total_log_prob, total_entropy, float(value.item())
        )


def evaluate_actions(
    model: AlakazamActorCritic,
    batch: dict[str, Tensor],
    sequences: Tensor,
    lengths: Tensor,
    stopped: Tensor,
) -> ActionEvaluation:
    state, options, value = model.encode(batch)
    actor = model.actor
    batch_size, option_count, _ = options.shape
    if sequences.size(0) != batch_size:
        raise ValueError("action sequence batch dimension mismatch")
    keys = actor.pointer_key(options)
    hidden = torch.tanh(actor.decoder_init(state))
    chosen = torch.zeros_like(batch["option_mask"], dtype=torch.bool)
    log_prob = torch.zeros(batch_size, device=options.device)
    entropy = torch.zeros(batch_size, device=options.device)
    rows = torch.arange(batch_size, device=options.device)
    maximum_tokens = int(lengths.max().item()) + int(stopped.any().item())
    for step in range(maximum_tokens):
        active_option = step < lengths
        active_stop = stopped & (step == lengths)
        active = active_option | active_stop
        if not active.any():
            continue
        logits = _step_logits(
            model,
            options,
            keys,
            hidden,
            batch["option_mask"],
            chosen,
            step >= batch["min_count"],
        )
        distribution = Categorical(logits=logits.float())
        safe_option = sequences[:, min(step, sequences.size(1) - 1)].clamp(0, option_count - 1)
        token = torch.where(active_stop, option_count, safe_option)
        log_prob = log_prob + torch.where(active, distribution.log_prob(token), 0.0)
        entropy = entropy + torch.where(active, distribution.entropy(), 0.0)
        selected = options[rows, safe_option]
        advanced = actor.decoder(selected, hidden)
        hidden = torch.where(active_option.unsqueeze(1), advanced, hidden)
        chosen.scatter_(
            1,
            safe_option.unsqueeze(1),
            chosen.gather(1, safe_option.unsqueeze(1)) | active_option.unsqueeze(1),
        )
    return ActionEvaluation(log_prob, entropy, value)


__all__ = [
    "ActionEvaluation",
    "SampledAction",
    "evaluate_actions",
    "greedy_actions",
    "sample_action",
    "sample_actions",
]
