"""Exact action-distribution mixture for the 0047 Actor experts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import fields, is_dataclass, replace
from typing import Any

import torch
from torch import Tensor
from torch.distributions import Categorical

from .compound_evaluation import evaluate_parameter_actions
from .moe_actor_critic import MetaRoutedMoEActorCritic


@dataclass(frozen=True, slots=True)
class MoEActionEvaluation:
    root_log_prob: Tensor
    joint_log_prob: Tensor
    root_entropy: Tensor
    allocation_entropy: Tensor
    value: Tensor
    expert_root_log_prob: Tensor
    gate: Tensor
    auxiliary: dict[str, Tensor] | None = None


def _log_gate(gate: Tensor) -> Tensor:
    return torch.where(gate.gt(0), gate.log(), torch.full_like(gate, -torch.inf))


def select_rows(value: Any, rows: Tensor, batch_size: int) -> Any:
    def selected(item: Any) -> Any:
        if isinstance(item, Tensor) and item.ndim and item.shape[0] == batch_size:
            return item.index_select(0, rows)
        return item

    if isinstance(value, Mapping):
        tensors = {name: selected(item) for name, item in value.items()}
        constructor = getattr(type(value), "from_mapping", None)
        return constructor(tensors) if callable(constructor) else tensors
    if is_dataclass(value):
        return replace(value, **{
            field.name: selected(getattr(value, field.name)) for field in fields(value)
        })
    return selected(value)


def routing_groups(
    model: MetaRoutedMoEActorCritic, meta_ids: Tensor,
) -> list[tuple[Tensor, tuple[int, ...]]]:
    groups: list[tuple[Tensor, tuple[int, ...]]] = []
    if model.soft_routing:
        unknown = meta_ids.lt(0).nonzero(as_tuple=False).flatten()
        if unknown.numel():
            groups.append((unknown, (0,)))
        identified = meta_ids.ge(0).nonzero(as_tuple=False).flatten()
        if identified.numel():
            groups.append((identified, tuple(range(model.expert_count))))
        return groups
    core = torch.zeros_like(meta_ids, dtype=torch.bool)
    for meta_id in model.priority_meta_ids:
        core |= meta_ids.eq(meta_id)
    fallback = (~core).nonzero(as_tuple=False).flatten()
    if fallback.numel():
        groups.append((fallback, (0,)))
    for meta_id in model.priority_meta_ids:
        rows = meta_ids.eq(meta_id).nonzero(as_tuple=False).flatten()
        if rows.numel():
            specialist = model.meta_to_expert[meta_id]
            groups.append((rows, (specialist,)))
    return groups


def _evaluate_group(
    model: MetaRoutedMoEActorCritic, validated: Any, state: Any, prefix: Tensor,
    value_options: Tensor, sequences: Tensor, lengths: Tensor, stopped: Tensor,
    routing_meta_ids: Tensor, macro_actions: tuple[dict | None, ...],
    active_experts: tuple[int, ...], value: Tensor,
) -> MoEActionEvaluation:
    gate = model.gate_probabilities(routing_meta_ids)
    active_index = torch.tensor(active_experts, device=value.device, dtype=torch.long)
    active_gate = gate.index_select(1, active_index)
    batch_size, option_count = validated.option_mask.shape
    expert_options = {
        index: model.expert_options(index, validated, state, prefix, value_options)
        for index in active_experts
    }
    decoder_states = {
        index: model.experts[index].action_decoder.initialize(validated, state.summary)
        for index in active_experts
    }
    expert_logprob = value.new_zeros((batch_size, len(active_experts)))
    root_entropy = value.new_zeros(batch_size)
    posterior_log = _log_gate(active_gate)
    maximum_steps = max(
        sequences.size(1), int(lengths.max().item()) + int(bool(stopped.any()))
    )
    for step in range(maximum_steps):
        choosing_option = lengths.gt(step)
        choosing_stop = stopped & lengths.eq(step)
        active = choosing_option | choosing_stop
        if not bool(active.any()):
            continue
        raw = (
            sequences[:, step] if step < sequences.size(1)
            else torch.full_like(lengths, -1)
        )
        if bool((choosing_option & (raw.lt(0) | raw.ge(option_count))).any()):
            raise ValueError("0047 trajectory contains an invalid option index")
        token = torch.where(choosing_option, raw, option_count)
        step_logps: list[Tensor] = []
        for index in active_experts:
            expert = model.experts[index]
            logits = expert.action_decoder.logits(
                validated, expert_options[index], decoder_states[index]
            )
            step_logps.append(torch.log_softmax(logits.float(), dim=-1))
        stacked = torch.stack(step_logps, dim=1)
        effective_log = torch.logsumexp(posterior_log.unsqueeze(-1) + stacked, dim=1)
        effective_dist = Categorical(logits=effective_log)
        root_entropy += torch.where(active, effective_dist.entropy(), 0.0)
        chosen_expert_log = stacked.gather(
            2, token[:, None, None].expand(-1, len(active_experts), 1)
        ).squeeze(-1)
        expert_logprob += torch.where(active[:, None], chosen_expert_log, 0.0)
        posterior_log = torch.where(
            active[:, None], posterior_log + chosen_expert_log, posterior_log
        )
        posterior_log = posterior_log - torch.logsumexp(
            posterior_log, dim=1, keepdim=True
        )
        for index in active_experts:
            expert = model.experts[index]
            decoder_states[index] = expert.action_decoder.consume(
                expert_options[index], decoder_states[index],
                torch.where(choosing_option, raw, -1),
            )

    root_log_prob = torch.logsumexp(_log_gate(active_gate) + expert_logprob, dim=1)
    allocation_logps: list[Tensor] = []
    allocation_entropies: list[Tensor] = []
    for index in active_experts:
        expert = model.experts[index]
        proxy = _ExpertAllocationProxy(expert.allocation_head)
        logprob, entropy = evaluate_parameter_actions(
            proxy, validated, state, expert_options[index], macro_actions
        )
        allocation_logps.append(logprob)
        allocation_entropies.append(entropy)
    allocation_matrix = torch.stack(allocation_logps, dim=1)
    joint_log_prob = torch.logsumexp(
        _log_gate(active_gate) + expert_logprob + allocation_matrix, dim=1
    )
    posterior = posterior_log.exp()
    allocation_entropy = (
        posterior * torch.stack(allocation_entropies, dim=1)
    ).sum(dim=1)
    all_expert_logprob = value.new_zeros((batch_size, model.expert_count)).index_copy(
        1, active_index, expert_logprob
    )
    return MoEActionEvaluation(
        root_log_prob, joint_log_prob, root_entropy, allocation_entropy,
        value, all_expert_logprob, gate,
    )


def evaluate_moe_actions(
    model: MetaRoutedMoEActorCritic,
    features: dict[str, Tensor],
    sequences: Tensor,
    lengths: Tensor,
    stopped: Tensor,
    routing_meta_ids: Tensor,
    macro_actions: tuple[dict | None, ...],
) -> MoEActionEvaluation:
    validated, state, prefix, value_options = model.encode_shared(features)
    value, auxiliary = model.value_and_aux_from_encoded(
        validated, state, value_options
    )
    batch_size = validated.batch_size
    root_log_prob = value.new_zeros(batch_size)
    joint_log_prob = value.new_zeros(batch_size)
    root_entropy = value.new_zeros(batch_size)
    allocation_entropy = value.new_zeros(batch_size)
    expert_logprob = value.new_zeros((batch_size, model.expert_count))
    gate = value.new_zeros((batch_size, model.expert_count))
    for rows, active_experts in routing_groups(model, routing_meta_ids):
        partial = _evaluate_group(
            model,
            select_rows(validated, rows, batch_size),
            select_rows(state, rows, batch_size),
            prefix.index_select(0, rows), value_options.index_select(0, rows),
            sequences.index_select(0, rows), lengths.index_select(0, rows),
            stopped.index_select(0, rows), routing_meta_ids.index_select(0, rows),
            tuple(macro_actions[int(row)] for row in rows), active_experts,
            value.index_select(0, rows),
        )
        root_log_prob = root_log_prob.index_copy(0, rows, partial.root_log_prob)
        joint_log_prob = joint_log_prob.index_copy(0, rows, partial.joint_log_prob)
        root_entropy = root_entropy.index_copy(0, rows, partial.root_entropy)
        allocation_entropy = allocation_entropy.index_copy(0, rows, partial.allocation_entropy)
        expert_logprob = expert_logprob.index_copy(0, rows, partial.expert_root_log_prob)
        gate = gate.index_copy(0, rows, partial.gate)
    return MoEActionEvaluation(
        root_log_prob, joint_log_prob, root_entropy, allocation_entropy,
        value, expert_logprob, gate, auxiliary,
    )


class _ExpertAllocationProxy:
    def __init__(self, allocation_head: Any) -> None:
        self.allocation_head = allocation_head


def decode_moe_device(
    model: MetaRoutedMoEActorCritic,
    validated: Any,
    state: Any,
    prefix: Tensor,
    value_options: Tensor,
    routing_meta_ids: Tensor,
    *,
    max_select: int,
    greedy: bool,
    compute_stats: bool,
    sampling_seeds: Tensor | None,
    sampling_counters: Tensor | None,
    active_experts: tuple[int, ...] | None = None,
) -> tuple[dict[str, Tensor], dict[int, Tensor], Tensor]:
    """Decode one sequence from `sum_k g_k pi_k`, never from averaged weights."""
    active_experts = tuple(range(model.expert_count)) if active_experts is None else active_experts
    active_index = torch.tensor(active_experts, device=model.device, dtype=torch.long)
    gate = model.gate_probabilities(routing_meta_ids)
    active_gate = gate.index_select(1, active_index)
    log_posterior = _log_gate(active_gate)
    options = {
        index: model.expert_options(index, validated, state, prefix, value_options)
        for index in active_experts
    }
    states = {
        index: model.experts[index].action_decoder.initialize(validated, state.summary)
        for index in active_experts
    }
    batch_size, option_count = validated.option_mask.shape
    actions = torch.full(
        (batch_size, max_select), -1, dtype=torch.long, device=model.device
    )
    lengths = torch.zeros(batch_size, dtype=torch.long, device=model.device)
    stopped = torch.zeros(batch_size, dtype=torch.bool, device=model.device)
    logprob = torch.zeros(batch_size, dtype=torch.float32, device=model.device)
    entropy = torch.zeros_like(logprob)
    maximum = validated.max_count.long().clamp(0, max_select)
    minimum = validated.min_count.long().clamp(0, max_select)
    active = maximum.gt(0)
    if not greedy:
        if sampling_seeds is None or sampling_counters is None:
            raise ValueError("0047 stochastic mixture decode requires seeds and counters")
        sampling_seeds = sampling_seeds.long().view(batch_size)
        sampling_counters = sampling_counters.long().view(batch_size)
    for step in range(max_select):
        expert_logps = []
        for index in active_experts:
            expert = model.experts[index]
            logits = expert.action_decoder.logits(validated, options[index], states[index])
            expert_logps.append(torch.log_softmax(logits.float(), dim=-1))
        stacked = torch.stack(expert_logps, dim=1)
        effective_log = torch.logsumexp(log_posterior.unsqueeze(-1) + stacked, dim=1)
        if greedy:
            choice = effective_log.argmax(dim=1)
        else:
            modulus = 2_147_483_647
            key = torch.remainder(sampling_seeds, modulus)
            key = torch.remainder(
                key + sampling_counters * 1_000_003 + (step + 1) * 9_176, modulus
            )
            for _ in range(3):
                key = torch.remainder(key * 48_271, modulus)
            uniform = (key.to(torch.float64) + 0.5) / modulus
            cumulative = effective_log.softmax(dim=1).double().cumsum(dim=1)
            choice = cumulative.ge(uniform.unsqueeze(1)).long().argmax(dim=1)
        distribution = Categorical(logits=effective_log)
        if compute_stats:
            logprob += torch.where(active, distribution.log_prob(choice), 0.0)
            entropy += torch.where(active, distribution.entropy(), 0.0)
        chosen_expert_log = stacked.gather(
            2, choice[:, None, None].expand(-1, len(active_experts), 1)
        ).squeeze(-1)
        log_posterior = torch.where(
            active[:, None], log_posterior + chosen_expert_log, log_posterior
        )
        log_posterior = log_posterior - torch.logsumexp(
            log_posterior, dim=1, keepdim=True
        )
        chosen_valid = active & choice.lt(option_count)
        stopped |= active & choice.eq(option_count)
        safe = choice.clamp(0, max(0, option_count - 1))
        actions[:, step] = torch.where(chosen_valid, safe, torch.full_like(safe, -1))
        for index in active_experts:
            expert = model.experts[index]
            states[index] = expert.action_decoder.consume(
                options[index], states[index], torch.where(chosen_valid, safe, -1)
            )
        lengths += chosen_valid.long()
        active = chosen_valid & lengths.lt(maximum)
    decoded = {
        "actions": actions, "lengths": lengths, "stopped": stopped,
        "logprob": logprob, "entropy": entropy,
    }
    posterior = torch.zeros_like(gate).index_copy(1, active_index, log_posterior.exp())
    return decoded, options, posterior


__all__ = [
    "MoEActionEvaluation", "decode_moe_device", "evaluate_moe_actions",
    "routing_groups", "select_rows",
]
