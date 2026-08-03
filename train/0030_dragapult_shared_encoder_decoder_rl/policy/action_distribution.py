"""Ordered action sampling/evaluation from cached 0028 decoder inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Categorical

from .actor_critic import DecoderPolicyHead, SemanticActorCritic


@dataclass(frozen=True)
class DecoderBatch:
    option_mask: Tensor
    min_count: Tensor
    max_count: Tensor

    @property
    def batch_size(self) -> int: return self.option_mask.size(0)
    @property
    def option_count(self) -> int: return self.option_mask.size(1)


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


def decoder_batch(batch) -> DecoderBatch:
    return DecoderBatch(batch.option_mask, batch.min_count, batch.max_count)


def cached_features(batch, summary: Tensor, options: Tensor) -> dict[str, Tensor]:
    return {
        "summary": summary,
        "options": options,
        "option_mask": batch.option_mask,
        "min_count": batch.min_count,
        "max_count": batch.max_count,
    }


def _sample(head: DecoderPolicyHead, batch: DecoderBatch, summary: Tensor, options: Tensor, *, greedy: bool, values: Tensor | None = None) -> list[SampledAction]:
    decoder = head.action_decoder
    state = decoder.initialize(batch, summary)
    maximum_steps = min(decoder.config.max_action_steps, batch.option_count, int(batch.max_count.max()))
    sequences = torch.full((batch.batch_size, maximum_steps), -1, dtype=torch.long, device=options.device)
    lengths = torch.zeros(batch.batch_size, dtype=torch.long, device=options.device)
    stopped = torch.zeros(batch.batch_size, dtype=torch.bool, device=options.device)
    finished = batch.max_count.eq(0)
    log_prob = torch.zeros(batch.batch_size, device=options.device)
    entropy = torch.zeros_like(log_prob)
    for step in range(maximum_steps):
        active = ~finished
        logits = decoder.logits(batch, options, state)
        distribution = Categorical(logits=logits.float())
        token = logits.argmax(1) if greedy else distribution.sample()
        log_prob += torch.where(active, distribution.log_prob(token), 0.0)
        entropy += torch.where(active, distribution.entropy(), 0.0)
        choose_stop = active & token.eq(batch.option_count)
        choose_option = active & ~choose_stop
        raw = torch.where(choose_option, token.clamp_max(batch.option_count - 1), -1)
        sequences[:, step] = raw
        state = decoder.consume(options, state, raw)
        lengths = state.selected_count
        stopped |= choose_stop
        finished |= choose_stop | lengths.ge(batch.max_count)
        if finished.all(): break
    if lengths.lt(batch.min_count).any():
        raise RuntimeError("sampled action violated min_count")
    if values is None:
        values = torch.zeros(batch.batch_size, device=options.device)
    return [SampledAction(tuple(int(v) for v in sequences[i, :lengths[i]].tolist()), bool(stopped[i]), float(log_prob[i]), float(entropy[i]), float(values[i])) for i in range(batch.batch_size)]


def infer_actions(model: SemanticActorCritic, batch, summary: Tensor, options: Tensor, *, focal: Tensor, greedy: bool) -> list[SampledAction]:
    compact = decoder_batch(batch)
    values = model.value_head(summary).squeeze(-1)
    output: list[SampledAction | None] = [None] * batch.batch_size
    for route, head in ((True, model.head), (False, model.opponent_head)):
        indices = torch.nonzero(focal.eq(route), as_tuple=False).flatten()
        if not len(indices): continue
        routed = DecoderBatch(compact.option_mask[indices], compact.min_count[indices], compact.max_count[indices])
        actions = _sample(head, routed, summary[indices], options[indices], greedy=greedy if route else True, values=values[indices] if route else None)
        for index, action in zip(indices.tolist(), actions, strict=True): output[index] = action
    return [value for value in output if value is not None]


def evaluate_actions_encoded(head: DecoderPolicyHead, features: dict[str, Tensor], sequences: Tensor, lengths: Tensor, stopped: Tensor) -> ActionEvaluation:
    summary, options = features["summary"], features["options"]
    batch = DecoderBatch(features["option_mask"], features["min_count"], features["max_count"])
    decoder = head.action_decoder
    state = decoder.initialize(batch, summary)
    total_log_prob = torch.zeros(batch.batch_size, device=summary.device)
    total_entropy = torch.zeros_like(total_log_prob)
    steps = max(sequences.size(1), int(lengths.max()) + int(stopped.any()))
    for step in range(steps):
        choosing_option = lengths.gt(step)
        choosing_stop = stopped & lengths.eq(step)
        active = choosing_option | choosing_stop
        if not active.any(): continue
        logits = decoder.logits(batch, options, state)
        raw = sequences[:, step] if step < sequences.size(1) else torch.full_like(lengths, -1)
        token = torch.where(choosing_option, raw, batch.option_count)
        distribution = Categorical(logits=logits.float())
        total_log_prob += torch.where(active, distribution.log_prob(token), 0.0)
        total_entropy += torch.where(active, distribution.entropy(), 0.0)
        state = decoder.consume(options, state, torch.where(choosing_option, raw, -1))
    if head.value_head is None:
        value = torch.zeros(batch.batch_size, device=summary.device)
    else:
        value = head.value_head(summary).squeeze(-1)
    return ActionEvaluation(total_log_prob, total_entropy, value)


__all__ = ["ActionEvaluation", "SampledAction", "cached_features", "evaluate_actions_encoded", "infer_actions"]
