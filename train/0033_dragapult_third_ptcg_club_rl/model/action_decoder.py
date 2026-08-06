"""Autoregressive ordered legal-option pointer with explicit STOP."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..contract import PodNativeBatch
from .config import ModelConfig


@dataclass(frozen=True, slots=True)
class DecoderState:
    hidden: Tensor
    available: Tensor
    selected_count: Tensor


@dataclass(frozen=True, slots=True)
class OrderedActions:
    sequences: Tensor
    lengths: Tensor
    legal: Tensor
    logprob: Tensor | None = None


@dataclass(frozen=True, slots=True)
class ActionEvaluation:
    logprob: Tensor
    entropy: Tensor


class ActionDecoder(nn.Module):
    """Parameter-compatible copy of the 0031 ordered action decoder."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.d_model
        self.config = config
        self.initial = nn.Linear(d, d)
        self.recurrent = nn.GRUCell(d, d)
        self.query = nn.Linear(d, d, bias=False)
        self.key = nn.Linear(d, d, bias=False)
        self.option_bias = nn.Linear(d, 1)
        self.stop = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def initialize(self, batch: PodNativeBatch, state_summary: Tensor) -> DecoderState:
        return DecoderState(
            hidden=torch.tanh(self.initial(state_summary)),
            available=batch.option_mask.clone(),
            selected_count=torch.zeros(
                batch.batch_size, dtype=torch.long, device=state_summary.device
            ),
        )

    def logits(
        self,
        batch: PodNativeBatch,
        options: Tensor,
        state: DecoderState,
        *,
        option_keys: Tensor | None = None,
        option_bias: Tensor | None = None,
    ) -> Tensor:
        option_keys = self.key(options) if option_keys is None else option_keys
        option_bias = (
            self.option_bias(options).squeeze(-1) if option_bias is None else option_bias
        )
        pointer = (
            self.query(state.hidden).unsqueeze(1) * option_keys
        ).sum(-1) / math.sqrt(self.config.d_model)
        pointer = pointer + option_bias
        pointer = pointer.masked_fill(
            ~(state.available & state.selected_count.lt(batch.max_count).unsqueeze(1)),
            torch.finfo(pointer.dtype).min,
        )
        stop = self.stop(state.hidden)
        stop = stop.masked_fill(
            ~state.selected_count.ge(batch.min_count).unsqueeze(1),
            torch.finfo(stop.dtype).min,
        )
        return torch.cat((pointer, stop), dim=1)

    def consume(
        self, options: Tensor, state: DecoderState, raw_index: Tensor
    ) -> DecoderState:
        option_count = options.shape[1]
        valid = raw_index.ge(0) & raw_index.lt(option_count)
        index = raw_index.clamp(min=0, max=option_count - 1)
        selected = options.gather(
            1, index[:, None, None].expand(-1, 1, options.shape[-1])
        ).squeeze(1)
        updated_hidden = self.recurrent(selected, state.hidden)
        hidden = torch.where(valid.unsqueeze(-1), updated_hidden, state.hidden)
        selected_mask = F.one_hot(index, num_classes=option_count).bool()
        selected_mask = selected_mask & valid.unsqueeze(1)
        return DecoderState(
            hidden=hidden,
            available=state.available & ~selected_mask,
            selected_count=state.selected_count + valid.long(),
        )

    def consume_prefix(
        self, options: Tensor, state: DecoderState, selected_prefix: Tensor | None
    ) -> DecoderState:
        if selected_prefix is not None:
            for step in range(selected_prefix.shape[1]):
                state = self.consume(options, state, selected_prefix[:, step])
        return state

    def generate(
        self,
        batch: PodNativeBatch,
        options: Tensor,
        state_summary: Tensor,
        *,
        stochastic: bool,
    ) -> OrderedActions:
        state = self.initialize(batch, state_summary)
        batch_size = batch.batch_size
        option_count = batch.option_count
        steps = self.config.max_action_steps
        sequences = torch.full(
            (batch_size, steps), -1, dtype=torch.long, device=options.device
        )
        lengths = torch.zeros(batch_size, dtype=torch.long, device=options.device)
        active = torch.ones(batch_size, dtype=torch.bool, device=options.device)
        logprob = torch.zeros(batch_size, dtype=options.dtype, device=options.device)
        option_keys = self.key(options)
        option_bias = self.option_bias(options).squeeze(-1)

        for step in range(steps):
            scores = self.logits(
                batch,
                options,
                state,
                option_keys=option_keys,
                option_bias=option_bias,
            )
            # Finished rows are forced to STOP, keeping the loop device-resident.
            scores = scores.masked_fill(~active.unsqueeze(1), torch.finfo(scores.dtype).min)
            scores[:, option_count] = torch.where(
                active, scores[:, option_count], torch.zeros_like(scores[:, option_count])
            )
            if stochastic:
                # Gumbel-max is exact categorical sampling and remains capturable.
                gumbel = torch.empty_like(scores).exponential_().log().neg()
                choice = (scores + gumbel).argmax(dim=1)
                selected_logprob = torch.log_softmax(scores.float(), dim=1).gather(
                    1, choice.unsqueeze(1)
                ).squeeze(1)
                logprob = logprob + torch.where(
                    active, selected_logprob, torch.zeros_like(logprob)
                )
            else:
                choice = scores.argmax(dim=1)
            chose_stop = choice.eq(option_count)
            selecting = active & ~chose_stop
            chosen = choice.clamp_max(option_count - 1)
            sequences[:, step] = torch.where(
                selecting, chosen, sequences[:, step]
            )
            lengths = lengths + selecting.long()
            state = self.consume(
                options,
                state,
                torch.where(selecting, chosen, torch.full_like(chosen, -1)),
            )
            active = selecting & lengths.lt(batch.max_count)

        legal = lengths.ge(batch.min_count) & lengths.le(batch.max_count)
        return OrderedActions(sequences, lengths, legal, logprob)

    def greedy(
        self, batch: PodNativeBatch, options: Tensor, state_summary: Tensor
    ) -> OrderedActions:
        return self.generate(batch, options, state_summary, stochastic=False)

    def evaluate(
        self,
        batch: PodNativeBatch,
        options: Tensor,
        state_summary: Tensor,
        sequences: Tensor,
        lengths: Tensor,
    ) -> ActionEvaluation:
        if sequences.shape != (batch.batch_size, self.config.max_action_steps):
            raise ValueError("sequences do not match the fixed action contract")
        state = self.initialize(batch, state_summary)
        logprob = torch.zeros(batch.batch_size, dtype=options.dtype, device=options.device)
        entropy = torch.zeros_like(logprob)
        option_count = batch.option_count
        rows = torch.arange(batch.batch_size, device=options.device)
        option_keys = self.key(options)
        option_bias = self.option_bias(options).squeeze(-1)
        stopped = lengths.lt(batch.max_count)
        for step in range(self.config.max_action_steps):
            active_option = step < lengths
            active_stop = stopped & (step == lengths)
            active = active_option | active_stop
            scores = self.logits(
                batch,
                options,
                state,
                option_keys=option_keys,
                option_bias=option_bias,
            )
            safe = sequences[:, step].clamp(min=0, max=option_count - 1)
            target = torch.where(active_stop, torch.full_like(safe, option_count), safe)
            distribution = torch.distributions.Categorical(logits=scores.float())
            logprob += torch.where(active, distribution.log_prob(target), 0.0)
            entropy += torch.where(active, distribution.entropy(), 0.0)
            selected = options[rows, safe]
            advanced = self.recurrent(selected, state.hidden)
            valid = active_option
            selected_mask = F.one_hot(safe, num_classes=option_count).bool()
            state = DecoderState(
                hidden=torch.where(valid.unsqueeze(1), advanced, state.hidden),
                available=state.available & ~(selected_mask & valid.unsqueeze(1)),
                selected_count=state.selected_count + valid.long(),
            )
        return ActionEvaluation(logprob=logprob, entropy=entropy)


__all__ = ["ActionDecoder", "ActionEvaluation", "DecoderState", "OrderedActions"]
