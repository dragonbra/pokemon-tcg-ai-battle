"""Autoregressive ordered legal-option pointer with explicit STOP."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..contracts.batch import DecisionBatch
from .config import ModelConfig


@dataclass(frozen=True, slots=True)
class DecoderState:
    hidden: Tensor
    available: Tensor
    selected_count: Tensor


@dataclass(frozen=True, slots=True)
class GreedyActions:
    sequences: Tensor
    lengths: Tensor
    legal: Tensor


class ActionDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.d_model
        self.config = config
        self.initial = nn.Linear(d, d)
        self.recurrent = nn.GRUCell(d, d)
        self.query = nn.Linear(d, d, bias=False)
        self.key = nn.Linear(d, d, bias=False)
        self.option_bias = nn.Linear(d, 1)
        self.stop = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def initialize(self, batch: DecisionBatch, state_summary: Tensor) -> DecoderState:
        return DecoderState(
            hidden=torch.tanh(self.initial(state_summary)),
            available=batch.option_mask.clone(),
            selected_count=torch.zeros(
                batch.batch_size,
                dtype=torch.long,
                device=state_summary.device,
            ),
        )

    def logits(
        self,
        batch: DecisionBatch,
        options: Tensor,
        state: DecoderState,
        *,
        option_keys: Tensor | None = None,
        option_bias: Tensor | None = None,
    ) -> Tensor:
        if option_keys is None:
            option_keys = self.key(options)
        if option_bias is None:
            option_bias = self.option_bias(options).squeeze(-1)
        pointer = (
            self.query(state.hidden).unsqueeze(1) * option_keys
        ).sum(-1) / math.sqrt(self.config.d_model)
        pointer = pointer + option_bias
        pointer = pointer.masked_fill(
            ~(
                state.available
                & state.selected_count.lt(batch.max_count).unsqueeze(1)
            ),
            torch.finfo(pointer.dtype).min,
        )
        stop = self.stop(state.hidden)
        stop = stop.masked_fill(
            ~state.selected_count.ge(batch.min_count).unsqueeze(1),
            torch.finfo(stop.dtype).min,
        )
        return torch.cat((pointer, stop), dim=1)

    def consume(
        self,
        options: Tensor,
        state: DecoderState,
        raw_index: Tensor,
    ) -> DecoderState:
        option_count = options.shape[1]
        valid = raw_index.ge(0) & raw_index.lt(option_count)
        index = raw_index.clamp(min=0, max=option_count - 1)
        selected = options.gather(
            1,
            index[:, None, None].expand(-1, 1, options.shape[-1]),
        ).squeeze(1)
        updated_hidden = self.recurrent(selected, state.hidden)
        hidden = torch.where(valid.unsqueeze(-1), updated_hidden, state.hidden)
        selected_mask = F.one_hot(index, num_classes=option_count).bool()
        selected_mask = selected_mask & valid.unsqueeze(1)
        available = state.available & ~selected_mask
        return DecoderState(
            hidden=hidden,
            available=available,
            selected_count=state.selected_count + valid.long(),
        )

    def consume_prefix(
        self,
        options: Tensor,
        state: DecoderState,
        selected_prefix: Tensor | None,
    ) -> DecoderState:
        if selected_prefix is None:
            return state
        for step in range(selected_prefix.shape[1]):
            state = self.consume(options, state, selected_prefix[:, step])
        return state

    def greedy(
        self,
        batch: DecisionBatch,
        options: Tensor,
        state_summary: Tensor,
    ) -> GreedyActions:
        state = self.initialize(batch, state_summary)
        maximum_steps = min(
            self.config.max_action_steps,
            batch.option_count,
            int(batch.max_count.max()),
        )
        sequences = torch.full(
            (batch.batch_size, maximum_steps),
            -1,
            dtype=torch.long,
            device=options.device,
        )
        lengths = torch.zeros(batch.batch_size, dtype=torch.long, device=options.device)
        legal = torch.ones(batch.batch_size, dtype=torch.bool, device=options.device)
        active = torch.ones(batch.batch_size, dtype=torch.bool, device=options.device)
        rows = torch.arange(batch.batch_size, device=options.device)

        for step in range(maximum_steps):
            scores = self.logits(batch, options, state)
            choice = scores.argmax(dim=1)
            chose_stop = choice.eq(batch.option_count)
            selecting = active & ~chose_stop
            legal &= ~(active & chose_stop & lengths.lt(batch.min_count))
            active &= ~chose_stop
            if not selecting.any():
                break

            chosen = choice.clamp_max(batch.option_count - 1)
            sequences[selecting, step] = chosen[selecting]
            selected = options[rows, chosen]
            updated_hidden = self.recurrent(selected, state.hidden)
            hidden = torch.where(selecting.unsqueeze(-1), updated_hidden, state.hidden)
            available = state.available.clone()
            available[rows[selecting], chosen[selecting]] = False
            lengths = lengths + selecting.long()
            active &= ~lengths.ge(batch.max_count)
            state = DecoderState(hidden, available, lengths)

        legal &= lengths.ge(batch.min_count) & lengths.le(batch.max_count)
        return GreedyActions(sequences, lengths, legal)


__all__ = ["ActionDecoder", "DecoderState", "GreedyActions"]
