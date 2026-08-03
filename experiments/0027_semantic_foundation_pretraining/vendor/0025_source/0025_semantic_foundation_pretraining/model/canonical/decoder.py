"""Autoregressive ordered-option pointer with an explicit STOP action."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .config import CanonicalModelConfig


@dataclass(frozen=True)
class GreedyActions:
    sequences: Tensor
    lengths: Tensor
    legal: Tensor


class OrderedOptionDecoder(nn.Module):
    def __init__(self, config: CanonicalModelConfig):
        super().__init__()
        d = config.d_model
        self.config = config
        self.initial = nn.Linear(d, d)
        self.recurrent = nn.GRUCell(d, d)
        self.query = nn.Linear(d, d, bias=False)
        self.key = nn.Linear(d, d, bias=False)
        self.option_bias = nn.Linear(d, 1)
        self.stop = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def initial_hidden(self, state_summary: Tensor) -> Tensor:
        return torch.tanh(self.initial(state_summary))

    def logits(
        self,
        batch: dict[str, Tensor],
        options: Tensor,
        hidden: Tensor,
        available: Tensor,
        selected_count: Tensor,
    ) -> Tensor:
        pointer = (
            self.query(hidden).unsqueeze(1) * self.key(options)
        ).sum(-1) / math.sqrt(self.config.d_model)
        pointer = pointer + self.option_bias(options).squeeze(-1)
        pointer = pointer.masked_fill(
            ~(available & selected_count.lt(batch["max_count"]).unsqueeze(1)),
            torch.finfo(pointer.dtype).min,
        )
        stop = self.stop(hidden)
        stop = stop.masked_fill(
            ~selected_count.ge(batch["min_count"]).unsqueeze(1),
            torch.finfo(stop.dtype).min,
        )
        return torch.cat((pointer, stop), dim=1)

    def consume(
        self,
        options: Tensor,
        hidden: Tensor,
        available: Tensor,
        selected_count: Tensor,
        raw_index: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        count = options.size(1)
        valid = raw_index.ge(0) & raw_index.lt(count)
        index = raw_index.clamp(min=0, max=count - 1)
        selected = options.gather(
            1, index[:, None, None].expand(-1, 1, options.size(-1))
        ).squeeze(1)
        updated = self.recurrent(selected, hidden)
        hidden = torch.where(valid.unsqueeze(-1), updated, hidden)
        available = available.clone()
        rows = torch.arange(options.size(0), device=options.device)
        available[rows[valid], index[valid]] = False
        return hidden, available, selected_count + valid.long()

    def greedy(
        self, batch: dict[str, Tensor], options: Tensor, state_summary: Tensor
    ) -> GreedyActions:
        hidden = self.initial_hidden(state_summary)
        batch_size, option_count, _ = options.shape
        available = batch["option_mask"].clone()
        lengths = torch.zeros(batch_size, dtype=torch.long, device=options.device)
        legal = torch.ones(batch_size, dtype=torch.bool, device=options.device)
        active = torch.ones(batch_size, dtype=torch.bool, device=options.device)
        maximum_steps = min(
            self.config.max_action_steps,
            option_count,
            int(batch["max_count"].max().item()),
        )
        sequences = torch.full(
            (batch_size, maximum_steps), -1, dtype=torch.long, device=options.device
        )
        rows = torch.arange(batch_size, device=options.device)
        for step in range(maximum_steps):
            scores = self.logits(batch, options, hidden, available, lengths)
            choice = scores.argmax(dim=1)
            chose_stop = choice.eq(option_count)
            selecting = active & ~chose_stop
            legal &= ~(active & chose_stop & lengths.lt(batch["min_count"]))
            active &= ~chose_stop
            if not selecting.any():
                break
            chosen = choice.clamp_max(option_count - 1)
            sequences[selecting, step] = chosen[selecting]
            selected = options[rows, chosen]
            updated = self.recurrent(selected, hidden)
            hidden = torch.where(selecting.unsqueeze(-1), updated, hidden)
            available[rows[selecting], chosen[selecting]] = False
            lengths += selecting.long()
            reached_max = lengths.ge(batch["max_count"])
            active &= ~reached_max
        legal &= lengths.ge(batch["min_count"]) & lengths.le(batch["max_count"])
        return GreedyActions(sequences, lengths, legal)


__all__ = ["GreedyActions", "OrderedOptionDecoder"]
