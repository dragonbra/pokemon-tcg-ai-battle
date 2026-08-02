"""Reusable typed field projections with one embedding per categorical field."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CategoricalFields(nn.Module):
    def __init__(self, vocabularies: tuple[int, ...], d_model: int):
        super().__init__()
        self.vocabularies = vocabularies
        self.fields = nn.ModuleList(
            nn.Embedding(size, d_model, padding_idx=0) for size in vocabularies
        )

    def forward(self, values: Tensor) -> Tensor:
        if values.size(-1) != len(self.fields):
            raise ValueError("categorical tensor width does not match its field specification")
        output = torch.zeros((*values.shape[:-1], self.fields[0].embedding_dim), device=values.device)
        for index, (embedding, vocabulary) in enumerate(zip(self.fields, self.vocabularies, strict=True)):
            field = values[..., index]
            if torch.any(field < 0) or torch.any(field >= vocabulary):
                raise ValueError(f"categorical field {index} exceeds vocabulary {vocabulary}")
            output = output + embedding(field)
        return output


class NumericFields(nn.Module):
    def __init__(self, width: int, d_model: int):
        super().__init__()
        self.width = width
        self.projection = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, values: Tensor) -> Tensor:
        if values.size(-1) != self.width:
            raise ValueError("numeric tensor width does not match its field specification")
        return self.projection(values)


def gather_relation(memory: Tensor, one_based_index: Tensor) -> Tensor:
    """Gather one-based relations; zero means no relation and returns zero."""
    maximum = memory.size(1)
    valid = one_based_index.gt(0) & one_based_index.le(maximum)
    index = (one_based_index - 1).clamp(min=0, max=max(0, maximum - 1))
    gathered = memory.gather(1, index.unsqueeze(-1).expand(-1, -1, memory.size(-1)))
    return gathered * valid.unsqueeze(-1)


def aggregate_relations(values: Tensor, parents: Tensor, mask: Tensor, option_count: int) -> Tensor:
    """Mean-pool explicitly parented relation tokens into their legal options."""
    batch, _, width = values.shape
    output = values.new_zeros((batch, option_count, width))
    counts = values.new_zeros((batch, option_count, 1))
    valid = mask & parents.gt(0) & parents.le(option_count)
    index = (parents - 1).clamp(min=0, max=max(0, option_count - 1))
    output.scatter_add_(1, index.unsqueeze(-1).expand(-1, -1, width), values * valid.unsqueeze(-1))
    counts.scatter_add_(1, index.unsqueeze(-1), valid.unsqueeze(-1).to(values.dtype))
    return output / counts.clamp_min(1.0)


__all__ = ["CategoricalFields", "NumericFields", "aggregate_relations", "gather_relation"]
