"""Independent categorical embeddings, numeric projections, and explicit relations."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CategoricalFields(nn.Module):
    def __init__(self, vocabularies: tuple[int, ...], d_model: int):
        super().__init__()
        if not vocabularies or min(vocabularies) < 1:
            raise ValueError("categorical vocabularies must be non-empty and positive")
        self.vocabularies = vocabularies
        offsets = []
        cursor = 0
        for size in vocabularies:
            offsets.append(cursor)
            cursor += size - 1
        self.register_buffer(
            "offsets", torch.tensor(offsets, dtype=torch.long), persistent=False
        )
        self.embedding = nn.Embedding(cursor + 1, d_model, padding_idx=0)

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[-1] != len(self.vocabularies):
            raise ValueError("categorical width does not match field specification")
        if values.device.type == "cpu":
            limits = values.new_tensor(self.vocabularies)
            if torch.any(values < 0) or torch.any(values >= limits):
                raise ValueError("categorical field exceeds its vocabulary")
        indices = values + self.offsets
        indices = indices.masked_fill(values.eq(0), 0)
        return self.embedding(indices).sum(dim=-2)


class NumericFields(nn.Module):
    """Project each scalar independently and keep its missingness attached."""

    def __init__(self, width: int, d_model: int):
        super().__init__()
        if width < 1:
            raise ValueError("numeric width must be positive")
        self.width = width
        self.d_model = d_model
        self.first_weight = nn.Parameter(torch.empty(width, d_model))
        self.first_bias = nn.Parameter(torch.empty(width, d_model))
        self.second_weight = nn.Parameter(torch.empty(width, d_model, d_model))
        self.second_bias = nn.Parameter(torch.empty(width, d_model))
        self.state_embedding = nn.Embedding(1 + width * 3, d_model, padding_idx=0)
        self.register_buffer(
            "state_offsets",
            torch.arange(width, dtype=torch.long) * 3,
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for index in range(self.width):
            nn.init.kaiming_uniform_(self.first_weight[index].unsqueeze(-1), a=5**0.5)
            bound = 1.0
            nn.init.uniform_(self.first_bias[index], -bound, bound)
            nn.init.kaiming_uniform_(self.second_weight[index], a=5**0.5)
            bound = 1.0 / self.d_model**0.5
            nn.init.uniform_(self.second_bias[index], -bound, bound)
        self.state_embedding.reset_parameters()

    def forward(self, values: Tensor, states: Tensor | None = None) -> Tensor:
        if values.shape[-1] != self.width:
            raise ValueError("numeric width does not match field specification")
        if states is None:
            states = values.new_ones(values.shape, dtype=torch.long)
        if states.shape != values.shape:
            raise ValueError("numeric values and field states must align")
        if states.device.type == "cpu" and (
            torch.any(states < 0) or torch.any(states >= 4)
        ):
            raise ValueError("numeric field state must be in [0, 3]")
        hidden = values.unsqueeze(-1) * self.first_weight + self.first_bias
        hidden = torch.nn.functional.gelu(hidden)
        projected = torch.einsum("...fi,foi->...fo", hidden, self.second_weight)
        projected = projected + self.second_bias
        projected = projected * states.eq(1).unsqueeze(-1)
        state_indices = states + self.state_offsets
        state_indices = state_indices.masked_fill(states.eq(0), 0)
        return (projected + self.state_embedding(state_indices)).sum(dim=-2)


def gather_one_based(memory: Tensor, one_based_index: Tensor) -> Tensor:
    """Gather explicit one-based relations; zero means no relation and yields zero."""
    maximum = memory.shape[1]
    valid = one_based_index.gt(0) & one_based_index.le(maximum)
    index = (one_based_index - 1).clamp(min=0, max=max(0, maximum - 1))
    gathered = memory.gather(1, index.unsqueeze(-1).expand(-1, -1, memory.shape[-1]))
    return gathered * valid.unsqueeze(-1)


def mean_pool_by_parent(
    values: Tensor,
    one_based_parent: Tensor,
    mask: Tensor,
    parent_count: int,
) -> Tensor:
    """Mean-pool relation tokens into their explicitly parented legal option."""
    batch, _, width = values.shape
    output = values.new_zeros((batch, parent_count, width))
    counts = values.new_zeros((batch, parent_count, 1))
    valid = mask & one_based_parent.gt(0) & one_based_parent.le(parent_count)
    index = (one_based_parent - 1).clamp(min=0, max=max(0, parent_count - 1))
    output.scatter_add_(
        1,
        index.unsqueeze(-1).expand(-1, -1, width),
        values * valid.unsqueeze(-1),
    )
    counts.scatter_add_(1, index.unsqueeze(-1), valid.unsqueeze(-1).to(values.dtype))
    return output / counts.clamp_min(1.0)


def sum_children_by_parent(values: Tensor, one_based_parent: Tensor, mask: Tensor) -> Tensor:
    """Aggregate child instance tokens into each card instance parent."""
    batch, count, width = values.shape
    output = values.new_zeros((batch, count, width))
    valid = mask & one_based_parent.gt(0) & one_based_parent.le(count)
    index = (one_based_parent - 1).clamp(min=0, max=max(0, count - 1))
    output.scatter_add_(1, index.unsqueeze(-1).expand(-1, -1, width), values * valid.unsqueeze(-1))
    return output


__all__ = [
    "CategoricalFields",
    "NumericFields",
    "gather_one_based",
    "mean_pool_by_parent",
    "sum_children_by_parent",
]
