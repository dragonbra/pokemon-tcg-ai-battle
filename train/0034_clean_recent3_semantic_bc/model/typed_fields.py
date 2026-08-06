"""Independent categorical embeddings, numeric projections, and explicit relations."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CategoricalFields(nn.Module):
    def __init__(
        self,
        vocabularies: tuple[int, ...],
        d_model: int,
        *,
        skip_indices: tuple[int, ...] = (),
    ):
        super().__init__()
        self.vocabularies = vocabularies
        self.d_model = d_model
        self.skip_indices = frozenset(skip_indices)
        if any(not 0 <= index < len(vocabularies) for index in self.skip_indices):
            raise ValueError("categorical skip index is outside the field specification")
        self.active_indices = tuple(
            index for index in range(len(vocabularies)) if index not in self.skip_indices
        )
        if not self.active_indices:
            raise ValueError("categorical group cannot skip every field")
        self.fields = nn.ModuleList(
            nn.Embedding(vocabularies[index], d_model, padding_idx=0)
            for index in self.active_indices
        )

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[-1] != len(self.vocabularies):
            raise ValueError("categorical width does not match field specification")
        output = torch.zeros(
            (*values.shape[:-1], self.d_model),
            device=values.device,
            dtype=self.fields[0].weight.dtype,
        )
        for index, embedding in zip(self.active_indices, self.fields, strict=True):
            vocabulary = self.vocabularies[index]
            field = values[..., index]
            # Canonical batches are validated before transfer. Reading a CUDA
            # boolean here forces a host synchronization for every field.
            if field.device.type == "cpu" and (
                torch.any(field < 0) or torch.any(field >= vocabulary)
            ):
                raise ValueError(f"categorical field {index} exceeds vocabulary {vocabulary}")
            output = output + embedding(field)
        return output


class NumericFields(nn.Module):
    """Project each scalar independently and keep its missingness attached."""

    def __init__(self, width: int, d_model: int):
        super().__init__()
        self.width = width
        self.projections = nn.ModuleList(
            nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            for _ in range(width)
        )
        self.states = nn.ModuleList(
            nn.Embedding(4, d_model, padding_idx=0) for _ in range(width)
        )

    def forward(self, values: Tensor, states: Tensor | None = None) -> Tensor:
        if values.shape[-1] != self.width:
            raise ValueError("numeric width does not match field specification")
        if states is None:
            states = values.new_ones(values.shape, dtype=torch.long)
        if states.shape != values.shape:
            raise ValueError("numeric values and field states must align")
        output = values.new_zeros((*values.shape[:-1], self.projections[0][-1].out_features))
        for index, (projection, state_embedding) in enumerate(
            zip(self.projections, self.states, strict=True)
        ):
            state = states[..., index]
            present = state.eq(1).unsqueeze(-1)
            output = output + projection(values[..., index : index + 1]) * present
            output = output + state_embedding(state)
        return output


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
