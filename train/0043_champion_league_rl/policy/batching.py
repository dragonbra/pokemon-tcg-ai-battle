"""Pad and concatenate already-collated one-observation feature batches."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def _pad_second(value: Tensor, width: int, *, fill: int = 0) -> Tensor:
    if value.ndim < 2 or value.size(1) == width:
        return value
    shape = list(value.shape)
    shape[1] = width - value.size(1)
    padding = torch.full(shape, fill, dtype=value.dtype, device=value.device)
    return torch.cat((value, padding), dim=1)


def collate_feature_batches(items: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
    if not items:
        raise ValueError("cannot collate an empty feature batch")
    keys = set(items[0])
    if any(set(item) != keys for item in items):
        raise ValueError("feature batch keys disagree")
    output: dict[str, Tensor] = {}
    for key in sorted(keys):
        values = [item[key] for item in items]
        if values[0].ndim >= 2:
            width = max(value.size(1) for value in values)
            values = [_pad_second(value, width, fill=-100 if key == "targets" else 0) for value in values]
        output[key] = torch.cat(values, dim=0)
    return output


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def cpu_batch(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach().cpu() for key, value in batch.items()}


__all__ = ["collate_feature_batches", "cpu_batch", "move_batch"]
