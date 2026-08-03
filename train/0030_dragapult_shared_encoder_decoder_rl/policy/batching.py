"""Collation for canonical observations and cached decoder inputs."""

from __future__ import annotations

import torch


def _pad(value: torch.Tensor, width: int) -> torch.Tensor:
    if value.ndim < 2 or value.size(1) == width: return value
    shape = list(value.shape); shape[1] = width - value.size(1)
    return torch.cat((value, torch.zeros(shape, dtype=value.dtype, device=value.device)), 1)


def collate_feature_batches(items):
    keys = set(items[0])
    output = {}
    for key in sorted(keys):
        values = [item[key] for item in items]
        if values[0].ndim >= 2:
            width = max(value.size(1) for value in values)
            values = [_pad(value, width) for value in values]
        output[key] = torch.cat(values, 0)
    return output


def move_batch(batch, device): return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
def cpu_batch(batch): return {key: value.detach().cpu() for key, value in batch.items()}


__all__ = ["collate_feature_batches", "cpu_batch", "move_batch"]
