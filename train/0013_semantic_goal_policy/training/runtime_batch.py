"""Lossless runtime compaction for materialized feature batches."""
from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor


def trim_target_padding(batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Drop shard-wide target padding that is unused by the current batch."""
    target_mask = batch["target_mask"]
    targets = batch["targets"]
    if target_mask.ndim != 2 or targets.shape != target_mask.shape:
        raise ValueError("targets and target_mask must be aligned rank-two tensors")
    maximum_steps = max(1, int(target_mask.sum(dim=1).max().item()))
    if maximum_steps >= targets.size(1):
        return dict(batch)
    compacted = dict(batch)
    compacted["targets"] = targets[:, :maximum_steps]
    compacted["target_mask"] = target_mask[:, :maximum_steps]
    return compacted


__all__ = ["trim_target_padding"]
