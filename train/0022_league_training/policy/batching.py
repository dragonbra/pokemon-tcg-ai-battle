from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


_VARIABLE_SECOND_DIM = {
    "entity_cat",
    "entity_num",
    "entity_mask",
    "option_cat",
    "option_mask",
    "registered_card_ids",
    "registered_multiplicity",
    "registered_mask",
    "ledger_cat",
    "ledger_num",
    "ledger_mask",
    "event_cat",
    "event_num",
    "event_mask",
    "known_opponent_hand_card_ids",
    "known_opponent_hand_mask",
}


def _pad_second_dim(value: Tensor, size: int) -> Tensor:
    if value.size(1) == size:
        return value
    shape = list(value.shape)
    shape[1] = size - value.size(1)
    padding = torch.zeros(shape, dtype=value.dtype, device=value.device)
    return torch.cat((value, padding), dim=1)


def collate_feature_batches(items: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
    if not items:
        raise ValueError("cannot collate an empty feature batch")
    keys = set(items[0])
    if any(set(item) != keys for item in items):
        raise ValueError("feature batch keys disagree")
    result: dict[str, Tensor] = {}
    for key in sorted(keys):
        values = [item[key] for item in items]
        if key in _VARIABLE_SECOND_DIM:
            width = max(value.size(1) for value in values)
            values = [_pad_second_dim(value, width) for value in values]
        result[key] = torch.cat(values, dim=0)
    return result


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def cpu_batch(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach().cpu() for key, value in batch.items()}


__all__ = ["collate_feature_batches", "cpu_batch", "move_batch"]
