from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor


INPUT_KEYS = (
    "state_numeric",
    "state_card_ids",
    "action_type_ids",
    "action_card_ids",
    "action_target_ids",
    "action_numeric",
    "action_mask",
)

UNIVERSAL_INPUT_KEYS = (
    "deck_card_ids",
    "deck_card_numeric",
    "entity_card_ids",
    "entity_numeric",
    "history_card_ids",
    "history_numeric",
    "expert_ids",
)


def collate_encoded(samples: Sequence[Mapping[str, object]]) -> dict[str, Tensor]:
    """Stack encoded observations into tensors matching the model contract."""
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    missing = [key for key in INPUT_KEYS if key not in samples[0]]
    if missing:
        raise ValueError(f"encoded sample is missing keys: {missing}")
    result: dict[str, Tensor] = {}
    keys = INPUT_KEYS + tuple(key for key in UNIVERSAL_INPUT_KEYS if key in samples[0])
    for key in keys:
        values = [sample[key] for sample in samples]
        dtype = torch.bool if key == "action_mask" else (
            torch.long if key.endswith("_ids") else torch.float32
        )
        result[key] = torch.tensor(values, dtype=dtype)
    return result
