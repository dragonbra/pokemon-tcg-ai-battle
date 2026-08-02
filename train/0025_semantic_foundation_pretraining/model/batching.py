"""Ragged semantic decision collation with explicit masks."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor


def _pad_1d(records: Sequence[Mapping[str, Any]], path: tuple[str, ...], dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    values = []
    for record in records:
        value: Any = record
        for key in path:
            value = value[key]
        values.append(list(value))
    maximum = max(1, max(map(len, values)))
    result = torch.zeros((len(records), maximum), dtype=dtype)
    mask = torch.zeros((len(records), maximum), dtype=torch.bool)
    for index, row in enumerate(values):
        if row:
            result[index, : len(row)] = torch.tensor(row, dtype=dtype)
            mask[index, : len(row)] = True
    return result, mask


def _pad_2d(records: Sequence[Mapping[str, Any]], path: tuple[str, ...], width: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    rows = []
    for record in records:
        value: Any = record
        for key in path:
            value = value[key]
        rows.append(list(value))
    maximum = max(1, max(map(len, rows)))
    result = torch.zeros((len(records), maximum, width), dtype=dtype)
    mask = torch.zeros((len(records), maximum), dtype=torch.bool)
    for index, value in enumerate(rows):
        if value:
            result[index, : len(value)] = torch.tensor(value, dtype=dtype)
            mask[index, : len(value)] = True
    return result, mask


def collate(records: Sequence[Mapping[str, Any]]) -> dict[str, Tensor]:
    if not records:
        raise ValueError("cannot collate an empty batch")
    output: dict[str, Tensor] = {}
    for name, width, dtype in (
        ("entity_cat", 7, torch.long), ("entity_num", 5, torch.float32),
        ("option_cat", 12, torch.long),
    ):
        values, mask = _pad_2d(records, ("legacy", name), width, dtype)
        output[name] = values
        output[f"{name.rsplit('_', 1)[0]}_mask"] = mask
    output["global_cat"] = torch.tensor([row["legacy"]["global_cat"] for row in records], dtype=torch.long)
    output["global_num"] = torch.tensor([row["legacy"]["global_num"] for row in records], dtype=torch.float32)
    output["min_count"] = torch.tensor([row["legacy"]["min_count"] for row in records], dtype=torch.long)
    output["max_count"] = torch.tensor([row["legacy"]["max_count"] for row in records], dtype=torch.long)
    output["semantic_option_cat"], _ = _pad_2d(records, ("semantic_option_cat",), 16, torch.long)
    output["semantic_option_num"], _ = _pad_2d(records, ("semantic_option_num",), 14, torch.float32)
    output["semantic_option_state"], _ = _pad_2d(records, ("semantic_option_state",), 14, torch.long)
    output["prototype_card_refs"], output["prototype_card_mask"] = _pad_1d(records, ("prototype_card_refs",), torch.long)
    output["prototype_attack_refs"], output["prototype_attack_mask"] = _pad_1d(records, ("prototype_attack_refs",), torch.long)
    output["prototype_skill_refs"], output["prototype_skill_mask"] = _pad_1d(records, ("prototype_skill_refs",), torch.long)
    output["prototype_effect_refs"], output["prototype_effect_mask"] = _pad_1d(records, ("prototype_effect_refs",), torch.long)
    output["deck_card_ids"], output["deck_mask"] = _pad_1d(records, ("deck_card_ids",), torch.long)
    output["deck_multiplicity"], _ = _pad_1d(records, ("deck_multiplicity",), torch.float32)
    output["ledger_cat"], output["ledger_mask"] = _pad_2d(records, ("ledger_cat",), 4, torch.long)
    output["ledger_num"], _ = _pad_2d(records, ("ledger_num",), 15, torch.float32)
    output["event_cat"], output["event_mask"] = _pad_2d(records, ("event_cat",), 8, torch.long)
    output["event_num"], _ = _pad_2d(records, ("event_num",), 4, torch.float32)
    output["known_hand_ids"], output["known_hand_mask"] = _pad_1d(records, ("known_opponent_hand_card_ids",), torch.long)
    output["unknown_hand_count"] = torch.tensor([row["unknown_opponent_hand_count"] for row in records], dtype=torch.float32).unsqueeze(-1)
    output["turn_budget"] = torch.tensor([row["turn_budget"] for row in records], dtype=torch.float32)
    return output


__all__ = ["collate"]
