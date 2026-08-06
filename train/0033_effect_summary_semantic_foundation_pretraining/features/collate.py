"""Ragged collation for the canonical actor contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from ..contracts.fields import ACTOR_KEYS, WIDTHS


def _pad_rows(values: Sequence[Sequence[Sequence[Any]]], width: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    maximum = max(1, max(len(row) for row in values))
    output = torch.zeros((len(values), maximum, width), dtype=dtype)
    mask = torch.zeros((len(values), maximum), dtype=torch.bool)
    for index, row in enumerate(values):
        if row:
            output[index, : len(row)] = torch.tensor(row, dtype=dtype)
            mask[index, : len(row)] = True
    return output, mask


def _pad_values(values: Sequence[Sequence[Any]], dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    maximum = max(1, max(len(row) for row in values))
    output = torch.zeros((len(values), maximum), dtype=dtype)
    mask = torch.zeros((len(values), maximum), dtype=torch.bool)
    for index, row in enumerate(values):
        if row:
            output[index, : len(row)] = torch.tensor(row, dtype=dtype)
            mask[index, : len(row)] = True
    return output, mask


def collate_canonical_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Tensor]:
    if not records:
        raise ValueError("cannot collate an empty canonical batch")
    actors: list[Mapping[str, Any]] = []
    actions: list[list[int]] = []
    for record in records:
        actor, target = record.get("actor"), record.get("target")
        if not isinstance(actor, Mapping) or set(actor) != ACTOR_KEYS:
            raise ValueError("canonical actor keys do not match the schema")
        if not isinstance(target, Mapping) or not isinstance(target.get("ordered_action"), list):
            raise ValueError("canonical record has no ordered action target")
        if "legacy" in actor or "action" in actor:
            raise ValueError("canonical actor contains a forbidden legacy/target field")
        option_count = len(actor["option_cat"])
        action = [int(value) for value in target["ordered_action"]]
        if option_count < 1 or len(set(action)) != len(action) or any(not 0 <= value < option_count for value in action):
            raise ValueError("canonical target is not a legal unique option sequence")
        if not int(actor["min_count"]) <= len(action) <= int(actor["max_count"]):
            raise ValueError("canonical target violates min/max count")
        actors.append(actor)
        actions.append(action)

    batch: dict[str, Tensor] = {
        "global_cat": torch.tensor([row["global_cat"] for row in actors], dtype=torch.long),
        "global_num": torch.tensor([row["global_num"] for row in actors], dtype=torch.float32),
        "global_state": torch.tensor([row["global_state"] for row in actors], dtype=torch.long),
        "min_count": torch.tensor([row["min_count"] for row in actors], dtype=torch.long),
        "max_count": torch.tensor([row["max_count"] for row in actors], dtype=torch.long),
    }
    for prefix, cat_width, num_width in (
        ("card", WIDTHS.card_cat, WIDTHS.card_num),
        ("resource", WIDTHS.resource_cat, WIDTHS.resource_num),
        ("event", WIDTHS.event_cat, WIDTHS.event_num),
        ("option", WIDTHS.option_cat, WIDTHS.option_num),
    ):
        batch[f"{prefix}_cat"], batch[f"{prefix}_mask"] = _pad_rows(
            [row[f"{prefix}_cat"] for row in actors], cat_width, torch.long
        )
        batch[f"{prefix}_num"], numeric_mask = _pad_rows(
            [row[f"{prefix}_num"] for row in actors], num_width, torch.float32
        )
        if not torch.equal(batch[f"{prefix}_mask"], numeric_mask):
            raise ValueError(f"canonical {prefix} categorical/numeric lengths disagree")
    for prefix, width in (
        ("card", WIDTHS.card_state),
        ("resource", WIDTHS.resource_state),
        ("event", WIDTHS.event_state),
        ("option", WIDTHS.option_state),
    ):
        batch[f"{prefix}_state"], state_mask = _pad_rows(
            [row[f"{prefix}_state"] for row in actors], width, torch.long
        )
        if not torch.equal(batch[f"{prefix}_mask"], state_mask):
            raise ValueError(f"canonical {prefix} state length disagrees")

    for name in ("card_parent", "event_source", "event_target", "option_source", "option_target"):
        batch[name], mask = _pad_values([row[name] for row in actors], torch.long)
        if name == "card_parent":
            expected = batch["card_mask"]
        elif name.startswith("event_"):
            expected = batch["event_mask"]
        else:
            expected = batch["option_mask"]
        if not torch.equal(mask, expected):
            raise ValueError(f"canonical relation length disagrees: {name}")
    option_width = batch["option_mask"].size(1)
    target_width = max(len(action) for action in actions) + 1
    targets = torch.full((len(records), target_width), -100, dtype=torch.long)
    for index, action in enumerate(actions):
        if action:
            targets[index, : len(action)] = torch.tensor(action, dtype=torch.long)
        targets[index, len(action)] = option_width
    batch["targets"] = targets
    return batch


__all__ = ["collate_canonical_records"]
