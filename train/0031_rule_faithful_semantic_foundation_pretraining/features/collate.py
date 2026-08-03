"""Ragged collation for the canonical actor contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..contracts.fields import ACTOR_KEYS, WIDTHS


@dataclass(frozen=True)
class BucketPadding:
    """Finite shape families for optional torch.compile training."""

    card: tuple[int, ...] = (16, 32, 64, 96, 128)
    event: tuple[int, ...] = (8, 16, 32, 64)
    option: tuple[int, ...] = (4, 8, 16, 32, 64)
    effect: tuple[int, ...] = (8, 16, 32, 64, 96, 128)
    skill: tuple[int, ...] = (8, 16, 32, 64)
    action: tuple[int, ...] = (2, 4, 8, 16, 32, 64)

    def upper_bound(self, family: str, length: int) -> int:
        bounds = getattr(self, family)
        if (
            not isinstance(bounds, tuple)
            or not bounds
            or any(type(value) is not int or value < 1 for value in bounds)
            or tuple(sorted(set(bounds))) != bounds
        ):
            raise ValueError(f"invalid fixed bucket bounds for {family}: {bounds!r}")
        for bound in bounds:
            if length <= bound:
                return bound
        raise ValueError(
            f"canonical {family} length {length} exceeds fixed bucket maximum {bounds[-1]}"
        )


def _pad_rows(
    values: Sequence[Sequence[Sequence[Any]]],
    width: int,
    dtype: torch.dtype,
    *,
    pad_to: int | None = None,
) -> tuple[Tensor, Tensor]:
    observed = max(1, max(len(row) for row in values))
    maximum = observed if pad_to is None else pad_to
    if maximum < observed:
        raise ValueError(f"padding bound {maximum} is below observed length {observed}")
    output = torch.zeros((len(values), maximum, width), dtype=dtype)
    mask = torch.zeros((len(values), maximum), dtype=torch.bool)
    for index, row in enumerate(values):
        if row:
            output[index, : len(row)] = torch.tensor(row, dtype=dtype)
            mask[index, : len(row)] = True
    return output, mask


def _pad_values(
    values: Sequence[Sequence[Any]],
    dtype: torch.dtype,
    *,
    pad_to: int | None = None,
) -> tuple[Tensor, Tensor]:
    observed = max(1, max(len(row) for row in values))
    maximum = observed if pad_to is None else pad_to
    if maximum < observed:
        raise ValueError(f"padding bound {maximum} is below observed length {observed}")
    output = torch.zeros((len(values), maximum), dtype=dtype)
    mask = torch.zeros((len(values), maximum), dtype=torch.bool)
    for index, row in enumerate(values):
        if row:
            output[index, : len(row)] = torch.tensor(row, dtype=dtype)
            mask[index, : len(row)] = True
    return output, mask


def collate_canonical_records(
    records: Sequence[Mapping[str, Any]],
    *,
    bucket_padding: BucketPadding | None = None,
) -> dict[str, Tensor]:
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
    padded_lengths: dict[str, int] = {}
    if bucket_padding is not None:
        for family, actor_key in (
            ("card", "card_cat"),
            ("event", "event_cat"),
            ("option", "option_cat"),
            ("effect", "option_effect_id"),
        ):
            padded_lengths[family] = bucket_padding.upper_bound(
                family, max(1, max(len(row[actor_key]) for row in actors))
            )
        padded_lengths["skill"] = bucket_padding.upper_bound(
            "skill", max(1, max(len(row["option_skill_id"]) for row in actors))
        )
        padded_lengths["action"] = bucket_padding.upper_bound(
            "action", max(len(action) for action in actions) + 1
        )
    for prefix, cat_width, num_width in (
        ("card", WIDTHS.card_cat, WIDTHS.card_num),
        ("resource", WIDTHS.resource_cat, WIDTHS.resource_num),
        ("event", WIDTHS.event_cat, WIDTHS.event_num),
        ("option", WIDTHS.option_cat, WIDTHS.option_num),
    ):
        batch[f"{prefix}_cat"], batch[f"{prefix}_mask"] = _pad_rows(
            [row[f"{prefix}_cat"] for row in actors],
            cat_width,
            torch.long,
            pad_to=padded_lengths.get(prefix),
        )
        batch[f"{prefix}_num"], numeric_mask = _pad_rows(
            [row[f"{prefix}_num"] for row in actors],
            num_width,
            torch.float32,
            pad_to=padded_lengths.get(prefix),
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
            [row[f"{prefix}_state"] for row in actors],
            width,
            torch.long,
            pad_to=padded_lengths.get(prefix),
        )
        if not torch.equal(batch[f"{prefix}_mask"], state_mask):
            raise ValueError(f"canonical {prefix} state length disagrees")

    for name in (
        "card_parent",
        "event_source", "event_target", "event_before", "event_after",
        "option_source", "option_target", "option_context", "option_effect_card",
    ):
        family = "card" if name == "card_parent" else "event" if name.startswith("event_") else "option"
        batch[name], mask = _pad_values(
            [row[name] for row in actors],
            torch.long,
            pad_to=padded_lengths.get(family),
        )
        if name == "card_parent":
            expected = batch["card_mask"]
        elif name.startswith("event_"):
            expected = batch["event_mask"]
        else:
            expected = batch["option_mask"]
        if not torch.equal(mask, expected):
            raise ValueError(f"canonical relation length disagrees: {name}")
    for prefix in ("option_skill", "option_effect"):
        batch[f"{prefix}_id"], batch[f"{prefix}_mask"] = _pad_values(
            [row[f"{prefix}_id"] for row in actors],
            torch.long,
            pad_to=padded_lengths.get("skill" if prefix == "option_skill" else "effect"),
        )
        for suffix in ("role", "parent"):
            batch[f"{prefix}_{suffix}"], mask = _pad_values(
                [row[f"{prefix}_{suffix}"] for row in actors],
                torch.long,
                pad_to=padded_lengths.get("skill" if prefix == "option_skill" else "effect"),
            )
            if not torch.equal(batch[f"{prefix}_mask"], mask):
                raise ValueError(f"canonical {prefix} relation lengths disagree")

    option_width = batch["option_mask"].size(1)
    target_width = (
        padded_lengths["action"]
        if bucket_padding is not None
        else max(len(action) for action in actions) + 1
    )
    targets = torch.full((len(records), target_width), -100, dtype=torch.long)
    for index, action in enumerate(actions):
        if action:
            targets[index, : len(action)] = torch.tensor(action, dtype=torch.long)
        targets[index, len(action)] = option_width
    batch["targets"] = targets
    return batch


__all__ = ["BucketPadding", "collate_canonical_records"]
