"""Session-owned persistent Tensor materialization for one canonical record."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from ..contracts.fields import ACTOR_KEYS, WIDTHS


def _capacity(length: int) -> int:
    target = max(1, length)
    return 1 << (target - 1).bit_length()


class TensorBankStats:
    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()

    def add(self, name: str, value: int = 1) -> None:
        self._counts[name] += value

    def snapshot(self) -> dict[str, int]:
        result = dict(sorted(self._counts.items()))
        for name in (
            "collates", "allocations", "slot_hits", "slot_misses",
            "elements_written", "resets",
        ):
            result.setdefault(name, 0)
        return result


class _FixedSlot:
    def __init__(self, dtype: torch.dtype, stats: TensorBankStats):
        self.dtype = dtype
        self.stats = stats
        self.storage: Tensor | None = None
        self.previous: Any = None

    def update(self, value: Any) -> Tensor:
        frozen = tuple(value) if isinstance(value, (list, tuple)) else value
        if self.storage is not None and frozen == self.previous:
            self.stats.add("slot_hits")
            return self.storage
        source = torch.tensor([value], dtype=self.dtype)
        if self.storage is None or self.storage.shape != source.shape:
            self.storage = torch.empty_like(source)
            self.stats.add("allocations")
        self.storage.copy_(source)
        self.previous = frozen
        self.stats.add("slot_misses")
        self.stats.add("elements_written", source.numel())
        return self.storage


class _RaggedSlot:
    def __init__(
        self,
        dtype: torch.dtype,
        stats: TensorBankStats,
        *,
        width: int | None = None,
    ) -> None:
        self.dtype = dtype
        self.stats = stats
        self.width = width
        self.storage: Tensor | None = None
        self.capacity = 0
        self.length = -1
        self.previous: Any = None

    def _allocate(self, length: int) -> None:
        self.capacity = _capacity(length)
        shape = (
            (1, self.capacity, self.width)
            if self.width is not None else (1, self.capacity)
        )
        self.storage = torch.empty(shape, dtype=self.dtype)
        self.stats.add("allocations")

    def update(self, values: Sequence[Any]) -> Tensor:
        length = len(values)
        reallocated = False
        if self.storage is None or self.capacity < max(1, length):
            self._allocate(length)
            self.previous = None
            reallocated = True
        if (
            not reallocated
            and self.previous is not None
            and self.length == length
        ):
            changed = [
                index
                for index, value in enumerate(values)
                if value != self.previous[index]
            ]
        else:
            changed = list(range(length))
        if not changed and self.length == length:
            self.stats.add("slot_hits")
        else:
            if length:
                if changed:
                    sparse = len(changed) * 8 <= length
                    selected = (
                        [values[index] for index in changed] if sparse else values
                    )
                    source = torch.tensor(selected, dtype=self.dtype)
                    write_count = len(changed) if sparse else length
                    if (
                        self.width is not None
                        and source.shape != (write_count, self.width)
                    ):
                        raise ValueError(
                            f"persistent tensor row width mismatch: expected "
                            f"{(write_count, self.width)}, got {tuple(source.shape)}"
                        )
                    if sparse:
                        indices = torch.tensor(changed, dtype=torch.long)
                        self.storage[0].index_copy_(0, indices, source)
                    else:
                        self.storage[0, :length].copy_(source)
                    self.stats.add("elements_written", source.numel())
                    self.stats.add("rows_written", write_count)
            else:
                # The canonical collator represents an empty ragged field with
                # one zero-padded row.  A persistent buffer may still contain
                # values from the preceding decision, so zero that logical
                # placeholder explicitly instead of exposing stale storage.
                self.storage[0, 0].zero_()
                self.stats.add(
                    "elements_written", self.width if self.width is not None else 1
                )
            self.previous = values
            self.length = length
            self.stats.add("slot_misses")
        logical = max(1, length)
        if self.width is None:
            return self.storage[:, :logical]
        return self.storage[:, :logical, :]


class _MaskSlot:
    def __init__(self, stats: TensorBankStats):
        self.stats = stats
        self.storage: Tensor | None = None
        self.capacity = 0
        self.length = -1

    def update(self, length: int) -> Tensor:
        logical = max(1, length)
        if self.storage is None or self.capacity < logical:
            self.capacity = _capacity(logical)
            self.storage = torch.empty((1, self.capacity), dtype=torch.bool)
            self.stats.add("allocations")
            self.length = -1
        if self.length == length:
            self.stats.add("slot_hits")
        else:
            self.storage[0, :logical] = False
            if length:
                self.storage[0, :length] = True
            self.length = length
            self.stats.add("slot_misses")
            self.stats.add("elements_written", logical)
        return self.storage[:, :logical]


class PersistentTensorBank:
    """Materialize one battle's canonical records into reusable CPU storage.

    Returned tensors alias the bank and remain valid until the same bank is
    collated again.  Online inference consumes them synchronously before the
    next decision; callers that need a historical snapshot must clone it.
    """

    _FAMILIES = {
        "card": (WIDTHS.card_cat, WIDTHS.card_num, WIDTHS.card_state),
        "resource": (WIDTHS.resource_cat, WIDTHS.resource_num, WIDTHS.resource_state),
        "event": (WIDTHS.event_cat, WIDTHS.event_num, WIDTHS.event_state),
        "option": (WIDTHS.option_cat, WIDTHS.option_num, WIDTHS.option_state),
    }

    def __init__(self) -> None:
        self.stats = TensorBankStats()
        self._slots: dict[str, _FixedSlot | _RaggedSlot] = {}
        self._masks: dict[str, _MaskSlot] = {}

    def reset(self) -> None:
        self._slots.clear()
        self._masks.clear()
        self.stats.add("resets")

    def _fixed(self, name: str, dtype: torch.dtype, value: Any) -> Tensor:
        slot = self._slots.get(name)
        if slot is None:
            slot = _FixedSlot(dtype, self.stats)
            self._slots[name] = slot
        if not isinstance(slot, _FixedSlot) or slot.dtype != dtype:
            raise RuntimeError(f"persistent fixed slot contract changed: {name}")
        return slot.update(value)

    def _ragged(
        self,
        name: str,
        dtype: torch.dtype,
        values: Sequence[Any],
        *,
        width: int | None = None,
    ) -> Tensor:
        slot = self._slots.get(name)
        if slot is None:
            slot = _RaggedSlot(dtype, self.stats, width=width)
            self._slots[name] = slot
        if (
            not isinstance(slot, _RaggedSlot)
            or slot.dtype != dtype
            or slot.width != width
        ):
            raise RuntimeError(f"persistent ragged slot contract changed: {name}")
        return slot.update(values)

    def _mask(self, family: str, length: int) -> Tensor:
        slot = self._masks.get(family)
        if slot is None:
            slot = _MaskSlot(self.stats)
            self._masks[family] = slot
        return slot.update(length)

    @staticmethod
    def _validate(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], list[int]]:
        actor, target = record.get("actor"), record.get("target")
        if not isinstance(actor, Mapping) or set(actor) != ACTOR_KEYS:
            raise ValueError("canonical actor keys do not match the schema")
        if not isinstance(target, Mapping) or not isinstance(
            target.get("ordered_action"), list
        ):
            raise ValueError("canonical record has no ordered action target")
        option_count = len(actor["option_cat"])
        action = [int(value) for value in target["ordered_action"]]
        if (
            option_count < 1
            or len(set(action)) != len(action)
            or any(not 0 <= value < option_count for value in action)
        ):
            raise ValueError("canonical target is not a legal unique option sequence")
        if not int(actor["min_count"]) <= len(action) <= int(actor["max_count"]):
            raise ValueError("canonical target violates min/max count")
        return actor, action

    def collate(self, record: Mapping[str, Any]) -> dict[str, Tensor]:
        actor, action = self._validate(record)
        self.stats.add("collates")
        batch: dict[str, Tensor] = {
            "global_cat": self._fixed("global_cat", torch.long, actor["global_cat"]),
            "global_num": self._fixed("global_num", torch.float32, actor["global_num"]),
            "global_state": self._fixed("global_state", torch.long, actor["global_state"]),
            "min_count": self._fixed("min_count", torch.long, actor["min_count"]),
            "max_count": self._fixed("max_count", torch.long, actor["max_count"]),
        }
        for family, (cat_width, num_width, state_width) in self._FAMILIES.items():
            length = len(actor[f"{family}_cat"])
            if not (
                len(actor[f"{family}_num"]) == length
                and len(actor[f"{family}_state"]) == length
            ):
                raise ValueError(f"canonical {family} field lengths disagree")
            batch[f"{family}_cat"] = self._ragged(
                f"{family}_cat", torch.long, actor[f"{family}_cat"], width=cat_width
            )
            batch[f"{family}_num"] = self._ragged(
                f"{family}_num", torch.float32, actor[f"{family}_num"], width=num_width
            )
            batch[f"{family}_state"] = self._ragged(
                f"{family}_state", torch.long, actor[f"{family}_state"], width=state_width
            )
            batch[f"{family}_mask"] = self._mask(family, length)

        for name in (
            "card_parent",
            "event_source", "event_target", "event_before", "event_after",
            "option_source", "option_target", "option_context", "option_effect_card",
        ):
            family = (
                "card" if name == "card_parent"
                else "event" if name.startswith("event_") else "option"
            )
            if len(actor[name]) != len(actor[f"{family}_cat"]):
                raise ValueError(f"canonical relation length disagrees: {name}")
            batch[name] = self._ragged(name, torch.long, actor[name])

        for prefix in ("option_skill", "option_effect"):
            length = len(actor[f"{prefix}_id"])
            if not (
                len(actor[f"{prefix}_role"]) == length
                and len(actor[f"{prefix}_parent"]) == length
            ):
                raise ValueError(f"canonical {prefix} relation lengths disagree")
            for suffix in ("id", "role", "parent"):
                name = f"{prefix}_{suffix}"
                batch[name] = self._ragged(name, torch.long, actor[name])
            batch[f"{prefix}_mask"] = self._mask(prefix, length)

        option_width = batch["option_mask"].size(1)
        targets = [*action, option_width]
        batch["targets"] = self._ragged("targets", torch.long, targets)
        return batch


__all__ = ["PersistentTensorBank", "TensorBankStats"]
