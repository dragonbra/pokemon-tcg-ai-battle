"""Session-owned persistent Tensor materialization for one canonical record."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..contracts.fields import ACTOR_KEYS, WIDTHS


def _capacity(length: int) -> int:
    target = max(1, length)
    return 1 << (target - 1).bit_length()


@dataclass(frozen=True, slots=True)
class TensorPatch:
    """One exact update to a session-resident canonical tensor.

    Ragged patches address rows along dimension one because dimension zero is
    the singleton online batch.  A replacement carries the complete logical
    tensor and therefore has no row indices.
    """

    name: str
    logical_shape: tuple[int, ...]
    rows: Tensor
    values: Tensor
    replace: bool


@dataclass(frozen=True, slots=True)
class TensorDelta:
    """Current aliasing CPU tensors plus self-contained changed-row payloads."""

    tensors: Mapping[str, Tensor]
    patches: tuple[TensorPatch, ...]
    full_reseed: bool


@dataclass(frozen=True, slots=True)
class _SlotResult:
    tensor: Tensor
    rows: tuple[int, ...]
    replace: bool
    logical_shape_changed: bool


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

    def update(self, value: Any) -> _SlotResult:
        frozen = tuple(value) if isinstance(value, (list, tuple)) else value
        if self.storage is not None and frozen == self.previous:
            self.stats.add("slot_hits")
            return _SlotResult(self.storage, (), False, False)
        source = torch.tensor([value], dtype=self.dtype)
        replace = self.storage is None or self.storage.shape != source.shape
        if replace:
            self.storage = torch.empty_like(source)
            self.stats.add("allocations")
        self.storage.copy_(source)
        self.previous = frozen
        self.stats.add("slot_misses")
        self.stats.add("elements_written", source.numel())
        return _SlotResult(self.storage, (), True, replace)


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

    def update(self, values: Sequence[Any]) -> _SlotResult:
        length = len(values)
        previous_length = self.length
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
                changed = [0]
                self.stats.add(
                    "elements_written", self.width if self.width is not None else 1
                )
            self.previous = values
            self.length = length
            self.stats.add("slot_misses")
        logical = max(1, length)
        if self.width is None:
            tensor = self.storage[:, :logical]
        else:
            tensor = self.storage[:, :logical, :]
        return _SlotResult(
            tensor,
            tuple(changed),
            reallocated,
            previous_length != length,
        )


class _MaskSlot:
    def __init__(self, stats: TensorBankStats):
        self.stats = stats
        self.storage: Tensor | None = None
        self.capacity = 0
        self.length = -1

    def update(self, length: int) -> _SlotResult:
        logical = max(1, length)
        previous_length = self.length
        reallocated = False
        if self.storage is None or self.capacity < logical:
            self.capacity = _capacity(logical)
            self.storage = torch.empty((1, self.capacity), dtype=torch.bool)
            self.stats.add("allocations")
            self.length = -1
            reallocated = True
        if self.length == length:
            self.stats.add("slot_hits")
            rows: tuple[int, ...] = ()
        else:
            self.storage[0, :logical] = False
            if length:
                self.storage[0, :length] = True
            self.length = length
            self.stats.add("slot_misses")
            self.stats.add("elements_written", logical)
            rows = tuple(range(logical))
        return _SlotResult(
            self.storage[:, :logical],
            rows,
            reallocated,
            previous_length != length,
        )


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
        self._pending: dict[str, _SlotResult] = {}
        self._seeded = False

    def reset(self) -> None:
        self._slots.clear()
        self._masks.clear()
        self._pending.clear()
        self._seeded = False
        self.stats.add("resets")

    def _remember(self, name: str, result: _SlotResult) -> Tensor:
        if result.replace or result.rows or result.logical_shape_changed:
            self._pending[name] = result
        return result.tensor

    def _fixed(self, name: str, dtype: torch.dtype, value: Any) -> Tensor:
        slot = self._slots.get(name)
        if slot is None:
            slot = _FixedSlot(dtype, self.stats)
            self._slots[name] = slot
        if not isinstance(slot, _FixedSlot) or slot.dtype != dtype:
            raise RuntimeError(f"persistent fixed slot contract changed: {name}")
        return self._remember(name, slot.update(value))

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
        return self._remember(name, slot.update(values))

    def _mask(self, family: str, length: int) -> Tensor:
        slot = self._masks.get(family)
        if slot is None:
            slot = _MaskSlot(self.stats)
            self._masks[family] = slot
        return self._remember(f"{family}_mask", slot.update(length))

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
        return dict(self.collate_delta(record).tensors)

    def collate_delta(self, record: Mapping[str, Any]) -> TensorDelta:
        actor, action = self._validate(record)
        self.stats.add("collates")
        self._pending = {}
        full_reseed = not self._seeded
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
        if full_reseed:
            patches = tuple(
                TensorPatch(
                    name=name,
                    logical_shape=tuple(tensor.shape),
                    rows=torch.empty(0, dtype=torch.long),
                    values=tensor.clone(),
                    replace=True,
                )
                for name, tensor in batch.items()
            )
        else:
            patches_list: list[TensorPatch] = []
            for name in batch:
                result = self._pending.get(name)
                if result is None:
                    continue
                rows = torch.empty(0, dtype=torch.long) if result.replace else torch.tensor(
                    result.rows, dtype=torch.long
                )
                values = (
                    result.tensor.clone()
                    if result.replace
                    else result.tensor[0].index_select(0, rows).clone()
                    if rows.numel()
                    else result.tensor.new_empty(
                        (0, *result.tensor.shape[2:])
                    )
                )
                patches_list.append(
                    TensorPatch(
                        name=name,
                        logical_shape=tuple(result.tensor.shape),
                        rows=rows,
                        values=values,
                        replace=result.replace,
                    )
                )
            patches = tuple(patches_list)
        self._seeded = True
        return TensorDelta(batch, patches, full_reseed)


EVENT_TENSOR_KEYS = frozenset(
    {
        "event_cat", "event_num", "event_state", "event_mask",
        "event_source", "event_target", "event_before", "event_after",
    }
)


class EventTensorBank:
    """Session delta authority limited to the large mostly-static event family."""

    def __init__(self) -> None:
        self.stats = TensorBankStats()
        self._slots: dict[str, _RaggedSlot] = {}
        self._mask_slot = _MaskSlot(self.stats)
        self._pending: dict[str, _SlotResult] = {}
        self._seeded = False

    def reset(self) -> None:
        self._slots.clear()
        self._mask_slot = _MaskSlot(self.stats)
        self._pending.clear()
        self._seeded = False
        self.stats.add("resets")

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
        if slot.dtype != dtype or slot.width != width:
            raise RuntimeError(f"event tensor slot contract changed: {name}")
        result = slot.update(values)
        if result.replace or result.rows or result.logical_shape_changed:
            self._pending[name] = result
        return result.tensor

    def collate_delta(self, record: Mapping[str, Any]) -> TensorDelta:
        actor, _action = PersistentTensorBank._validate(record)
        length = len(actor["event_cat"])
        if not (
            len(actor["event_num"]) == length
            and len(actor["event_state"]) == length
            and all(
                len(actor[name]) == length
                for name in (
                    "event_source", "event_target", "event_before", "event_after"
                )
            )
        ):
            raise ValueError("canonical event field lengths disagree")
        self.stats.add("collates")
        self._pending = {}
        full_reseed = not self._seeded
        batch = {
            "event_cat": self._ragged(
                "event_cat", torch.long, actor["event_cat"], width=WIDTHS.event_cat
            ),
            "event_num": self._ragged(
                "event_num", torch.float32, actor["event_num"], width=WIDTHS.event_num
            ),
            "event_state": self._ragged(
                "event_state", torch.long, actor["event_state"], width=WIDTHS.event_state
            ),
        }
        mask_result = self._mask_slot.update(length)
        if (
            mask_result.replace
            or mask_result.rows
            or mask_result.logical_shape_changed
        ):
            self._pending["event_mask"] = mask_result
        batch["event_mask"] = mask_result.tensor
        for name in (
            "event_source", "event_target", "event_before", "event_after"
        ):
            batch[name] = self._ragged(name, torch.long, actor[name])

        if full_reseed:
            patches = tuple(
                TensorPatch(
                    name,
                    tuple(tensor.shape),
                    torch.empty(0, dtype=torch.long),
                    tensor.clone(),
                    True,
                )
                for name, tensor in batch.items()
            )
        else:
            output: list[TensorPatch] = []
            for name, tensor in batch.items():
                result = self._pending.get(name)
                if result is None:
                    continue
                rows = (
                    torch.empty(0, dtype=torch.long)
                    if result.replace
                    else torch.tensor(result.rows, dtype=torch.long)
                )
                values = (
                    tensor.clone()
                    if result.replace
                    else tensor[0].index_select(0, rows).clone()
                    if rows.numel()
                    else tensor.new_empty((0, *tensor.shape[2:]))
                )
                output.append(
                    TensorPatch(
                        name, tuple(tensor.shape), rows, values, result.replace
                    )
                )
            patches = tuple(output)
        self._seeded = True
        return TensorDelta(batch, patches, full_reseed)


__all__ = [
    "PersistentTensorBank",
    "TensorBankStats",
    "TensorDelta",
    "TensorPatch",
    "EventTensorBank",
    "EVENT_TENSOR_KEYS",
]
