"""Device-resident canonical tensors keyed by chronological battle session."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ..contracts.batch import DecisionBatch
from ..contracts.fields import EXPECTED_BATCH_KEYS
from ..features.collate import BucketPadding
from ..features.tensor_bank import TensorDelta, TensorPatch


@dataclass(frozen=True, slots=True)
class SessionTensorKey:
    session_id: str
    actor: int
    deck_sha256: str

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session ID must be non-empty")
        if self.actor not in (0, 1):
            raise ValueError("session actor must be 0 or 1")
        if not self.deck_sha256:
            raise ValueError("session deck identity must be non-empty")


@dataclass(slots=True)
class _ResidentSession:
    key: SessionTensorKey
    slot: int
    logical_shapes: dict[str, tuple[int, ...]]


@dataclass(slots=True)
class _TensorSlab:
    tensor: Tensor
    ragged_capacity: int | None


class GpuSessionTensorStore:
    """Apply CPU row patches once, then batch resident device tensors exactly."""

    def __init__(
        self,
        device: torch.device | str,
        floating_dtype: torch.dtype,
        *,
        max_sessions: int = 512,
        bucket_padding: BucketPadding | None = None,
        max_resource_rows: int = 64,
        expected_keys: frozenset[str] = EXPECTED_BATCH_KEYS,
    ) -> None:
        self.device = torch.device(device)
        if not floating_dtype.is_floating_point:
            raise ValueError("runtime floating dtype must be floating point")
        self.floating_dtype = floating_dtype
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        if max_resource_rows < 1:
            raise ValueError("max_resource_rows must be positive")
        self.max_sessions = max_sessions
        self.bucket_padding = bucket_padding or BucketPadding()
        self.max_resource_rows = max_resource_rows
        if not expected_keys or not expected_keys <= EXPECTED_BATCH_KEYS:
            raise ValueError("resident tensor expected_keys must be a canonical subset")
        self.expected_keys = frozenset(expected_keys)
        self._sessions: dict[str, _ResidentSession] = {}
        self._slabs: dict[str, _TensorSlab] = {}
        self._free_slots: list[int] = []
        self._next_slot = 0
        self._counts: Counter[str] = Counter()

    def _runtime_dtype(self, value: Tensor) -> torch.dtype:
        return self.floating_dtype if value.dtype.is_floating_point else value.dtype

    @staticmethod
    def _validate_patch(patch: TensorPatch) -> None:
        if patch.name not in EXPECTED_BATCH_KEYS:
            raise ValueError(f"unknown tensor patch: {patch.name}")
        shape = patch.logical_shape
        if not shape or shape[0] != 1 or any(
            type(length) is not int or length < 1 for length in shape
        ):
            raise ValueError(f"invalid tensor patch shape: {patch.name}={shape}")
        if patch.rows.dtype is not torch.long or patch.rows.ndim != 1:
            raise ValueError(f"tensor patch rows must be one-dimensional long: {patch.name}")
        if patch.replace:
            if patch.rows.numel() or tuple(patch.values.shape) != shape:
                raise ValueError(f"replacement tensor patch is malformed: {patch.name}")
            return
        if len(shape) < 2:
            raise ValueError(f"sparse tensor patch requires a row dimension: {patch.name}")
        rows = patch.rows.tolist()
        if len(rows) != len(set(rows)) or any(
            type(row) is not int or not 0 <= row < shape[1] for row in rows
        ):
            raise ValueError(f"tensor patch row outside logical shape: {patch.name}")
        expected = (len(rows), *shape[2:])
        if tuple(patch.values.shape) != expected:
            raise ValueError(
                f"tensor patch values mismatch: {patch.name}; "
                f"expected={expected}, actual={tuple(patch.values.shape)}"
            )

    def _validate_delta(self, delta: TensorDelta) -> None:
        if set(delta.tensors) != self.expected_keys:
            raise ValueError("tensor delta keys do not match the resident authority")
        names = [patch.name for patch in delta.patches]
        if len(names) != len(set(names)):
            raise ValueError("tensor delta contains duplicate patch names")
        for patch in delta.patches:
            self._validate_patch(patch)
            authority = delta.tensors[patch.name]
            if tuple(authority.shape) != patch.logical_shape:
                raise ValueError(f"tensor patch authority shape drift: {patch.name}")
            if patch.values.dtype != authority.dtype:
                raise ValueError(f"tensor patch dtype drift: {patch.name}")
        if delta.full_reseed and set(names) != self.expected_keys:
            raise ValueError("full tensor reseed must replace every resident key")
        if delta.full_reseed and not all(patch.replace for patch in delta.patches):
            raise ValueError("full tensor reseed must contain replacements only")

    @staticmethod
    def _payload_bytes(patch: TensorPatch) -> int:
        return (
            patch.values.numel() * patch.values.element_size()
            + patch.rows.numel() * patch.rows.element_size()
        )

    def _transfer(self, value: Tensor) -> Tensor:
        return value.to(
            device=self.device,
            dtype=self._runtime_dtype(value),
            non_blocking=self.device.type == "cuda",
        )

    def _ragged_capacity(self, name: str, shape: tuple[int, ...]) -> int | None:
        if len(shape) < 2:
            return None
        if name.startswith("card_"):
            return self.bucket_padding.card[-1]
        if name.startswith("resource_"):
            return self.max_resource_rows
        if name.startswith("event_"):
            return self.bucket_padding.event[-1]
        if name.startswith("option_effect_"):
            return self.bucket_padding.effect[-1]
        if name.startswith("option_skill_"):
            return self.bucket_padding.skill[-1]
        if name.startswith("option_"):
            return self.bucket_padding.option[-1]
        if name == "targets":
            return self.bucket_padding.action[-1]
        return None

    def _slab(self, patch: TensorPatch) -> _TensorSlab:
        shape = patch.logical_shape
        capacity = self._ragged_capacity(patch.name, shape)
        if capacity is not None and shape[1] > capacity:
            raise ValueError(
                f"resident tensor exceeds {patch.name} capacity: {shape[1]} > {capacity}"
            )
        dtype = self._runtime_dtype(patch.values)
        expected_shape = (
            (self.max_sessions, capacity, *shape[2:])
            if capacity is not None
            else (self.max_sessions, *shape[1:])
        )
        slab = self._slabs.get(patch.name)
        if slab is None:
            tensor = torch.full(
                expected_shape,
                -100 if patch.name == "targets" else 0,
                dtype=dtype,
                device=self.device,
            )
            slab = _TensorSlab(tensor, capacity)
            self._slabs[patch.name] = slab
            self._counts["slab_allocations"] += 1
        elif (
            slab.ragged_capacity != capacity
            or slab.tensor.dtype != dtype
            or tuple(slab.tensor.shape) != expected_shape
        ):
            raise ValueError(f"resident tensor slab contract changed: {patch.name}")
        return slab

    def _allocate_slot(self) -> int:
        if self._free_slots:
            return self._free_slots.pop()
        if self._next_slot >= self.max_sessions:
            raise RuntimeError("resident tensor session capacity exhausted")
        slot = self._next_slot
        self._next_slot += 1
        return slot

    def _apply_patch(
        self, session: _ResidentSession, patch: TensorPatch
    ) -> None:
        slab = self._slab(patch)
        shape = patch.logical_shape
        previous_shape = session.logical_shapes.get(patch.name)
        if not patch.replace and previous_shape is None:
            raise ValueError(f"tensor patch requires a reseed: {patch.name}")
        if (
            not patch.replace
            and previous_shape != shape
            and tuple(patch.rows.tolist()) != tuple(range(shape[1]))
        ):
            raise ValueError(
                f"shape-changing tensor patch must replace all logical rows: {patch.name}"
            )
        destination = slab.tensor[session.slot]
        fill = -100 if patch.name == "targets" else 0
        if patch.replace or previous_shape != shape:
            destination.fill_(fill)
        if patch.replace:
            values = self._transfer(patch.values)
            if slab.ragged_capacity is None:
                destination.copy_(values[0])
            else:
                destination[: shape[1]].copy_(values[0])
        elif patch.rows.numel():
            rows = patch.rows.to(self.device, non_blocking=self.device.type == "cuda")
            values = self._transfer(patch.values)
            destination.index_copy_(0, rows, values)
        session.logical_shapes[patch.name] = shape

    def apply(self, key: SessionTensorKey, delta: TensorDelta) -> None:
        self.apply_many(((key, delta),))

    def apply_many(
        self,
        updates: Sequence[tuple[SessionTensorKey, TensorDelta]],
    ) -> None:
        """Aggregate every batch's patches into one transfer/scatter per key."""
        if not updates:
            return
        if len({key.session_id for key, _delta in updates}) != len(updates):
            raise ValueError("resident update batch contains duplicate sessions")

        prepared: list[tuple[_ResidentSession, TensorDelta]] = []
        for key, delta in updates:
            self._validate_delta(delta)
            existing = self._sessions.get(key.session_id)
            if existing is not None and existing.key != key:
                raise ValueError("session identity changed without close_session")
            if existing is None and not delta.full_reseed:
                raise ValueError("new session requires a full tensor reseed")
            if existing is None:
                session = _ResidentSession(key, self._allocate_slot(), {})
            elif delta.full_reseed:
                session = _ResidentSession(key, existing.slot, {})
            else:
                session = existing
            prepared.append((session, delta))

        grouped: dict[str, list[tuple[_ResidentSession, TensorPatch]]] = {}
        for session, delta in prepared:
            for patch in delta.patches:
                grouped.setdefault(patch.name, []).append((session, patch))

        actual_bytes_by_session: Counter[str] = Counter()
        for name, items in grouped.items():
            slab = self._slab(items[0][1])
            reset_slots: list[int] = []
            flat_indices: list[int] = []
            values: list[Tensor] = []
            for session, patch in items:
                if self._slab(patch) is not slab:
                    raise ValueError(f"resident tensor slab changed inside batch: {name}")
                shape = patch.logical_shape
                previous_shape = session.logical_shapes.get(name)
                if not patch.replace and previous_shape is None:
                    raise ValueError(f"tensor patch requires a reseed: {name}")
                if (
                    not patch.replace
                    and previous_shape != shape
                    and tuple(patch.rows.tolist()) != tuple(range(shape[1]))
                ):
                    raise ValueError(
                        f"shape-changing tensor patch must replace all logical rows: {name}"
                    )
                if patch.replace or previous_shape != shape:
                    reset_slots.append(session.slot)
                if slab.ragged_capacity is None:
                    if not patch.replace:
                        raise ValueError(f"fixed tensor patch must replace: {name}")
                    flat_indices.append(session.slot)
                    values.append(patch.values[0].unsqueeze(0))
                    row_count = 1
                else:
                    rows = (
                        list(range(shape[1]))
                        if patch.replace
                        else patch.rows.tolist()
                    )
                    flat_indices.extend(
                        session.slot * slab.ragged_capacity + row for row in rows
                    )
                    values.append(
                        patch.values[0] if patch.replace else patch.values
                    )
                    row_count = len(rows)
                actual_bytes_by_session[session.key.session_id] += (
                    patch.values.numel() * patch.values.element_size()
                    + row_count * torch.tensor([], dtype=torch.long).element_size()
                )
                session.logical_shapes[name] = shape

            if reset_slots:
                slots = torch.tensor(
                    sorted(set(reset_slots)), dtype=torch.long, device=self.device
                )
                slab.tensor.index_fill_(
                    0, slots, -100 if name == "targets" else 0
                )
            if flat_indices:
                host_values = torch.cat(values, dim=0)
                device_values = self._transfer(host_values)
                device_indices = torch.tensor(
                    flat_indices, dtype=torch.long, device=self.device
                )
                if slab.ragged_capacity is None:
                    destination = slab.tensor
                else:
                    destination = slab.tensor.reshape(
                        self.max_sessions * slab.ragged_capacity,
                        *slab.tensor.shape[2:],
                    )
                destination.index_copy_(0, device_indices, device_values)
            self._counts["aggregated_tensor_updates"] += 1

        for session, delta in prepared:
            if set(session.logical_shapes) != self.expected_keys:
                raise ValueError("resident session is missing canonical tensors")
            self._sessions[session.key.session_id] = session
            self._counts["applies"] += 1
            self._counts[
                "reseeds" if delta.full_reseed else "delta_updates"
            ] += 1
            self._counts[
                "initial_h2d_bytes" if delta.full_reseed else "delta_h2d_bytes"
            ] += actual_bytes_by_session[session.key.session_id]
            self._counts["patches"] += len(delta.patches)

    def batch(
        self,
        keys: Sequence[SessionTensorKey],
        *,
        local_stop_targets: bool = False,
    ) -> DecisionBatch:
        if not keys:
            raise ValueError("cannot batch zero resident sessions")
        if len({key.session_id for key in keys}) != len(keys):
            raise ValueError("resident batch contains duplicate sessions")
        sessions: list[_ResidentSession] = []
        for key in keys:
            state = self._sessions.get(key.session_id)
            if state is None:
                raise KeyError(f"resident session is absent: {key.session_id}")
            if state.key != key:
                raise ValueError("resident batch session identity mismatch")
            sessions.append(state)

        slot_indices = torch.tensor(
            [session.slot for session in sessions],
            dtype=torch.long,
            device=self.device,
        )
        batch: dict[str, Tensor] = {}
        for name, shape in sessions[0].logical_shapes.items():
            slab = self._slabs[name]
            output = slab.tensor.index_select(0, slot_indices)
            if slab.ragged_capacity is not None:
                maximum = max(session.logical_shapes[name][1] for session in sessions)
                output = output[:, :maximum]
            batch[name] = output

        # The STOP target is represented by the *batched* option width.  Each
        # resident singleton carries its local option count, so rebase only the
        # final non-padding target after the batch option width is known.
        if "targets" in batch:
            target_mask = batch["targets"].ne(-100)
            stop_columns = target_mask.sum(dim=1).sub(1).clamp_min(0)
            stop_values = (
                [session.logical_shapes["option_mask"][1] for session in sessions]
                if local_stop_targets
                else [batch["option_mask"].shape[1]] * len(keys)
            )
            batch["targets"].scatter_(
                1,
                stop_columns.unsqueeze(1),
                batch["targets"].new_tensor(stop_values).unsqueeze(1),
            )

        result = (
            DecisionBatch.from_mapping(batch)
            if self.expected_keys == EXPECTED_BATCH_KEYS
            else batch
        )
        self._counts["batches"] += 1
        self._counts["batched_sessions"] += len(keys)
        self._counts["d2d_batch_bytes"] += sum(
            value.numel() * value.element_size() for value in batch.values()
        )
        return result

    def close_session(self, key: SessionTensorKey) -> None:
        existing = self._sessions.get(key.session_id)
        if existing is None:
            return
        if existing.key != key:
            raise ValueError("close_session identity mismatch")
        self._free_slots.append(existing.slot)
        del self._sessions[key.session_id]
        self._counts["closed_sessions"] += 1

    def clear(self, reason: str) -> None:
        if not reason:
            raise ValueError("resident tensor cache clear requires a reason")
        self._counts["clears"] += 1
        self._counts[f"clear/{reason}"] += 1
        self._sessions.clear()
        self._slabs.clear()
        self._free_slots.clear()
        self._next_slot = 0

    def reset_stats(self) -> None:
        """Reset measurements without invalidating resident session contents."""
        self._counts.clear()

    def stats(self) -> dict[str, int | float]:
        result: dict[str, int | float] = dict(sorted(self._counts.items()))
        result["sessions"] = len(self._sessions)
        result["resident_bytes"] = sum(
            slab.tensor.numel() * slab.tensor.element_size()
            for slab in self._slabs.values()
        )
        for name in (
            "applies", "reseeds", "delta_updates", "initial_h2d_bytes",
            "delta_h2d_bytes", "patches", "batches", "batched_sessions",
            "d2d_batch_bytes", "closed_sessions", "clears", "slab_allocations",
            "aggregated_tensor_updates",
        ):
            result.setdefault(name, 0)
        return result


__all__ = ["GpuSessionTensorStore", "SessionTensorKey"]
