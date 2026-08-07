"""GPU-resident immutable event fields keyed by causal source-event identity."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from ..contracts.fields import WIDTHS
from .session_tensor_cache import SessionTensorKey


@dataclass(slots=True)
class _EventSession:
    key: SessionTensorKey
    slot: int
    positions: dict[int, int] = field(default_factory=dict)
    commitments: dict[int, tuple[Any, ...]] = field(default_factory=dict)


class EventStaticStore:
    """Cache immutable event payload tensors; transfer only ages and relations."""

    EVENT_KEYS = frozenset(
        {
            "event_cat", "event_num", "event_state", "event_mask",
            "event_source", "event_target", "event_before", "event_after",
        }
    )

    def __init__(
        self,
        device: torch.device | str,
        floating_dtype: torch.dtype,
        *,
        max_sessions: int = 512,
        max_events: int = 64,
    ) -> None:
        self.device = torch.device(device)
        if not floating_dtype.is_floating_point:
            raise ValueError("event cache floating dtype must be floating point")
        if max_sessions < 1 or max_events < 1:
            raise ValueError("event cache capacities must be positive")
        self.floating_dtype = floating_dtype
        self.max_sessions = max_sessions
        self.max_events = max_events
        self._sessions: dict[str, _EventSession] = {}
        self._free_slots: list[int] = []
        self._next_slot = 0
        self._counts: Counter[str] = Counter()
        self._cat = torch.zeros(
            (max_sessions, max_events, WIDTHS.event_cat),
            dtype=torch.long,
            device=self.device,
        )
        self._num = torch.zeros(
            (max_sessions, max_events, WIDTHS.event_num),
            dtype=floating_dtype,
            device=self.device,
        )
        self._state = torch.zeros(
            (max_sessions, max_events, WIDTHS.event_state),
            dtype=torch.long,
            device=self.device,
        )

    def _allocate_slot(self) -> int:
        if self._free_slots:
            return self._free_slots.pop()
        if self._next_slot >= self.max_sessions:
            raise RuntimeError("event cache session capacity exhausted")
        result = self._next_slot
        self._next_slot += 1
        return result

    def _session(self, key: SessionTensorKey) -> _EventSession:
        current = self._sessions.get(key.session_id)
        if current is not None:
            if current.key != key:
                raise ValueError("event cache session identity changed")
            return current
        current = _EventSession(key, self._allocate_slot())
        self._sessions[key.session_id] = current
        self._counts["session_starts"] += 1
        return current

    @staticmethod
    def _commitment(actor: Mapping[str, Any], index: int) -> tuple[Any, ...]:
        event_num = actor["event_num"][index]
        return (
            tuple(actor["event_cat"][index]),
            tuple(event_num[1:]),
            tuple(actor["event_state"][index]),
        )

    def update_batch(
        self,
        keys: Sequence[SessionTensorKey],
        records: Sequence[Mapping[str, Any]],
        source_events: Sequence[Sequence[int]],
    ) -> dict[str, Tensor]:
        if not keys or not (len(keys) == len(records) == len(source_events)):
            raise ValueError("event cache batch inputs must be nonempty and aligned")
        if len({key.session_id for key in keys}) != len(keys):
            raise ValueError("event cache batch contains duplicate sessions")
        batch_size = len(keys)
        maximum = max(1, max(len(values) for values in source_events))
        if maximum > self.max_events:
            raise ValueError("event cache observation exceeds event capacity")

        flat_positions = torch.zeros((batch_size, maximum), dtype=torch.long)
        mask = torch.zeros((batch_size, maximum), dtype=torch.bool)
        ages = torch.zeros((batch_size, maximum, WIDTHS.event_num), dtype=torch.float32)
        relations = {
            name: torch.zeros((batch_size, maximum), dtype=torch.long)
            for name in ("event_source", "event_target", "event_before", "event_after")
        }
        upload_indices: list[int] = []
        upload_cat: list[list[int]] = []
        upload_num: list[list[float]] = []
        upload_state: list[list[int]] = []

        for batch_index, (key, record, identities) in enumerate(
            zip(keys, records, source_events, strict=True)
        ):
            actor = record.get("actor")
            if not isinstance(actor, Mapping):
                raise ValueError("event cache record has no canonical actor")
            count = len(actor.get("event_cat", ()))
            if count != len(identities) or len(set(identities)) != len(identities):
                raise ValueError("event cache source identities do not match event rows")
            if any(type(identity) is not int or identity < 0 for identity in identities):
                raise ValueError("event cache source identities must be nonnegative integers")
            if not (
                len(actor["event_num"]) == count
                and len(actor["event_state"]) == count
                and all(len(actor[name]) == count for name in relations)
            ):
                raise ValueError("event cache canonical event lengths disagree")
            session = self._session(key)
            live = set(identities)
            for expired in tuple(session.positions):
                if expired not in live:
                    del session.positions[expired]
                    del session.commitments[expired]
            free = [
                position
                for position in range(self.max_events)
                if position not in session.positions.values()
            ]
            for event_index, identity in enumerate(identities):
                commitment = self._commitment(actor, event_index)
                position = session.positions.get(identity)
                if position is None:
                    if not free:
                        raise RuntimeError("event cache ring has no free position")
                    position = free.pop(0)
                    session.positions[identity] = position
                    session.commitments[identity] = commitment
                    upload_indices.append(session.slot * self.max_events + position)
                    upload_cat.append(list(actor["event_cat"][event_index]))
                    upload_num.append([0.0, *actor["event_num"][event_index][1:]])
                    upload_state.append(list(actor["event_state"][event_index]))
                    self._counts["static_misses"] += 1
                elif session.commitments[identity] != commitment:
                    raise ValueError("immutable event payload changed for one source identity")
                else:
                    self._counts["static_hits"] += 1
                flat_positions[batch_index, event_index] = (
                    session.slot * self.max_events + position
                )
                mask[batch_index, event_index] = True
                ages[batch_index, event_index, 0] = float(
                    actor["event_num"][event_index][0]
                )
                for name, output in relations.items():
                    output[batch_index, event_index] = int(actor[name][event_index])

        if upload_indices:
            indices = torch.tensor(upload_indices, dtype=torch.long, device=self.device)
            self._cat.view(-1, WIDTHS.event_cat).index_copy_(
                0, indices, torch.tensor(upload_cat, dtype=torch.long).to(self.device)
            )
            self._num.view(-1, WIDTHS.event_num).index_copy_(
                0,
                indices,
                torch.tensor(upload_num, dtype=self.floating_dtype).to(self.device),
            )
            self._state.view(-1, WIDTHS.event_state).index_copy_(
                0, indices, torch.tensor(upload_state, dtype=torch.long).to(self.device)
            )
            self._counts["static_upload_bytes"] += (
                len(upload_indices)
                * (
                    WIDTHS.event_cat * 8
                    + WIDTHS.event_num * torch.tensor([], dtype=self.floating_dtype).element_size()
                    + WIDTHS.event_state * 8
                    + 8
                )
            )

        positions = flat_positions.to(self.device)
        event_cat = self._cat.view(-1, WIDTHS.event_cat).index_select(
            0, positions.reshape(-1)
        ).reshape(batch_size, maximum, WIDTHS.event_cat)
        event_num = self._num.view(-1, WIDTHS.event_num).index_select(
            0, positions.reshape(-1)
        ).reshape(batch_size, maximum, WIDTHS.event_num)
        event_state = self._state.view(-1, WIDTHS.event_state).index_select(
            0, positions.reshape(-1)
        ).reshape(batch_size, maximum, WIDTHS.event_state)
        device_mask = mask.to(self.device)
        event_cat = event_cat.masked_fill(~device_mask.unsqueeze(-1), 0)
        event_num = event_num.masked_fill(~device_mask.unsqueeze(-1), 0)
        event_state = event_state.masked_fill(~device_mask.unsqueeze(-1), 0)
        event_num[..., 0] = ages[..., 0].to(
            self.device, dtype=self.floating_dtype
        )
        output = {
            "event_cat": event_cat,
            "event_num": event_num,
            "event_state": event_state,
            "event_mask": device_mask,
            **{name: value.to(self.device) for name, value in relations.items()},
        }
        self._counts["batches"] += 1
        self._counts["dynamic_upload_bytes"] += (
            flat_positions.numel() * 8
            + ages[..., 0].numel() * torch.tensor([], dtype=self.floating_dtype).element_size()
            + mask.numel()
            + sum(value.numel() * 8 for value in relations.values())
        )
        return output

    def close_session(self, key: SessionTensorKey) -> None:
        session = self._sessions.get(key.session_id)
        if session is None:
            return
        if session.key != key:
            raise ValueError("event cache close_session identity mismatch")
        self._free_slots.append(session.slot)
        del self._sessions[key.session_id]
        self._counts["closed_sessions"] += 1

    def reset_stats(self) -> None:
        self._counts.clear()

    def stats(self) -> dict[str, int | float]:
        output: dict[str, int | float] = dict(sorted(self._counts.items()))
        output["sessions"] = len(self._sessions)
        output["resident_bytes"] = sum(
            value.numel() * value.element_size()
            for value in (self._cat, self._num, self._state)
        )
        for name in (
            "batches", "static_hits", "static_misses", "static_upload_bytes",
            "dynamic_upload_bytes", "closed_sessions",
        ):
            output.setdefault(name, 0)
        return output


__all__ = ["EventStaticStore"]
