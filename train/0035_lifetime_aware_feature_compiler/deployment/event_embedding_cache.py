"""GPU-resident pre-Transformer event lookup embeddings by source identity."""

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
class _Session:
    key: SessionTensorKey
    slot: int
    positions: dict[int, int] = field(default_factory=dict)


class EventEmbeddingStore:
    """Cache categorical/prototype event embeddings, not contextual tokens."""

    def __init__(
        self,
        model: Any,
        *,
        max_sessions: int = 512,
        max_events: int = 64,
    ) -> None:
        parameter = next(model.parameters())
        self.device = parameter.device
        self.dtype = parameter.dtype
        self.model = model
        self.state_encoder = model.state_encoder
        self.prototype_memory = model.prepare_prototype_cache()
        self.max_sessions = max_sessions
        self.max_events = max_events
        self.d_model = model.config.d_model
        self._categorical = torch.zeros(
            (max_sessions, max_events, self.d_model),
            device=self.device,
            dtype=self.dtype,
        )
        self._prototype = torch.zeros_like(self._categorical)
        self._sessions: dict[str, _Session] = {}
        self._free_slots: list[int] = []
        self._next_slot = 0
        self._counts: Counter[str] = Counter()

    def _allocate_slot(self) -> int:
        if self._free_slots:
            return self._free_slots.pop()
        if self._next_slot >= self.max_sessions:
            raise RuntimeError("event embedding cache session capacity exhausted")
        slot = self._next_slot
        self._next_slot += 1
        return slot

    def _session(self, key: SessionTensorKey) -> _Session:
        session = self._sessions.get(key.session_id)
        if session is not None:
            if session.key != key:
                raise ValueError("event embedding cache session identity changed")
            return session
        session = _Session(key, self._allocate_slot())
        self._sessions[key.session_id] = session
        self._counts["session_starts"] += 1
        return session

    def update_batch(
        self,
        keys: Sequence[SessionTensorKey],
        records: Sequence[Mapping[str, Any]],
        source_events: Sequence[Sequence[int]],
    ) -> tuple[dict[str, Tensor], tuple[Tensor, Tensor]]:
        if not keys or not (len(keys) == len(records) == len(source_events)):
            raise ValueError("event embedding cache inputs must be nonempty and aligned")
        if len({key.session_id for key in keys}) != len(keys):
            raise ValueError("event embedding cache batch contains duplicate sessions")
        batch_size = len(keys)
        maximum = max(1, max(len(values) for values in source_events))
        if maximum > self.max_events:
            raise ValueError("event embedding cache observation exceeds event capacity")

        positions = torch.zeros((batch_size, maximum), dtype=torch.long)
        mask = torch.zeros((batch_size, maximum), dtype=torch.bool)
        event_num = torch.zeros(
            (batch_size, maximum, WIDTHS.event_num), dtype=torch.float32
        )
        event_state = torch.zeros(
            (batch_size, maximum, WIDTHS.event_state), dtype=torch.long
        )
        relations = {
            name: torch.zeros((batch_size, maximum), dtype=torch.long)
            for name in ("event_source", "event_target", "event_before", "event_after")
        }
        upload_indices: list[int] = []
        upload_cat: list[list[int]] = []

        for batch_index, (key, record, identities) in enumerate(
            zip(keys, records, source_events, strict=True)
        ):
            actor = record.get("actor")
            if not isinstance(actor, Mapping):
                raise ValueError("event embedding cache record has no canonical actor")
            count = len(actor.get("event_cat", ()))
            if count != len(identities) or len(set(identities)) != len(identities):
                raise ValueError("source-event identities do not match event rows")
            if any(type(identity) is not int or identity < 0 for identity in identities):
                raise ValueError("source-event identities must be nonnegative integers")
            if not (
                len(actor["event_num"]) == count
                and len(actor["event_state"]) == count
                and all(len(actor[name]) == count for name in relations)
            ):
                raise ValueError("canonical event lengths disagree")
            session = self._session(key)
            live = set(identities)
            for expired in tuple(session.positions):
                if expired not in live:
                    del session.positions[expired]
            free = [
                position
                for position in range(self.max_events)
                if position not in session.positions.values()
            ]
            if count:
                mask[batch_index, :count] = True
                event_num[batch_index, :count] = torch.tensor(
                    actor["event_num"], dtype=torch.float32
                )
                event_state[batch_index, :count] = torch.tensor(
                    actor["event_state"], dtype=torch.long
                )
                for name, output in relations.items():
                    output[batch_index, :count] = torch.tensor(
                        actor[name], dtype=torch.long
                    )
            for event_index, identity in enumerate(identities):
                position = session.positions.get(identity)
                if position is None:
                    if not free:
                        raise RuntimeError("event embedding cache ring has no free position")
                    position = free.pop(0)
                    session.positions[identity] = position
                    upload_indices.append(session.slot * self.max_events + position)
                    upload_cat.append(list(actor["event_cat"][event_index]))
                    self._counts["static_misses"] += 1
                else:
                    self._counts["static_hits"] += 1
                positions[batch_index, event_index] = (
                    session.slot * self.max_events + position
                )

        if upload_indices:
            indices = torch.tensor(upload_indices, dtype=torch.long, device=self.device)
            categorical = torch.tensor(upload_cat, dtype=torch.long, device=self.device)
            with torch.inference_mode():
                categorical_embedding = self.state_encoder.event_cat(categorical)
                prototype_embedding = self.prototype_memory.card(categorical[:, 2])
            self._categorical.view(-1, self.d_model).index_copy_(
                0, indices, categorical_embedding
            )
            self._prototype.view(-1, self.d_model).index_copy_(
                0, indices, prototype_embedding
            )
            self._counts["static_upload_bytes"] += (
                len(upload_indices) * (WIDTHS.event_cat * 8 + 8)
            )

        device_positions = positions.to(self.device)
        flat = device_positions.reshape(-1)
        categorical_batch = self._categorical.view(-1, self.d_model).index_select(
            0, flat
        ).reshape(batch_size, maximum, self.d_model)
        prototype_batch = self._prototype.view(-1, self.d_model).index_select(
            0, flat
        ).reshape(batch_size, maximum, self.d_model)
        device_mask = mask.to(self.device)
        categorical_batch = categorical_batch.masked_fill(
            ~device_mask.unsqueeze(-1), 0
        )
        prototype_batch = prototype_batch.masked_fill(~device_mask.unsqueeze(-1), 0)
        output = {
            "event_cat": torch.zeros(
                (batch_size, maximum, WIDTHS.event_cat),
                dtype=torch.long,
                device=self.device,
            ),
            "event_num": event_num.to(self.device, dtype=self.dtype),
            "event_state": event_state.to(self.device),
            "event_mask": device_mask,
            **{name: value.to(self.device) for name, value in relations.items()},
        }
        self._counts["batches"] += 1
        self._counts["dynamic_upload_bytes"] += (
            event_num.numel() * event_num.element_size()
            + event_state.numel() * event_state.element_size()
            + mask.numel()
            + positions.numel() * positions.element_size()
            + sum(value.numel() * value.element_size() for value in relations.values())
        )
        return output, (categorical_batch, prototype_batch)

    def close_session(self, key: SessionTensorKey) -> None:
        session = self._sessions.get(key.session_id)
        if session is None:
            return
        if session.key != key:
            raise ValueError("event embedding cache close identity mismatch")
        self._free_slots.append(session.slot)
        del self._sessions[key.session_id]
        self._counts["closed_sessions"] += 1

    def reset_stats(self) -> None:
        self._counts.clear()

    def stats(self) -> dict[str, int | float]:
        output: dict[str, int | float] = dict(sorted(self._counts.items()))
        output["sessions"] = len(self._sessions)
        output["resident_bytes"] = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self._categorical, self._prototype)
        )
        for name in (
            "batches", "static_hits", "static_misses", "static_upload_bytes",
            "dynamic_upload_bytes", "closed_sessions",
        ):
            output.setdefault(name, 0)
        return output


__all__ = ["EventEmbeddingStore"]
