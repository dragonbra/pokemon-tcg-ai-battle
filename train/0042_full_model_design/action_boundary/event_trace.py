"""Complete engine-event records, deliberately separate from PolicyTrajectory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class EngineEvent:
    sequence: int
    observation: dict[str, Any]
    primitive_select: tuple[int, ...] | None
    gate: str
    forced: bool
    reward: float = 0.0
    terminal: bool = False
    winner: int | None = None
    chance_boundary: bool = False
    information_boundary: bool = False
    priority_transfer: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EngineEventTrace:
    battle_id: str
    events: list[EngineEvent] = field(default_factory=list)

    def append(self, **fields: Any) -> int:
        index = len(self.events)
        self.events.append(EngineEvent(sequence=index, **fields))
        return index

    def span_from(self, start: int) -> tuple[int, int]:
        if not 0 <= start <= len(self.events):
            raise ValueError("invalid event span start")
        return start, len(self.events)


__all__ = ["EngineEvent", "EngineEventTrace"]
