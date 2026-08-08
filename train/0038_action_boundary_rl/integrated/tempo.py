"""Observable tempo metrics; 0038 v1 intentionally defines no tempo reward."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TempoEvent:
    own_turn_index: int
    focal_first: bool
    opponent_archetype: str
    attack_legal: bool
    attacked: bool
    damage: int = 0
    prizes: int = 0
    interruption_reason: str | None = None


@dataclass(slots=True)
class TempoTracker:
    events: list[TempoEvent] = field(default_factory=list)

    @staticmethod
    def attack_opportunity(public_root_options: list[dict[str, Any]]) -> bool:
        """Use only the legal options in the current public observation."""
        return any(
            option.get("action") == "attack"
            or option.get("actionType") == "attack"
            or option.get("attackId") is not None
            for option in public_root_options
        )

    def add(self, event: TempoEvent) -> None:
        if event.attacked and not event.attack_legal:
            raise ValueError("an attack cannot be recorded without a legal public option")
        self.events.append(event)

    def summary(self) -> dict[str, float | int]:
        opportunities = [item for item in self.events if item.attack_legal]
        second = [item for item in self.events if item.own_turn_index == 2]
        streak = longest_attack_streak(self.events)
        return {
            "own_turns": len(self.events),
            "attack_opportunities": len(opportunities),
            "opportunity_conversion": (
                sum(item.attacked for item in opportunities) / len(opportunities)
                if opportunities else 0.0
            ),
            "second_turn_attacked": int(any(item.attacked for item in second)),
            "second_turn_attack_legal": int(any(item.attack_legal for item in second)),
            "longest_attack_streak": streak,
            "attack_damage": sum(item.damage for item in self.events if item.attacked),
            "attack_prizes": sum(item.prizes for item in self.events if item.attacked),
        }


def longest_attack_streak(events: list[TempoEvent]) -> int:
    current = best = 0
    for item in sorted(events, key=lambda row: row.own_turn_index):
        current = current + 1 if item.attacked else 0
        best = max(best, current)
    return best


__all__ = ["TempoEvent", "TempoTracker", "longest_attack_streak"]
