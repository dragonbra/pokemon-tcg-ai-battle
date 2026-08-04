"""Deterministic current-state facts that do not require policy learning."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..domain.prototypes import FieldState, PrototypeIndex


@dataclass(frozen=True, slots=True)
class AttackFacts:
    base_damage: float
    base_damage_state: int
    required_energy_count: int
    attached_energy_count: int
    exact_energy_matches: int
    typed_energy_deficit: int
    total_energy_deficit: int


def integer(value: Any, default: int = 0) -> int:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else default


def card_id(value: Any) -> int:
    if isinstance(value, Mapping):
        return max(0, integer(value.get("id", value.get("cardId", 0))))
    return max(0, integer(value))


def field(record: Mapping[str, Any] | None, name: str) -> tuple[float, int]:
    if record is None or not isinstance(record.get(name), Mapping):
        return 0.0, int(FieldState.UNKNOWN)
    item = record[name]
    return float(item.get("value", 0) or 0), int(item.get("state", FieldState.UNKNOWN))


def attached_energy_types(entity: Mapping[str, Any] | None, prototypes: PrototypeIndex) -> list[int]:
    if entity is None:
        return []
    result: list[int] = []
    cards = entity.get("energyCards", entity.get("energies", ()))
    if not isinstance(cards, Sequence):
        return result
    for item in cards:
        prototype = prototypes.cards.get(card_id(item))
        value, state = field(prototype, "energy_type")
        if state == int(FieldState.PRESENT):
            result.append(int(value))
    return result


def energy_gap(required: Sequence[int], attached: Sequence[int]) -> tuple[int, int, int]:
    """Match typed costs first, then satisfy Colorless with remaining Energy."""
    available = Counter(int(value) for value in attached)
    exact = 0
    typed_deficit = 0
    colorless = 0
    for raw in required:
        energy_type = int(raw)
        if energy_type == 0:
            colorless += 1
        elif available[energy_type] > 0:
            available[energy_type] -= 1
            exact += 1
        elif available[10] > 0:
            available[10] -= 1
            exact += 1
        else:
            typed_deficit += 1
    colorless_matches = min(colorless, sum(available.values()))
    exact += colorless_matches
    return exact, typed_deficit, typed_deficit + colorless - colorless_matches


def attack_facts(
    attack_id: int,
    source: Mapping[str, Any] | None,
    prototypes: PrototypeIndex,
) -> AttackFacts:
    attack = prototypes.attacks.get(attack_id)
    damage, state = field(attack, "base_damage")
    required = list(attack.get("energy_types", ())) if attack else []
    attached = attached_energy_types(source, prototypes)
    exact, typed, total = energy_gap(required, attached)
    return AttackFacts(damage, state, len(required), len(attached), exact, typed, total)


__all__ = [
    "AttackFacts",
    "attack_facts",
    "attached_energy_types",
    "card_id",
    "energy_gap",
    "field",
    "integer",
]
