"""Priority-ordered 14-main-axis plus Other opponent classification."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONTRACT = Path(__file__).resolve().parents[1] / "assets" / "archetypes_v1.json"


@dataclass(frozen=True, slots=True)
class ArchetypeClass:
    class_id: int
    name: str
    trigger_card_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ArchetypeContract:
    schema_version: str
    classes: tuple[ArchetypeClass, ...]
    other_class_id: int

    @classmethod
    def load(cls, path: Path = DEFAULT_CONTRACT) -> "ArchetypeContract":
        raw = json.loads(path.read_text(encoding="utf-8"))
        classes = tuple(
            ArchetypeClass(
                int(item["class_id"]),
                str(item["name"]),
                tuple(int(value) for value in item["trigger_card_ids"]),
            )
            for item in raw["classes"]
        )
        ids = tuple(item.class_id for item in classes)
        if ids != tuple(range(15)) or int(raw["other_class_id"]) != 14:
            raise ValueError("0036 archetype contract must define ordered classes 0..14")
        if classes[-1].trigger_card_ids or classes[-1].name != "other":
            raise ValueError("0036 class 14 must be trigger-free Other")
        triggers = [value for item in classes[:-1] for value in item.trigger_card_ids]
        if len(triggers) != len(set(triggers)):
            raise ValueError("0036 archetype trigger IDs must be priority-unambiguous")
        return cls(str(raw["schema_version"]), classes, 14)

    def classify(self, deck: Sequence[int]) -> int:
        cards = set(deck)
        for item in self.classes:
            if item.class_id == self.other_class_id:
                continue
            if cards.intersection(item.trigger_card_ids):
                return item.class_id
        return self.other_class_id

    def name(self, class_id: int) -> str:
        return self.classes[class_id].name


__all__ = ["ArchetypeClass", "ArchetypeContract", "DEFAULT_CONTRACT"]
