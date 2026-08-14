"""Training-only opponent Meta targets, independent from actor own-deck identity."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


DEFAULT_TAXONOMY = Path(__file__).resolve().parent / "assets/meta/opponent_archetypes_v1.json"
EXPECTED_TAXONOMY_SHA256 = "44865cc88612d91154adffe13f523a63103ab0c0d1a3d981af671b96ce4a8c36"
OPPONENT_ARCHETYPE_TARGET_VERSION = "0036_opponent_archetypes_v1"


@dataclass(frozen=True, slots=True)
class OpponentArchetypeTarget:
    """Training-only Meta target; never valid as an actor conditioning ID."""

    value: int

    def __post_init__(self) -> None:
        if not 0 <= self.value < 15:
            raise ValueError("OpponentArchetypeTarget must be in [0, 15)")


@dataclass(frozen=True, slots=True)
class OpponentArchetypeClass:
    class_id: int
    name: str
    trigger_card_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class OpponentArchetypeTaxonomy:
    classes: tuple[OpponentArchetypeClass, ...]
    others_id: int
    source_schema_version: str
    sha256: str

    @classmethod
    def load(cls, path: Path = DEFAULT_TAXONOMY) -> "OpponentArchetypeTaxonomy":
        raw_bytes = path.read_bytes()
        digest = hashlib.sha256(raw_bytes).hexdigest()
        if path == DEFAULT_TAXONOMY and digest != EXPECTED_TAXONOMY_SHA256:
            raise ValueError(f"0044 opponent Meta taxonomy SHA mismatch: {digest}")
        raw = json.loads(raw_bytes)
        classes = tuple(
            OpponentArchetypeClass(
                int(item["class_id"]),
                str(item["name"]),
                tuple(int(value) for value in item["trigger_card_ids"]),
            )
            for item in raw["classes"]
        )
        if tuple(item.class_id for item in classes) != tuple(range(15)):
            raise ValueError("0044 opponent Meta taxonomy must define classes 0..14")
        others_id = int(raw["other_class_id"])
        if others_id != 14 or classes[others_id].name != "other":
            raise ValueError("0044 opponent Meta class 14 must be Other")
        triggers = [card for item in classes[:-1] for card in item.trigger_card_ids]
        if len(triggers) != len(set(triggers)):
            raise ValueError("0044 opponent Meta triggers must be priority-unambiguous")
        return cls(classes, others_id, str(raw["schema_version"]), digest)

    def classify_target(self, deck: Sequence[int]) -> OpponentArchetypeTarget:
        cards = set(int(card) for card in deck)
        for item in self.classes[:-1]:
            if cards.intersection(item.trigger_card_ids):
                return OpponentArchetypeTarget(item.class_id)
        return OpponentArchetypeTarget(self.others_id)


__all__ = [
    "DEFAULT_TAXONOMY",
    "EXPECTED_TAXONOMY_SHA256",
    "OPPONENT_ARCHETYPE_TARGET_VERSION",
    "OpponentArchetypeClass",
    "OpponentArchetypeTarget",
    "OpponentArchetypeTaxonomy",
]
