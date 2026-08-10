"""0042 own-strategy identity, deliberately distinct from opponent Meta labels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


DEFAULT_TAXONOMY = Path(__file__).resolve().parent / "assets/own_archetypes_v1.json"
EXPECTED_TAXONOMY_SHA256 = "44865cc88612d91154adffe13f523a63103ab0c0d1a3d981af671b96ce4a8c36"
OWN_ARCHETYPE_VOCABULARY_VERSION = "0042_own_archetypes_v1"


@dataclass(frozen=True, slots=True)
class OwnArchetypeId:
    """Actor-owned strategic identity; never use as an opponent target type."""

    value: int

    def __post_init__(self) -> None:
        if not 0 <= self.value < 15:
            raise ValueError("OwnArchetypeId must be in [0, 15)")


@dataclass(frozen=True, slots=True)
class ArchetypeClass:
    class_id: int
    name: str
    trigger_card_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class OwnArchetypeVocabulary:
    classes: tuple[ArchetypeClass, ...]
    others_id: int
    source_schema_version: str
    sha256: str

    @classmethod
    def load(cls, path: Path = DEFAULT_TAXONOMY) -> "OwnArchetypeVocabulary":
        raw_bytes = path.read_bytes()
        digest = hashlib.sha256(raw_bytes).hexdigest()
        if path == DEFAULT_TAXONOMY and digest != EXPECTED_TAXONOMY_SHA256:
            raise ValueError(f"0042 own-archetype taxonomy SHA mismatch: {digest}")
        raw = json.loads(raw_bytes)
        classes = tuple(
            ArchetypeClass(
                int(item["class_id"]),
                str(item["name"]),
                tuple(int(value) for value in item["trigger_card_ids"]),
            )
            for item in raw["classes"]
        )
        if tuple(item.class_id for item in classes) != tuple(range(15)):
            raise ValueError("0042 own-archetype taxonomy must define classes 0..14")
        others_id = int(raw["other_class_id"])
        if others_id != 14 or classes[others_id].name != "other":
            raise ValueError("0042 own-archetype class 14 must be Other")
        triggers = [card for item in classes[:-1] for card in item.trigger_card_ids]
        if len(triggers) != len(set(triggers)):
            raise ValueError("0042 own-archetype triggers must be priority-unambiguous")
        return cls(classes, others_id, str(raw["schema_version"]), digest)

    def classify_own_deck(self, deck: Sequence[int]) -> OwnArchetypeId:
        cards = set(int(card) for card in deck)
        for item in self.classes[:-1]:
            if cards.intersection(item.trigger_card_ids):
                return OwnArchetypeId(item.class_id)
        return OwnArchetypeId(self.others_id)

    def classify_opponent_target(self, deck: Sequence[int]) -> int:
        """Training-only label under the pretrained q1 Meta taxonomy."""
        return self.classify_own_deck(deck).value


__all__ = [
    "ArchetypeClass",
    "DEFAULT_TAXONOMY",
    "EXPECTED_TAXONOMY_SHA256",
    "OWN_ARCHETYPE_VOCABULARY_VERSION",
    "OwnArchetypeId",
    "OwnArchetypeVocabulary",
]
