"""Versioned, append-only own-strategy taxonomy; independent from opponent Meta."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .assets import AssetIntegrityError, AssetRegistry, canonical_deck_sha256


ASSET_ROOT = Path(__file__).resolve().parent / "assets/taxonomy"
TAXONOMIES = {
    "0042_own_archetypes_v1": ASSET_ROOT / "own_archetypes_v1.json",
    "own_archetypes_v2": ASSET_ROOT / "own_archetypes_v2.json",
}
MAPPING_V2 = ASSET_ROOT / "deck_own_archetype_mapping_v2.json"


@dataclass(frozen=True, slots=True)
class OwnArchetypeId:
    value: int
    taxonomy_version: str
    class_count: int

    def __post_init__(self) -> None:
        if not 0 <= self.value < self.class_count:
            raise ValueError(f"OwnArchetypeId must be in [0, {self.class_count})")


@dataclass(frozen=True, slots=True)
class ArchetypeClass:
    archetype_id: int
    name: str
    display_name: str
    parent_archetype: str | None
    definition: str
    strategic_rationale: str
    deck_ids: tuple[str, ...]
    tags: tuple[str, ...]
    embedding_init_from: int
    introduced_in: str
    status: str
    trigger_card_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DeckMapping:
    deck_id: str
    content_sha256: str
    old_archetype_id: int
    archetype_id: int
    decision: str
    strategic_axis: str


@dataclass(frozen=True, slots=True)
class OwnArchetypeVocabulary:
    taxonomy_version: str
    schema_version: str
    taxonomy_sha256: str
    classes: tuple[ArchetypeClass, ...]
    other_class_id: int
    embedding_width: int
    mapping_sha256: str | None
    mappings: tuple[DeckMapping, ...]

    @property
    def class_count(self) -> int:
        return len(self.classes)

    @classmethod
    def load_version(
        cls, taxonomy_version: str, *, project_root: Path | None = None
    ) -> "OwnArchetypeVocabulary":
        try:
            path = TAXONOMIES[taxonomy_version]
        except KeyError as error:
            raise AssetIntegrityError(f"unknown own taxonomy: {taxonomy_version}") from error
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
        classes = tuple(
            ArchetypeClass(
                archetype_id=int(row["archetype_id"]),
                name=str(row["name"]),
                display_name=str(row.get("display_name", row["name"])),
                parent_archetype=row.get("parent_archetype"),
                definition=str(row.get("definition", "frozen V1 class")),
                strategic_rationale=str(row.get("strategic_rationale", "frozen V1 meaning")),
                deck_ids=tuple(row.get("deck_ids", ())),
                tags=tuple(row.get("tags", ())),
                embedding_init_from=int(row.get("embedding_init_from", row["archetype_id"])),
                introduced_in=str(row.get("introduced_in", "0042")),
                status=str(row.get("status", "active")),
                trigger_card_ids=tuple(int(value) for value in row.get("trigger_card_ids", ())),
            )
            for row in raw["classes"]
        )
        if tuple(row.archetype_id for row in classes) != tuple(range(len(classes))):
            raise AssetIntegrityError("own taxonomy IDs must be contiguous and append-only")
        other_id = int(raw["other_class_id"])
        if classes[other_id].name != "other":
            raise AssetIntegrityError("declared Other ID does not name the fallback class")
        mappings: tuple[DeckMapping, ...] = ()
        mapping_hash: str | None = None
        if taxonomy_version == "own_archetypes_v2":
            if project_root is None:
                project_root = Path(__file__).resolve().parent
            mapping_bytes = MAPPING_V2.read_bytes()
            mapping_raw = json.loads(mapping_bytes)
            registry = AssetRegistry.load(project_root)
            content_by_id = {deck.deck_id: deck.content_sha256 for deck in registry.decks}
            mappings = tuple(
                DeckMapping(
                    deck_id=row["deck_id"], content_sha256=content_by_id[row["deck_id"]],
                    old_archetype_id=int(row["old_archetype_id"]),
                    archetype_id=int(row["archetype_id"]), decision=str(row["decision"]),
                    strategic_axis=str(row["strategic_axis"]),
                )
                for row in mapping_raw["decks"]
            )
            if tuple(row.deck_id for row in mappings) != tuple(f"{i:03d}" for i in range(1, 68)):
                raise AssetIntegrityError("V2 must map each exact deck 001-067 exactly once")
            declared = {deck_id for row in classes for deck_id in row.deck_ids}
            if declared != {row.deck_id for row in mappings}:
                raise AssetIntegrityError("taxonomy deck_ids and exact mapping disagree")
            for row in mappings:
                if row.archetype_id >= len(classes):
                    raise AssetIntegrityError(f"mapping references unknown class: {row.deck_id}")
            mapping_hash = hashlib.sha256(mapping_bytes).hexdigest()
        return cls(
            taxonomy_version=taxonomy_version,
            schema_version=str(raw["schema_version"]),
            taxonomy_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            classes=classes,
            other_class_id=other_id,
            embedding_width=int(raw.get("embedding_width", 16)),
            mapping_sha256=mapping_hash,
            mappings=mappings,
        )

    def resolve_exact_deck(self, deck_id: str, cards: Sequence[int]) -> OwnArchetypeId:
        content = canonical_deck_sha256(int(card) for card in cards)
        row = next((item for item in self.mappings if item.deck_id == deck_id), None)
        if row is None or row.content_sha256 != content:
            raise AssetIntegrityError(f"exact known-deck identity mismatch: {deck_id}")
        return OwnArchetypeId(row.archetype_id, self.taxonomy_version, self.class_count)

    def classify_unknown_deck(self, cards: Sequence[int]) -> OwnArchetypeId:
        present = set(int(card) for card in cards)
        for row in self.classes:
            if row.archetype_id != self.other_class_id and present.intersection(row.trigger_card_ids):
                return OwnArchetypeId(row.archetype_id, self.taxonomy_version, self.class_count)
        return OwnArchetypeId(self.other_class_id, self.taxonomy_version, self.class_count)


__all__ = ["ArchetypeClass", "DeckMapping", "OwnArchetypeId", "OwnArchetypeVocabulary"]
