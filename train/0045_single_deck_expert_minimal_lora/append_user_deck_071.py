"""Append the user-approved Hydrapple/Meganium list as deck 071."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from .assets import canonical_deck_sha256, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = PROJECT_ROOT / "assets/decks/registry.json"
TAXONOMY_PATH = PROJECT_ROOT / "assets/taxonomy/own_archetypes_v2.json"
MAPPING_PATH = PROJECT_ROOT / "assets/taxonomy/deck_own_archetype_mapping_v2.json"
DECK_ID = "071"
ARCHETYPE_ID = 27

# Shared card names use the same official-engine print identities as immutable
# Hydrapple/Meganium deck 023. The two Applin and Bayleef copies intentionally
# retain the one-plus-one print split from that exact list.
CARD_COUNTS = {
    1: 14,
    93: 2, 96: 4, 140: 1, 149: 1, 150: 2, 346: 1,
    709: 1, 710: 2, 917: 2, 918: 1, 920: 1, 1071: 2,
    1080: 1, 1094: 4, 1097: 1, 1121: 4, 1152: 2,
    1182: 2, 1184: 1, 1213: 1, 1227: 4, 1231: 2, 1261: 4,
}


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append() -> dict[str, object]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    existing = {row["deck_id"]: row for row in registry["decks"]}
    expected_existing = {f"{index:03d}" for index in range(1, 71)}
    if set(existing) not in (expected_existing, expected_existing | {DECK_ID}):
        raise RuntimeError(
            "deck 071 append requires immutable contiguous registry 001-070 or 001-071"
        )

    cards = tuple(
        card_id for card_id, count in CARD_COUNTS.items() for _ in range(count)
    )
    if len(cards) != 60:
        raise RuntimeError("deck 071 is not exact 60")
    content_hash = canonical_deck_sha256(cards)
    if any(
        row["content_sha256"] == content_hash and row["deck_id"] != DECK_ID
        for row in existing.values()
    ):
        raise RuntimeError("deck 071 duplicates an immutable prior exact list")

    destination = PROJECT_ROOT / "assets/decks/definitions/071/deck.csv"
    content = "\n".join(str(card_id) for card_id in sorted(cards)) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_text(encoding="ascii") != content:
        raise RuntimeError("refusing to overwrite immutable deck 071")
    if not destination.exists():
        temporary = destination.with_suffix(".csv.tmp")
        temporary.write_text(content, encoding="ascii")
        temporary.replace(destination)

    expected_row = {
        "deck_id": DECK_ID,
        "name": "Hydrapple ex / Meganium",
        "archetype": "Hydrapple ex / Meganium",
        "deck_path": "assets/decks/definitions/071/deck.csv",
        "file_sha256": sha256_file(destination),
        "content_sha256": content_hash,
        "card_count": 60,
        "cards": [
            {"card_id": card_id, "count": count}
            for card_id, count in sorted(Counter(cards).items())
        ],
        "roles": ["training"],
        "tags": [
            "user_approved_append_2026_08_16",
            "hydrapple_meganium",
            "forest_of_vitality",
            "training_pool_append_2026_08_16",
        ],
        "source": {
            "source_type": "user_provided_exact_list",
            "source_list": (
                "user-provided 2026-08-16: Hydrapple/Meganium exact 60"
            ),
            "print_identity_resolution": (
                "all shared names use immutable deck 023 official-engine print IDs"
            ),
            "exact_list_delta_from_023": {
                "added": ["Dawn x1", "Judge x1", "Poke Pad x1"],
                "removed": [
                    "Celebi x1",
                    "Ciphermaniac's Codebreaking x1",
                    "Briar x1",
                ],
            },
            "appended_at": "2026-08-16T00:00:00+08:00",
        },
    }
    if DECK_ID in existing and existing[DECK_ID] != expected_row:
        raise RuntimeError("registered deck 071 metadata differs from the approved append")
    existing[DECK_ID] = expected_row
    registry["decks"] = [existing[f"{index:03d}"] for index in range(1, 72)]
    registry["identity_contract"] = "deck_id is append-only; 001-071 are immutable"
    registry["training_pool_version"] = (
        "FrozenPool65_plus_user_decks_066_071_2026-08-16"
    )
    registry["training_pool_append_2026_08_16_deck_071"] = {
        "added_deck_ids": [DECK_ID],
        "benchmark_v2_historical_pool_changed": False,
        "formal_training_opponent_pool_changed": True,
    }
    _write_json(REGISTRY_PATH, registry)

    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    matching = [row for row in taxonomy["classes"] if row["archetype_id"] == ARCHETYPE_ID]
    if len(matching) != 1 or matching[0]["name"] != "hydrapple_meganium":
        raise RuntimeError("Own Archetype 27 is not the immutable Hydrapple/Meganium class")
    matching[0]["deck_ids"] = sorted(set(matching[0]["deck_ids"]) | {DECK_ID})
    _write_json(TAXONOMY_PATH, taxonomy)

    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    by_id = {row["deck_id"]: row for row in mapping["decks"]}
    expected_mapping = {
        "deck_id": DECK_ID,
        "old_archetype_id": 6,
        "archetype_id": ARCHETYPE_ID,
        "decision": "split",
        "strategic_axis": (
            "Hydrapple/Meganium energy-wide scaling with Judge and additional "
            "Dawn/Poke Pad consistency"
        ),
    }
    if DECK_ID in by_id and by_id[DECK_ID] != expected_mapping:
        raise RuntimeError("deck 071 taxonomy mapping differs from the approved append")
    by_id[DECK_ID] = expected_mapping
    mapping["mapping_provenance"] = (
        "manual 001-067 audit plus user-approved exact decks 068-071 through 2026-08-16"
    )
    mapping["decks"] = [by_id[f"{index:03d}"] for index in range(1, 72)]
    _write_json(MAPPING_PATH, mapping)

    return {
        "status": "PASS",
        "deck_count": 71,
        "deck_id": DECK_ID,
        "content_sha256": content_hash,
        "file_sha256": sha256_file(destination),
        "own_archetype_id": ARCHETYPE_ID,
    }


if __name__ == "__main__":
    print(json.dumps(append(), sort_keys=True))
