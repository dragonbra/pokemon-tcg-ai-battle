"""Append the user-approved Dragapult/Dusknoir/Munkidori list as deck 070."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from .assets import canonical_deck_sha256, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = PROJECT_ROOT / "assets/decks/registry.json"
TAXONOMY_PATH = PROJECT_ROOT / "assets/taxonomy/own_archetypes_v2.json"
MAPPING_PATH = PROJECT_ROOT / "assets/taxonomy/deck_own_archetype_mapping_v2.json"
DECK_ID = "070"
ARCHETYPE_ID = 15

# Display-set aliases resolve to the following immutable official-engine IDs:
# PRE 35 -> SFA 18, ASC 16 -> PRE 4, ASC 142 -> SFA 38,
# ASC 196 -> SFA 61, MEE basic Energy -> SVE basic Energy.
CARD_COUNTS = {
    2: 3, 5: 3, 7: 2,
    112: 2, 119: 4, 120: 4, 121: 3, 131: 2, 132: 1, 133: 1,
    140: 1, 235: 1, 1071: 1,
    1079: 3, 1080: 1, 1086: 4, 1097: 2, 1121: 4, 1152: 4,
    1182: 3, 1198: 3, 1227: 4, 1231: 2, 1260: 2,
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
    expected_existing = {f"{index:03d}" for index in range(1, 70)}
    if set(existing) not in (expected_existing, expected_existing | {DECK_ID}):
        raise RuntimeError("deck 070 append requires immutable contiguous registry 001-069 or 001-070")

    cards = tuple(
        card_id for card_id, count in CARD_COUNTS.items() for _ in range(count)
    )
    if len(cards) != 60:
        raise RuntimeError("deck 070 is not exact 60")
    content_hash = canonical_deck_sha256(cards)
    if any(
        row["content_sha256"] == content_hash and row["deck_id"] != DECK_ID
        for row in existing.values()
    ):
        raise RuntimeError("deck 070 duplicates an immutable prior exact list")

    destination = PROJECT_ROOT / "assets/decks/definitions/070/deck.csv"
    content = "\n".join(str(card_id) for card_id in sorted(cards)) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_text(encoding="ascii") != content:
        raise RuntimeError("refusing to overwrite immutable deck 070")
    if not destination.exists():
        temporary = destination.with_suffix(".csv.tmp")
        temporary.write_text(content, encoding="ascii")
        temporary.replace(destination)

    expected_row = {
        "deck_id": DECK_ID,
        "name": "Dragapult ex / Dusknoir / Munkidori",
        "archetype": "Dragapult ex / Dusknoir / Munkidori",
        "deck_path": "assets/decks/definitions/070/deck.csv",
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
            "dragapult_dusknoir",
            "munkidori",
            "training_pool_append_2026_08_16",
        ],
        "source": {
            "source_type": "user_provided_exact_list",
            "source_list": "user-provided 2026-08-16: Dragapult/Dusknoir/Munkidori exact 60",
            "set_code_resolution": {
                "PRE_35_duskull": "SFA 18 engine identity",
                "PRE_36_dusclops": "SFA 19 engine identity",
                "PRE_37_dusknoir": "SFA 20 engine identity",
                "ASC_16_budew": "PRE 4 engine identity",
                "ASC_142_fezandipiti_ex": "SFA 38 engine identity",
                "ASC_196_night_stretcher": "SFA 61 engine identity",
                "MEE_basic_energy": "SVE basic Energy engine identity",
                "MEG_114_boss": "PAL 172 engine identity",
                "MEG_125_rare_candy": "SVI 191 engine identity",
                "MEG_131_ultra_ball": "SVI 196 engine identity",
            },
            "appended_at": "2026-08-16T00:00:00+08:00",
        },
    }
    if DECK_ID in existing and existing[DECK_ID] != expected_row:
        raise RuntimeError("registered deck 070 metadata differs from the approved append")
    existing[DECK_ID] = expected_row
    registry["decks"] = [existing[f"{index:03d}"] for index in range(1, 71)]
    registry["identity_contract"] = "deck_id is append-only; 001-070 are immutable"
    registry["training_pool_version"] = "FrozenPool65_plus_user_decks_066_070_2026-08-16"
    registry["training_pool_append_2026_08_16"] = {
        "added_deck_ids": [DECK_ID],
        "benchmark_v2_historical_pool_changed": False,
        "formal_training_opponent_pool_changed": True,
    }
    _write_json(REGISTRY_PATH, registry)

    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    matching = [row for row in taxonomy["classes"] if row["archetype_id"] == ARCHETYPE_ID]
    if len(matching) != 1 or matching[0]["name"] != "dragapult_dusknoir":
        raise RuntimeError("Own Archetype 15 is not the immutable Dragapult/Dusknoir class")
    matching[0]["deck_ids"] = sorted(set(matching[0]["deck_ids"]) | {DECK_ID})
    _write_json(TAXONOMY_PATH, taxonomy)

    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    by_id = {row["deck_id"]: row for row in mapping["decks"]}
    expected_mapping = {
        "deck_id": DECK_ID,
        "old_archetype_id": ARCHETYPE_ID,
        "archetype_id": ARCHETYPE_ID,
        "decision": "keep",
        "strategic_axis": (
            "Dragapult spread plus Dusknoir self-KO breakpoints, Munkidori damage "
            "transfer, and Risky Ruins pressure"
        ),
    }
    if DECK_ID in by_id and by_id[DECK_ID] != expected_mapping:
        raise RuntimeError("deck 070 taxonomy mapping differs from the approved append")
    by_id[DECK_ID] = expected_mapping
    mapping["mapping_provenance"] = (
        "manual 001-067 audit plus user-approved exact decks 068-070 through 2026-08-16"
    )
    mapping["decks"] = [by_id[f"{index:03d}"] for index in range(1, 71)]
    _write_json(MAPPING_PATH, mapping)

    return {
        "status": "PASS",
        "deck_count": 70,
        "deck_id": DECK_ID,
        "content_sha256": content_hash,
        "file_sha256": sha256_file(destination),
        "own_archetype_id": ARCHETYPE_ID,
    }


if __name__ == "__main__":
    print(json.dumps(append(), sort_keys=True))
