"""Append the two user-approved Dragapult exact lists as immutable decks 066/067."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from .assets import canonical_deck_sha256, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = PROJECT_ROOT / "assets/decks/registry.json"
TAXONOMY_PATH = PROJECT_ROOT / "assets/taxonomy/own_archetypes_v2.json"
MAPPING_PATH = PROJECT_ROOT / "assets/taxonomy/deck_own_archetype_mapping_v2.json"

# Card IDs are resolved against data/official/EN_Card_Data.csv. User-facing set
# aliases PRE/ASC/MEE and later print codes resolve to the same engine Card IDs
# SFA/SVE/SVI/PAL used by the current official runtime data.
DECKS = {
    "066": {
        "name": "Dragapult ex / Dusknoir Control",
        "archetype_id": 15,
        "strategic_axis": "Dusknoir self-KO, spread breakpoints, and Hammer/Watchtower control",
        "cards": {
            119: 4, 120: 4, 121: 3, 131: 2, 132: 2, 133: 1,
            140: 1, 112: 1, 235: 1, 1071: 1, 791: 1,
            1227: 4, 1182: 3, 1198: 3, 1231: 1, 1086: 4,
            1152: 4, 1121: 4, 1120: 3, 1097: 2, 1080: 1,
            1256: 2, 2: 4, 5: 3, 7: 1,
        },
        "source_list": "user-provided 2026-08-13: Dragapult/Dusknoir control exact 60",
    },
    "067": {
        "name": "Dragapult ex / Munkidori Control",
        "archetype_id": 0,
        "strategic_axis": "Phantom Dive spread with Munkidori and Hammer/Stadium control",
        "cards": {
            119: 4, 120: 4, 121: 3, 112: 2, 235: 2, 1071: 1,
            140: 1, 791: 1, 1227: 4, 1182: 3, 1198: 3,
            1213: 1, 1120: 4, 1086: 4, 1152: 4, 1121: 4,
            1097: 2, 1080: 1, 1256: 2, 1260: 1, 2: 4, 5: 3, 7: 2,
        },
        "source_list": "user-provided 2026-08-13: Dragapult/Munkidori control exact 60",
    },
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
    if set(existing) not in (
        {f"{index:03d}" for index in range(1, 66)},
        {f"{index:03d}" for index in range(1, 68)},
    ):
        raise RuntimeError("user Dragapult append requires immutable contiguous 001-065 or 001-067")
    existing_hashes = {row["content_sha256"] for row in existing.values() if row["deck_id"] < "066"}
    appended = []
    for deck_id, spec in DECKS.items():
        cards = tuple(card_id for card_id, count in spec["cards"].items() for _ in range(count))
        if len(cards) != 60:
            raise RuntimeError(f"deck {deck_id} is not exact 60")
        content_hash = canonical_deck_sha256(cards)
        if content_hash in existing_hashes:
            raise RuntimeError(f"deck {deck_id} duplicates an immutable prior exact list")
        destination = PROJECT_ROOT / f"assets/decks/definitions/{deck_id}/deck.csv"
        content = "\n".join(str(card_id) for card_id in sorted(cards)) + "\n"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_text(encoding="ascii") != content:
            raise RuntimeError(f"refusing to overwrite immutable deck {deck_id}")
        if not destination.exists():
            temporary = destination.with_suffix(".csv.tmp")
            temporary.write_text(content, encoding="ascii")
            temporary.replace(destination)
        existing[deck_id] = {
            "deck_id": deck_id,
            "name": spec["name"],
            "archetype": spec["name"],
            "deck_path": f"assets/decks/definitions/{deck_id}/deck.csv",
            "file_sha256": sha256_file(destination),
            "content_sha256": content_hash,
            "card_count": 60,
            "cards": [
                {"card_id": card_id, "count": count}
                for card_id, count in sorted(Counter(cards).items())
            ],
            "roles": ["training"],
            "tags": ["user_approved_append_2026_08_13", "dragapult_control"],
            "source": {
                "source_type": "user_provided_exact_list",
                "source_list": spec["source_list"],
                "set_code_resolution": {
                    "ASC": "SFA engine identity", "PRE": "SFA engine identity",
                    "MEE": "SVE engine identity", "POR_71_crushing_hammer": "SVI 168 engine identity",
                    "POR_76_judge": "SVI 176 engine identity", "MEG_114_boss": "PAL 172 engine identity",
                    "MEG_131_ultra_ball": "SVI 196 engine identity",
                },
                "appended_at": "2026-08-13T00:00:00+08:00",
            },
        }
        appended.append({"deck_id": deck_id, "content_sha256": content_hash})
    registry["decks"] = [existing[f"{index:03d}"] for index in range(1, 68)]
    registry["identity_contract"] = "deck_id is append-only; 001-067 are immutable"
    registry["training_pool_version"] = "FrozenPool65_plus_user_dragapult_066_067_2026-08-13"
    registry["training_pool_append"] = {
        "base": "FrozenPool_65_decks_2026-08-12", "added_deck_ids": ["066", "067"],
        "frozen_evaluation_changed": False,
    }
    _write_json(REGISTRY_PATH, registry)

    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    for row in taxonomy["classes"]:
        if row["archetype_id"] == 0:
            row["deck_ids"] = sorted(set(row["deck_ids"]) | {"067"})
        if row["archetype_id"] == 15:
            row["deck_ids"] = sorted(set(row["deck_ids"]) | {"066"})
    _write_json(TAXONOMY_PATH, taxonomy)

    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    mapping["mapping_provenance"] = "manual 001-067 exact-list strategic audit against official card data on 2026-08-13"
    by_id = {row["deck_id"]: row for row in mapping["decks"]}
    for deck_id, spec in DECKS.items():
        by_id[deck_id] = {
            "deck_id": deck_id, "old_archetype_id": spec["archetype_id"],
            "archetype_id": spec["archetype_id"], "decision": "keep",
            "strategic_axis": spec["strategic_axis"],
        }
    mapping["decks"] = [by_id[f"{index:03d}"] for index in range(1, 68)]
    _write_json(MAPPING_PATH, mapping)
    return {"status": "PASS", "deck_count": 67, "appended": appended}


if __name__ == "__main__":
    print(json.dumps(append(), sort_keys=True))
