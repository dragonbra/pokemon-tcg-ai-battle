"""Import the approved 65-deck ZIP without changing FrozenMeta256-V1."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import zipfile

from .assets import canonical_deck_sha256, sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ZIP = REPOSITORY_ROOT / "FrozenPool_65_decks_2026-08-12.zip"
EXPECTED_ZIP_SHA256 = "2937c23a85c0e3805b40fb014f0274dbfc3f08e04ce3816c120105616ce320ee"
DISPLAY_NAME_OVERRIDES = {
    # Upstream manifest says Dusknoir, but this exact list contains the
    # Staryu/Starmie + Snorunt/Froslass lines and no Duskull-family card.
    "654f54a66c951ab2b6f4a714286552a614ecb94e84752b47bf03770c4cd7c933":
        "Mega Starmie ex / Mega Froslass ex",
}


def _write_json(path: Path, payload: object) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def import_pool() -> dict[str, object]:
    actual_zip_hash = sha256_file(SOURCE_ZIP)
    if actual_zip_hash != EXPECTED_ZIP_SHA256:
        raise RuntimeError(f"65-deck ZIP identity mismatch: {actual_zip_hash}")
    registry_path = PROJECT_ROOT / "assets/decks/registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    existing = {row["deck_id"]: row for row in registry["decks"]}
    if set(existing) not in (
        {f"{index:03d}" for index in range(1, 56)},
        {f"{index:03d}" for index in range(1, 66)},
    ):
        raise RuntimeError("65-deck migration requires immutable contiguous 001-055 or 001-065")
    imported: list[str] = []
    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        names = archive.namelist()
        for index in range(1, 66):
            deck_id = f"{index:03d}"
            prefix = f"decks\\{deck_id}_"
            deck_member = next(
                (name for name in names if name.startswith(prefix) and name.endswith("\\deck.csv")),
                None,
            )
            manifest_member = next(
                (name for name in names if name.startswith(prefix) and name.endswith("\\manifest.json")),
                None,
            )
            if deck_member is None or manifest_member is None:
                raise RuntimeError(f"ZIP is missing exact deck assets for {deck_id}")
            raw = archive.read(deck_member)
            cards = tuple(int(value) for value in raw.decode("utf-8").splitlines())
            content_hash = canonical_deck_sha256(cards)
            source_manifest = json.loads(archive.read(manifest_member))
            if index <= 55:
                row = existing[deck_id]
                if row["content_sha256"] != content_hash:
                    raise RuntimeError(f"ZIP rewrites immutable deck {deck_id}")
                continue
            destination = PROJECT_ROOT / f"assets/decks/definitions/{deck_id}/deck.csv"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.read_bytes() != raw:
                raise RuntimeError(f"refusing to overwrite imported deck {deck_id}")
            if not destination.exists():
                temporary = destination.with_suffix(".csv.tmp")
                temporary.write_bytes(raw)
                temporary.replace(destination)
            slug = deck_member.split("\\", 2)[1]
            source_name = slug.removeprefix(deck_id + "_").replace("_", " ").title()
            name = DISPLAY_NAME_OVERRIDES.get(content_hash, source_name)
            existing[deck_id] = {
                "deck_id": deck_id,
                "name": name,
                "archetype": name,
                "deck_path": f"assets/decks/definitions/{deck_id}/deck.csv",
                "file_sha256": sha256_file(destination),
                "content_sha256": content_hash,
                "card_count": 60,
                "cards": [
                    {"card_id": card_id, "count": count}
                    for card_id, count in sorted(Counter(cards).items())
                ],
                "roles": ["training"],
                "tags": ["bro_rogue_or_semantic_coverage", "pool_append_2026_08_12"],
                "source": {
                    "imported_from": SOURCE_ZIP.name,
                    "source_zip_sha256": actual_zip_hash,
                    "source_member": deck_member,
                    "source_manifest": source_manifest,
                    "source_display_name": source_name,
                    "display_name_correction": (
                        "exact-list correction: Starmie/Froslass lines; no Duskull-family card"
                        if name != source_name else None
                    ),
                    "imported_at": "2026-08-13T00:00:00+08:00",
                },
            }
            imported.append(deck_id)
    rows = [existing[f"{index:03d}"] for index in range(1, 66)]
    registry["decks"] = rows
    registry["identity_contract"] = (
        "deck_id is the append-only zero-padded numeric identity; 001-065 are immutable"
    )
    registry["training_pool_version"] = "FrozenPool_65_decks_2026-08-12"
    registry["training_pool_source_sha256"] = actual_zip_hash
    _write_json(registry_path, registry)
    return {"status": "PASS", "deck_count": 65, "imported": imported}


if __name__ == "__main__":
    print(json.dumps(import_pool(), sort_keys=True))
