from __future__ import annotations

import csv
from pathlib import Path


SET_IMAGE_IDS = {
    "ASC": "me2pt5",
    "BLK": "zsv10pt5",
    "DRI": "sv10",
    "JTG": "sv9",
    "MEG": "me1",
    "PAL": "sv2",
    "PFL": "me2",
    "POR": "me3",
    "PRE": "sv8pt5",
    "SCR": "sv7",
    "SFA": "sv6pt5",
    "SSP": "sv8",
    "SVE": "sve",
    "SVI": "sv1",
    "SVP": "svp",
    "TEF": "sv5",
    "TWM": "sv6",
    "WHT": "rsv10pt5",
}
SCRYDEX_SET_IDS = {"me2pt5", "me3"}


def load_card_catalog(path: Path) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            try:
                card_id = int(row.get("Card ID", ""))
            except (TypeError, ValueError):
                continue
            result.setdefault(
                card_id,
                {
                    "name": str(row.get("Card Name", "")),
                    "expansion": str(row.get("Expansion", "")),
                    "collection_number": str(row.get("Collection No.", "")),
                    "stage_or_type": str(
                        row.get("Stage (Pokémon)/Type (Energy and Trainer)", "")
                    ),
                },
            )
    return result


def card_image_url(expansion: str, collection_number: str) -> str | None:
    set_id = SET_IMAGE_IDS.get(expansion)
    number = collection_number.strip()
    if not set_id or not number:
        return None
    if set_id in SCRYDEX_SET_IDS:
        return f"https://images.scrydex.com/pokemon/{set_id}-{number}/small"
    return f"https://images.pokemontcg.io/{set_id}/{number}.png"
