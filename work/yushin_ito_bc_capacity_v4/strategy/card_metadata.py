from __future__ import annotations

import csv
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any


CARD_METADATA_DIM = 12
DEFAULT_CARD_DATA = Path(__file__).resolve().parents[2] / "data" / "official" / "EN_Card_Data.csv"


def _number(value: Any, divisor: float) -> float:
    try:
        return float(value) / divisor
    except (TypeError, ValueError):
        return 0.0


@lru_cache(maxsize=4)
def load_card_metadata(path: str | Path = DEFAULT_CARD_DATA) -> dict[int, list[float]]:
    """Build compact, language-independent card semantics from official data."""
    source = Path(path)
    rows_by_card: dict[int, list[dict[str, str]]] = defaultdict(list)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                card_id = int(row.get("Card ID", ""))
            except (TypeError, ValueError):
                continue
            rows_by_card[card_id].append(row)

    result: dict[int, list[float]] = {}
    for card_id, rows in rows_by_card.items():
        first = rows[0]
        stage = str(first.get("Stage (Pokémon)/Type (Energy and Trainer)", "")).lower()
        category = str(first.get("Category", "")).lower()
        card_name = str(first.get("Card Name", "")).lower()
        is_energy = "energy" in stage or "energy" in card_name
        is_trainer = any(token in stage for token in ("item", "supporter", "stadium", "tool"))
        is_pokemon = not is_energy and not is_trainer and first.get("HP") not in (None, "", "n/a")
        attack_rows = [row for row in rows if str(row.get("Move Name", "")).strip() not in ("", "n/a")]
        effect_chars = sum(len(str(row.get("Effect Explanation", ""))) for row in rows)
        result[card_id] = [
            float(is_pokemon),
            float(is_trainer),
            float(is_energy),
            float("basic" in stage and is_pokemon),
            float("stage 1" in stage),
            float("stage 2" in stage),
            _number(first.get("HP", 0), 400.0),
            _number(first.get("Retreat", 0), 5.0),
            float(bool(str(first.get("Weakness", "")).strip())),
            float(bool(str(first.get("Resistance (Type)", "")).strip())),
            min(len(attack_rows), 4) / 4.0,
            min(effect_chars, 1000) / 1000.0,
        ]
    return result


def serialize_card_metadata(values: dict[int, list[float]]) -> dict[str, list[float]]:
    return {str(card_id): list(features) for card_id, features in sorted(values.items())}
