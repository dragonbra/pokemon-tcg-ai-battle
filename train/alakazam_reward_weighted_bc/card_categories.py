from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


DEFAULT_CARD_DATA = Path(__file__).resolve().parents[2] / "data" / "official" / "EN_Card_Data.csv"
CATEGORY_NAMES = (
    "Basic Pokémon",
    "Stage 1 Pokémon",
    "Stage 2 Pokémon",
    "Item",
    "Supporter",
    "Pokémon Tool",
    "Stadium",
    "Special Energy",
    "Basic Energy",
)
CATEGORY_TO_ID = {name: index + 1 for index, name in enumerate(CATEGORY_NAMES)}


def load_card_category_lookup(
    path: str | Path = DEFAULT_CARD_DATA,
    *,
    card_vocab_size: int = 4096,
) -> tuple[list[int], dict[str, int]]:
    """Map padded model card IDs (raw ID + 1) to an exact official category."""
    lookup = [0] * (card_vocab_size + 1)
    seen: dict[int, int] = {}
    counts: Counter[str] = Counter()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                raw_card_id = int(row.get("Card ID", ""))
            except (TypeError, ValueError):
                continue
            category = str(
                row.get("Stage (Pokémon)/Type (Energy and Trainer)", "")
            ).strip()
            if category not in CATEGORY_TO_ID:
                raise ValueError(f"unsupported official card category: {category!r}")
            category_id = CATEGORY_TO_ID[category]
            previous = seen.setdefault(raw_card_id, category_id)
            if previous != category_id:
                raise ValueError(f"card {raw_card_id} has conflicting official categories")
    for raw_card_id, category_id in seen.items():
        encoded_id = min(card_vocab_size, raw_card_id + 1)
        previous = lookup[encoded_id]
        if previous not in (0, category_id):
            raise ValueError(f"encoded card ID {encoded_id} aliases different categories")
        lookup[encoded_id] = category_id
        counts[CATEGORY_NAMES[category_id - 1]] += 1
    return lookup, dict(sorted(counts.items()))
