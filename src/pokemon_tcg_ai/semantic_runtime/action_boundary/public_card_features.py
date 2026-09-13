"""Public card-derived features shared by CUDA, official, and package macros."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping


@lru_cache(maxsize=None)
def card_prize_counts(prototype_path: Path) -> dict[int, int]:
    """Return public printed Prize liability by card ID.

    The official observation exposes a Bench card's public identity but does not
    include its Prize liability.  Derive it from the same versioned public card
    ontology in every runtime.  `startswith("mega ")` is intentional: Yanmega ex
    is a regular two-Prize Pokémon ex, not a Mega Evolution Pokémon ex.
    """

    payload = json.loads(Path(prototype_path).read_text(encoding="utf-8"))
    cards = payload.get("cards")
    if not isinstance(cards, list):
        raise ValueError("prototype payload has no cards list")
    result: dict[int, int] = {}
    for row in cards:
        if not isinstance(row, Mapping):
            raise ValueError("prototype card row is not a mapping")
        card_id = int(row["card_id"])
        name = str(row.get("name_en") or "").strip().lower()
        if row.get("no_prize"):
            prizes = 0
        elif name.startswith("mega ") and " ex" in name:
            prizes = 3
        elif " ex" in name:
            prizes = 2
        else:
            prizes = 1
        result[card_id] = prizes
    if not result:
        raise ValueError("prototype Prize map is empty")
    return result


def with_public_prize(
    target: Mapping[str, Any], prize_counts: Mapping[int, int]
) -> dict[str, Any]:
    result = dict(target)
    card_id = result.get("id")
    if not isinstance(card_id, int) or isinstance(card_id, bool):
        raise ValueError("visible target has no public integer card ID")
    if card_id not in prize_counts:
        raise ValueError(f"visible card ID is absent from Prize map: {card_id}")
    result["prize"] = int(prize_counts[card_id])
    return result


__all__ = ["card_prize_counts", "with_public_prize"]
