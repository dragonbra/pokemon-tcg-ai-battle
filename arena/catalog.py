from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
import re
from typing import Mapping


def load_card_metadata(path: Path) -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            int(row["Card ID"]): {str(key): str(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        }


def classify_deck(
    deck: list[int],
    card_metadata: Mapping[int, Mapping[str, str]],
    overrides: Mapping[str, object] | None = None,
) -> tuple[str, tuple[int, ...], str]:
    overrides = overrides or {}
    pokemon = Counter(
        card_id
        for card_id in deck
        if _is_pokemon(card_metadata.get(card_id, {}))
    )
    primary_ids = tuple(
        card_id
        for card_id, _ in sorted(
            pokemon.items(),
            key=lambda item: (
                -item[1],
                -_stage_rank(card_metadata.get(item[0], {})),
                -int(" ex" in card_metadata.get(item[0], {}).get("Card Name", "").lower()),
                card_metadata.get(item[0], {}).get("Card Name", ""),
                item[0],
            ),
        )[:3]
    )
    if not primary_ids:
        return "Unknown Archetype", (), "Unknown Archetype"
    names = {
        card_metadata.get(card_id, {}).get("Card Name", "")
        for card_id in pokemon
    }
    primary_name = card_metadata[primary_ids[0]].get("Card Name", "Unknown Pokémon")
    archetype = str(overrides.get("archetype", _known_archetype(names, primary_name)))
    display_name = str(overrides.get("display_name", archetype))
    return archetype, primary_ids, display_name


def _is_pokemon(card: Mapping[str, str]) -> bool:
    value = card.get("Stage (Pokémon)/Type (Energy and Trainer)", "")
    return value.strip().endswith(("Pokémon", "Pokemon"))


def _stage_rank(card: Mapping[str, str]) -> int:
    value = card.get("Stage (Pokémon)/Type (Energy and Trainer)", "")
    if value.startswith("Stage 2"):
        return 3
    if value.startswith("Stage 1"):
        return 2
    if value.startswith("Basic"):
        return 1
    return 0


def _known_archetype(names: set[str], fallback: str) -> str:
    joined = " ".join(names).casefold()
    for needle, label in (
        ("rocket's mewtwo", "Rocket Team Mewtwo"),
        ("rocket mewtwo", "Rocket Team Mewtwo"),
        ("alakazam", "Alakazam"),
        ("lucario", "Lucario"),
        ("archaludon", "Archaludon"),
        ("abomasnow", "Mega Abomasnow"),
        ("dragapult", "Dragapult"),
        ("gengar", "Gengar"),
        ("starmie", "Starmie"),
    ):
        if needle in joined:
            return label
    return fallback.replace(" ex", "")


def card_image_url(card: Mapping[str, str]) -> str | None:
    """根据官方 CSV 的 expansion/collection 生成稳定的卡图地址。"""
    expansion = card.get("Expansion", "").strip()
    collection = card.get("Collection No.", "").strip()
    if not expansion or not re.fullmatch(r"[0-9A-Za-z-]+", collection):
        return None
    return f"https://images.pokemontcg.io/{expansion}/{collection}.png"
