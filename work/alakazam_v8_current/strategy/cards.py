from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


# V8 deck card IDs.
ALAKAZAM = 743
ABRA = 741
KADABRA = 742
DUNSPARCE = 305
DUDUNSPARCE = 66
FEZANDIPITI_EX = 140
SHAYMIN = 343

BASIC_PSYCHIC = 5
ENRICHING_ENERGY = 13
TELEPATH_ENERGY = 19

RARE_CANDY = 1079
ENHANCED_HAMMER = 1081
POFFIN = 1086
NIGHT_STRETCHER = 1097
SACRED_ASH = 1129
POKE_PAD = 1152
BOSS_ORDERS = 1182
LANAS_AID = 1184
XEROSIC = 1197
HILDA = 1225
DAWN = 1231
NIGHTTIME_MINE = 1266

# Opponent cards/effects that change legal or strategic behavior.
BUDEW = 235
MIST_ENERGY = 11
ROCK_FIGHTING_ENERGY = 20

PSYCHIC_ENERGY_TYPE = 5
FIGHTING_ENERGY_TYPE = 6
TRADING_PLACES_ATTACK = 423
TELEPORTATION_ATTACK = 1070
ITCHY_POLLEN_ATTACK = 323
POWERFUL_HAND_ATTACK = 1072

POKEMON = frozenset(
    {ALAKAZAM, ABRA, KADABRA, DUNSPARCE, DUDUNSPARCE, FEZANDIPITI_EX, SHAYMIN}
)
ATTACK_LINE = frozenset({ABRA, KADABRA, ALAKAZAM})
EVOLUTION = frozenset({KADABRA, ALAKAZAM, DUDUNSPARCE})
SUPPORTERS = frozenset({BOSS_ORDERS, LANAS_AID, XEROSIC, HILDA, DAWN})
ITEMS = frozenset(
    {RARE_CANDY, ENHANCED_HAMMER, POFFIN, NIGHT_STRETCHER, SACRED_ASH, POKE_PAD}
)
BASIC_ENERGY_IDS = frozenset(range(1, 10))
PROTECTIVE_DAMAGE_ENERGIES = frozenset({MIST_ENERGY, ROCK_FIGHTING_ENERGY})
RULE_BOX_POKEMON = frozenset({FEZANDIPITI_EX})

ATTACK_DAMAGE = MappingProxyType({ABRA: 10, KADABRA: 30, DUNSPARCE: 20, DUDUNSPARCE: 90})
ATTACK_ENERGY_COUNT = MappingProxyType({ABRA: 1, KADABRA: 1, DUNSPARCE: 1, DUDUNSPARCE: 3})
DRAW_ABILITY_GAIN = MappingProxyType(
    {KADABRA: 2, ALAKAZAM: 3, DUDUNSPARCE: 3, FEZANDIPITI_EX: 3}
)


@dataclass(frozen=True)
class DeckSpec:
    cards: tuple[int, ...]
    counts: Mapping[int, int]

    @property
    def card_ids(self) -> frozenset[int]:
        return frozenset(self.counts)

    def count(self, card_id: int) -> int:
        return int(self.counts.get(card_id, 0))


def candidate_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_deck(root: Path | None = None) -> list[int]:
    deck_path = (root or candidate_root()) / "deck.csv"
    cards = [
        int(line.strip())
        for line in deck_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(cards) != 60:
        raise ValueError(f"{deck_path.name} must contain 60 cards, got {len(cards)}")
    return cards


def make_deck_spec(deck: list[int] | tuple[int, ...]) -> DeckSpec:
    cards = tuple(int(card_id) for card_id in deck)
    if len(cards) != 60:
        raise ValueError(f"deck must contain 60 cards, got {len(cards)}")
    return DeckSpec(cards=cards, counts=MappingProxyType(dict(Counter(cards))))
