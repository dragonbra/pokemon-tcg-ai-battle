"""Alakazam V1: a deterministic, card-aware baseline policy.

The policy follows the researched game plan in a deliberately conservative
form: establish Abra and Dunsparce, evolve the attack line, use draw abilities,
attach Psychic Energy, and preserve enough hand cards for Powerful Hand. It
only chooses from options supplied by the simulator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DECK_PATH = ROOT / "deck.csv"

# Card IDs from data/official/EN_Card_Data.csv.
ALAKAZAM = 743
ABRA = 741
KADABRA = 742
DUNSPARCE = 305
DUDUNSPARCE = 66
FEZANDIPITI_EX = 140
PSYDUCK = 858
SHAYMIN = 343

ENRICHING_ENERGY = 13
TELEPATH_ENERGY = 19
BASIC_PSYCHIC = 5

ERI = 1186
NIGHT_STRETCHER = 1097
POFFIN = 1086
RARE_CANDY = 1079
WONDROUS_PATCH = 1146
SACRED_ASH = 1129
BATTLE_CAGE = 1264
LANAS_AID = 1184
POKE_PAD = 1152
HILDA = 1225
DAWN = 1231
ENHANCED_HAMMER = 1081
BOSS_ORDERS = 1182

POKEMON = {ALAKAZAM, ABRA, KADABRA, DUNSPARCE, DUDUNSPARCE, FEZANDIPITI_EX, PSYDUCK, SHAYMIN}
EVOLUTION = {ALAKAZAM, KADABRA, DUDUNSPARCE}
BASIC_SETUP = {ABRA, DUNSPARCE, FEZANDIPITI_EX, PSYDUCK, SHAYMIN}
DRAW_CARDS = {KADABRA, ALAKAZAM, DUDUNSPARCE}


def read_deck_csv() -> list[int]:
    values = [line.strip() for line in DECK_PATH.read_text(encoding="utf-8").splitlines()]
    deck = [int(value) for value in values if value]
    if len(deck) != 60:
        raise ValueError(f"{DECK_PATH.name} must contain 60 cards, got {len(deck)}")
    return deck


def _cards(player: dict[str, Any], area: str) -> list[dict[str, Any]]:
    return player.get(area) or []


def _card_ids(cards: Iterable[dict[str, Any] | None]) -> list[int]:
    return [card["id"] for card in cards if card is not None and card.get("id") is not None]


def _your_state(obs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    current = obs.get("current") or {}
    players = current.get("players") or []
    your_index = int(current.get("yourIndex", 0))
    return current, players[your_index]


def _hand_ids(player: dict[str, Any]) -> list[int]:
    return _card_ids(player.get("hand") or [])


def _active(player: dict[str, Any]) -> dict[str, Any] | None:
    active = _cards(player, "active")
    return active[0] if active else None


def _bench(player: dict[str, Any]) -> list[dict[str, Any]]:
    return _cards(player, "bench")


def _energy_count(pokemon: dict[str, Any] | None) -> int:
    if not pokemon:
        return 0
    return len(pokemon.get("energies") or [])


def _hand_index_score(card_id: int, hand: list[int], player: dict[str, Any]) -> int:
    """Return a discard score; higher means a more acceptable discard."""
    counts = {card: hand.count(card) for card in set(hand)}
    if card_id in {ENRICHING_ENERGY, TELEPATH_ENERGY, BASIC_PSYCHIC}:
        return 3 if counts.get(card_id, 0) > 1 else 0
    if card_id in EVOLUTION:
        return 1 if counts.get(card_id, 0) > 1 else 0
    if card_id in {ERI, ENHANCED_HAMMER, SACRED_ASH, WONDROUS_PATCH}:
        return 2
    if card_id in {ABRA, KADABRA, ALAKAZAM, DUNSPARCE, DUDUNSPARCE}:
        return 0
    return 1


def _choose_card_option(
    options: list[dict[str, Any]],
    player: dict[str, Any],
    context: int,
    *,
    count: int,
) -> list[int]:
    """Choose visible card options for setup and effect selections."""
    hand = _hand_ids(player)
    bench_count = len(_bench(player))
    active_id = (_active(player) or {}).get("id")

    def score(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        index, option = item
        card_id = option.get("cardId")
        if card_id is None:
            card_index = option.get("index")
            card_id = hand[card_index] if card_index is not None and card_index < len(hand) else None

        value = 0
        if context == 1:  # SETUP_ACTIVE_POKEMON
            value = {ABRA: 100, DUNSPARCE: 90, FEZANDIPITI_EX: 35, PSYDUCK: 20, SHAYMIN: 10}.get(card_id, 0)
        elif context in {2, 5, 6}:  # SETUP_BENCH / TO_BENCH / TO_FIELD
            value = {ABRA: 100, DUNSPARCE: 90, FEZANDIPITI_EX: 45, PSYDUCK: 20, SHAYMIN: 10}.get(card_id, 0)
            if card_id in {ABRA, DUNSPARCE} and bench_count >= 4:
                value -= 30
        elif context in {7, 9, 10}:  # TO_HAND / TO_DECK / TO_DECK_BOTTOM
            value = {ALAKAZAM: 90, KADABRA: 80, DUDUNSPARCE: 75, ABRA: 65, DUNSPARCE: 60}.get(card_id, 20)
        elif context in {8, 29}:  # DISCARD / DISCARD_CARD_OR_ATTACHED_CARD
            value = _hand_index_score(card_id, hand, player) if card_id is not None else 0
        elif card_id in DRAW_CARDS:
            value = 80
        elif card_id in {ABRA, DUNSPARCE}:
            value = 70
        else:
            value = 10
        if card_id == active_id:
            value -= 10
        return -value, index

    ranked = sorted(enumerate(options), key=score)
    return [index for index, _ in ranked[:count]]


def _choose_energy_option(options: list[dict[str, Any]], player: dict[str, Any]) -> list[int]:
    """Prefer Psychic Energy attached to the current or next attacker."""
    active = _active(player)
    bench = _bench(player)

    def score(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        index, option = item
        area = option.get("inPlayArea")
        target_index = option.get("inPlayIndex")
        target = active if area == 4 and target_index == 0 else bench[target_index] if area == 5 and target_index is not None and target_index < len(bench) else None
        target_id = (target or {}).get("id")
        value = {
            ALAKAZAM: 100,
            KADABRA: 80,
            ABRA: 65,
            DUNSPARCE: 20,
            DUDUNSPARCE: 25,
        }.get(target_id, 10)
        if target is active and target_id == ALAKAZAM and _energy_count(target) < 1:
            value += 30
        return -value, index

    ranked = sorted(enumerate(options), key=score)
    return [ranked[0][0]] if ranked else []


def _choose_switch_option(options: list[dict[str, Any]], player: dict[str, Any]) -> list[int]:
    bench = _bench(player)
    def score(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        index, option = item
        target_index = option.get("index", option.get("inPlayIndex"))
        target = bench[target_index] if target_index is not None and target_index < len(bench) else None
        target_id = (target or {}).get("id")
        return -{ALAKAZAM: 100, KADABRA: 70, ABRA: 45, DUNSPARCE: 20}.get(target_id, 10), index
    ranked = sorted(enumerate(options), key=score)
    return [ranked[0][0]] if ranked else []


def _select_effect(obs: dict[str, Any]) -> list[int]:
    select = obs["select"]
    options = select.get("option") or []
    min_count = int(select.get("minCount", 0))
    max_count = int(select.get("maxCount", len(options)))
    current, player = _your_state(obs)
    context = int(select.get("context", 0))
    select_type = int(select.get("type", 0))

    if not options:
        if min_count:
            raise ValueError("simulator returned a required selection with no options")
        return []

    if context in {1, 2, 5, 6, 7, 8, 9, 10, 29}:
        count = min_count if min_count else min(max_count, 2)
        return _choose_card_option(options, player, context, count=count)

    if select_type == 4:  # ENERGY
        count = max(1, min_count)
        return _choose_energy_option(options, player)[:count]

    if context in {3, 4}:  # SWITCH / TO_ACTIVE
        return _choose_switch_option(options, player)

    if select_type == 8:  # COUNT
        # Powerful draw/search effects benefit from the largest legal value.
        ranked = sorted(enumerate(options), key=lambda item: -(item[1].get("number", 0)))
        count = min_count if min_count else 1
        return [index for index, _ in ranked[:count]]

    # For yes/no effects, accept the effect when it is available. For other
    # required selections, first legal is safer than inventing hidden state.
    if select_type == 9 or all(option.get("type") in {1, 2} for option in options):
        yes = [index for index, option in enumerate(options) if option.get("type") == 1]
        if yes and min_count:
            return yes[:min_count]
        if not min_count:
            return yes[:1]
    count = min_count if min_count else 0
    return list(range(min(count, max_count, len(options))))


def _active_attack_score(option: dict[str, Any], active_id: int | None, hand_count: int) -> tuple[int, int]:
    attack_id = option.get("attackId", -1)
    # The V1 attack plan prefers Powerful Hand when available. The engine
    # exposes only legal attacks, so unknown attack IDs remain valid fallbacks.
    if active_id == ALAKAZAM:
        return (0 if attack_id >= 0 else 10, attack_id)
    if active_id == KADABRA:
        return (1, attack_id)
    return (2, attack_id)


def _main_action(obs: dict[str, Any]) -> list[int]:
    select = obs["select"]
    options = select.get("option") or []
    current, player = _your_state(obs)
    hand = _hand_ids(player)
    active = _active(player)
    active_id = (active or {}).get("id")
    bench = _bench(player)
    bench_space = int(player.get("benchMax", 5)) - len(bench)

    def score(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        index, option = item
        option_type = option.get("type")
        card_index = option.get("index")
        card_id = hand[card_index] if card_index is not None and card_index < len(hand) else None

        if option_type == 10 and card_id in DRAW_CARDS:
            return (0, index)
        if option_type == 9:
            evolved_id = card_id
            return ({ALAKAZAM: 1, KADABRA: 2, DUDUNSPARCE: 3}.get(evolved_id, 4), index)
        if option_type == 8:
            energy_id = card_id
            target_area = option.get("inPlayArea")
            target_index = option.get("inPlayIndex")
            target = active if target_area == 4 else bench[target_index] if target_area == 5 and target_index is not None and target_index < len(bench) else None
            target_id = (target or {}).get("id")
            if energy_id == TELEPATH_ENERGY:
                return (4 if target_id in {ALAKAZAM, KADABRA, ABRA} else 7, index)
            if energy_id == BASIC_PSYCHIC:
                return (5 if target_id in {ALAKAZAM, KADABRA, ABRA} else 8, index)
            if energy_id == ENRICHING_ENERGY:
                return (6, index)
            return (8, index)
        if option_type == 7:
            priorities = {
                DAWN: 10,
                HILDA: 11,
                POFFIN: 12 if bench_space else 30,
                RARE_CANDY: 13,
                BATTLE_CAGE: 14,
                POKE_PAD: 15,
                BOSS_ORDERS: 18,
                NIGHT_STRETCHER: 19,
                WONDROUS_PATCH: 20,
                LANAS_AID: 21,
                SACRED_ASH: 22,
                ENHANCED_HAMMER: 23,
                ERI: 24,
            }
            return priorities.get(card_id, 25), index
        if option_type == 13:
            return (30,) + _active_attack_score(option, active_id, len(hand))
        if option_type == 12:
            return (40, index)
        if option_type == 14:
            return (99, index)
        return (50, index)

    ranked = sorted(enumerate(options), key=score)
    return [ranked[0][0]] if ranked else []


def agent(obs_dict: dict[str, Any]) -> list[int]:
    """Return one legal action for the Alakazam V1 policy."""
    if obs_dict.get("select") is None:
        return read_deck_csv()

    select = obs_dict["select"]
    if int(select.get("type", 0)) == 0 and int(select.get("context", 0)) == 0:
        return _main_action(obs_dict)
    return _select_effect(obs_dict)
