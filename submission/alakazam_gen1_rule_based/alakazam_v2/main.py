"""Alakazam V2: a deterministic, attack-first Alakazam policy.

V2 keeps the V1 card-aware option handling but makes the turn objective
explicit: prepare an Alakazam attack, take a legal knockout when available,
and only spend draw or gust resources when they change the result. It only
chooses from options supplied by the simulator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def _submission_root() -> Path:
    """Resolve the directory where Kaggle unpacked the submission bundle."""
    source_file = globals().get("__file__")
    if isinstance(source_file, str):
        source_root = Path(source_file).resolve().parent
        if (source_root / "deck.csv").exists():
            return source_root

    # Kaggle's simulation loader may execute main.py with ``exec`` and omit
    # __file__. The official FAQ says submission files are then available in
    # /kaggle_simulations/agent, while cwd remains /kaggle/working.
    kaggle_root = Path("/kaggle_simulations/agent")
    if (kaggle_root / "deck.csv").exists():
        return kaggle_root

    return Path.cwd()


ROOT = _submission_root()
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
BASIC_SETUP = {ABRA, DUNSPARCE, FEZANDIPITI_EX}
DRAW_CARDS = {KADABRA, ALAKAZAM, DUDUNSPARCE}
ATTACK_LINE = {ABRA, KADABRA, ALAKAZAM}
PSYCHIC_ENERGY_TYPE = 5

# Alakazam only needs one Psychic Energy to attack. Do not spend a second
# Energy on an Alakazam when another legal target can use it.
ENERGY_CAPPED_POKEMON = {ALAKAZAM}

ATTACK_DAMAGE = {
    ABRA: 10,
    DUNSPARCE: 20,
    DUDUNSPARCE: 90,
    KADABRA: 30,
}

ATTACK_ENERGY_COUNT = {
    ABRA: 1,
    DUNSPARCE: 2,
    DUDUNSPARCE: 3,
    KADABRA: 1,
}

DRAW_ABILITY_GAIN = {
    KADABRA: 2,
    ALAKAZAM: 3,
    DUDUNSPARCE: 3,
    FEZANDIPITI_EX: 3,
}

RULE_BOX_POKEMON = {FEZANDIPITI_EX, 306, 389, 481, 723, 997}
_EFFECT_PROGRESS: dict[int, int] = {}


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
    return current, players[your_index] if your_index < len(players) else {}


def _your_index(current: dict[str, Any]) -> int:
    return int(current.get("yourIndex", 0))


def _player(current: dict[str, Any], player_index: int) -> dict[str, Any]:
    players = current.get("players") or []
    return players[player_index] if 0 <= player_index < len(players) else {}


def _opponent_state(current: dict[str, Any]) -> dict[str, Any]:
    return _player(current, 1 - _your_index(current))


def _hand_ids(player: dict[str, Any]) -> list[int]:
    return _card_ids(player.get("hand") or [])


def _active(player: dict[str, Any]) -> dict[str, Any] | None:
    active = _cards(player, "active")
    return active[0] if active else None


def _bench(player: dict[str, Any]) -> list[dict[str, Any]]:
    return _cards(player, "bench")


def _field_pokemon(player: dict[str, Any]) -> list[dict[str, Any]]:
    return [card for card in [*_cards(player, "active"), *_bench(player)] if card]


def _field_ids(player: dict[str, Any]) -> set[int]:
    return set(_card_ids(_field_pokemon(player)))


def _has_field_card(player: dict[str, Any], card_ids: set[int]) -> bool:
    return bool(_field_ids(player) & card_ids)


def _discard_ids(player: dict[str, Any]) -> list[int]:
    return _card_ids(player.get("discard") or [])


def _has_psychic_energy(pokemon: dict[str, Any] | None) -> bool:
    return PSYCHIC_ENERGY_TYPE in (pokemon or {}).get("energies", [])


def _has_ready_alakazam(player: dict[str, Any]) -> bool:
    return any(
        pokemon.get("id") == ALAKAZAM and _has_psychic_energy(pokemon)
        for pokemon in _field_pokemon(player)
    )


def _has_attack_ready_pokemon(player: dict[str, Any]) -> bool:
    """Whether the Active can attack now without paying a retreat cost."""
    active = _active(player)
    if not active:
        return False
    active_id = active.get("id")
    if active_id == ALAKAZAM:
        return _has_psychic_energy(active)
    return _energy_count(active) >= ATTACK_ENERGY_COUNT.get(active_id, 99)


def _has_alakazam_attack_path(player: dict[str, Any]) -> bool:
    """Whether the Active can evolve into an energized Alakazam immediately."""
    active = _active(player)
    if not active or active.get("id") not in {ABRA, KADABRA}:
        return False
    hand_ids = _hand_ids(player)
    has_psychic = _has_psychic_energy(active)
    return has_psychic and ALAKAZAM in hand_ids and (
        RARE_CANDY in hand_ids or active.get("id") == KADABRA
    )


def _has_usable_attack_path(current: dict[str, Any], player: dict[str, Any]) -> bool:
    """Whether this turn has an attack now or a visible immediate evolution path."""
    return _has_attack_ready_pokemon(player) or _has_alakazam_attack_path(player)


def _boss_changes_prize_race(
    current: dict[str, Any], player: dict[str, Any], attack_is_legal: bool
) -> bool:
    """Require Boss's Orders to expose a knockout or a higher-value prize."""
    if not attack_is_legal:
        return False
    active = _active(player)
    damage = _estimated_attack_damage(
        (active or {}).get("id"),
        int(player.get("handCount", len(player.get("hand") or []))),
    )
    opponent = _opponent_state(current)
    target = _opponent_active(current)
    if not target or damage <= 0:
        return False
    active_ko = int(target.get("hp", 9999)) <= damage
    active_prize = _prize_value(target.get("id"))
    return not active_ko and any(
        int(pokemon.get("hp", 9999)) <= damage
        and _prize_value(pokemon.get("id")) >= active_prize
        for pokemon in _bench(opponent)
    )


def _retreat_improves_attack(player: dict[str, Any]) -> bool:
    """Allow retreat only when an energized bench attacker can take over."""
    active = _active(player)
    if not active or _has_attack_ready_pokemon(player):
        return False
    return any(
        pokemon.get("id") == ALAKAZAM and _has_psychic_energy(pokemon)
        for pokemon in _bench(player)
    )


def _own_turn_number(current: dict[str, Any]) -> int:
    """Convert the simulator's shared turn counter into each player's turn."""
    turn = int(current.get("turn", 0))
    your_index = _your_index(current)
    return (turn + (1 if your_index == 0 else 0)) // 2


def _needs_attack_line(player: dict[str, Any]) -> bool:
    """Whether the board lacks a future Abra -> Alakazam attacker."""
    field_ids = _field_ids(player)
    hand_ids = set(_hand_ids(player))
    if field_ids & {ALAKAZAM, KADABRA}:
        return False
    if ABRA in field_ids:
        return not bool(hand_ids & {KADABRA, ALAKAZAM})
    return not bool((field_ids | hand_ids) & ATTACK_LINE)


def _recovery_need(player: dict[str, Any]) -> int:
    """Score how much the discard pile can rebuild the next attack line."""
    discard = set(_discard_ids(player))
    return len(discard & {ABRA, KADABRA, ALAKAZAM, DUNSPARCE, DUDUNSPARCE}) + len(
        discard & {BASIC_PSYCHIC}
    )


def _pokemon_from_option(option: dict[str, Any], current: dict[str, Any]) -> dict[str, Any] | None:
    owner = int(option.get("playerIndex", _your_index(current)))
    area = option.get("inPlayArea", option.get("area"))
    index = option.get("inPlayIndex", option.get("index"))
    player = _player(current, owner)
    if area == 4 and isinstance(index, int) and 0 <= index < len(_cards(player, "active")):
        active = _cards(player, "active")
        return active[index]
    if area == 5 and isinstance(index, int) and 0 <= index < len(_bench(player)):
        bench = _bench(player)
        return bench[index]
    return None


def _option_card_id(
    option: dict[str, Any],
    select: dict[str, Any],
    current: dict[str, Any],
) -> int | None:
    """Resolve a visible option card without confusing deck and hand indexes."""
    if option.get("cardId") is not None:
        return int(option["cardId"])

    index = option.get("index")
    if index is None:
        return None
    option_type = option.get("type")
    area = option.get("area")
    owner = int(option.get("playerIndex", _your_index(current)))
    player = _player(current, owner)

    if option_type == 7:  # PLAY always indexes your hand.
        hand = player.get("hand") or []
        return (
            hand[index].get("id")
            if owner == _your_index(current) and 0 <= index < len(hand)
            else None
        )
    if area == 1:  # Search effects index the cards in select.deck.
        deck = select.get("deck") or []
        return deck[index].get("id") if 0 <= index < len(deck) else None
    if area == 2:  # Opponent hand cards are hidden and cannot be inferred.
        hand = player.get("hand") or []
        return (
            hand[index].get("id")
            if owner == _your_index(current) and 0 <= index < len(hand)
            else None
        )
    if area == 3:
        discard = player.get("discard") or []
        if 0 <= index < len(discard):
            return discard[index].get("id")
        return None

    # Attached-energy selections identify the Pokémon with area/index and the
    # attached card with energyIndex. Resolve this before the plain field-card
    # cases below.
    energy_index = option.get("energyIndex")
    if energy_index is not None:
        pokemon = _pokemon_from_option(option, current)
        energy_cards = (pokemon or {}).get("energyCards") or []
        if 0 <= energy_index < len(energy_cards):
            return energy_cards[energy_index].get("id")
    if area == 4:
        active = _cards(player, "active")
        return active[index].get("id") if 0 <= index < len(active) and active[index] else None
    if area == 5:
        bench = _bench(player)
        return bench[index].get("id") if 0 <= index < len(bench) else None
    if area == 6:
        prize = player.get("prize") or []
        card = prize[index] if 0 <= index < len(prize) else None
        return card.get("id") if card else None
    return None


def _effect_id(select: dict[str, Any]) -> int | None:
    effect = select.get("effect") or {}
    return effect.get("id")


def _effect_step(select: dict[str, Any]) -> int:
    effect = select.get("effect") or {}
    serial = effect.get("serial")
    return _EFFECT_PROGRESS.get(serial, 0) if serial is not None else 0


def _advance_effect(select: dict[str, Any]) -> None:
    effect = select.get("effect") or {}
    serial = effect.get("serial")
    if serial is not None:
        _EFFECT_PROGRESS[serial] = _EFFECT_PROGRESS.get(serial, 0) + 1


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
    select: dict[str, Any],
    current: dict[str, Any],
    player: dict[str, Any],
    context: int,
    *,
    count: int,
) -> list[int]:
    """Choose visible card options for setup and effect selections."""
    hand = _hand_ids(player)
    bench_count = len(_bench(player))
    active_id = (_active(player) or {}).get("id")
    effect_id = _effect_id(select)
    effect_step = _effect_step(select)

    def category(card_id: int | None) -> int:
        if effect_id == DAWN:
            groups = (
                BASIC_SETUP,
                {KADABRA, DUDUNSPARCE},
                {ALAKAZAM},
            )
            group = groups[min(effect_step, len(groups) - 1)]
            if any(_option_card_id(option, select, current) in group for option in options):
                return 0 if card_id in group else 4
        if effect_id == HILDA:
            wanted = (
                EVOLUTION
                if effect_step == 0
                else {TELEPATH_ENERGY, BASIC_PSYCHIC, ENRICHING_ENERGY}
            )
            if any(_option_card_id(option, select, current) in wanted for option in options):
                return 0 if card_id in wanted else 4
        if effect_id == POFFIN:
            # Buddy-Buddy Poffin is limited to Basic Pokémon with 70 HP or
            # less. Keep the V2 opening plan focused on Abra and Dunsparce.
            wanted = {ABRA, DUNSPARCE}
            return 0 if card_id in wanted else 4
        if effect_id == TELEPATH_ENERGY:
            return 0 if card_id == ABRA else 4
        if effect_id == POKE_PAD:
            return 0 if card_id in POKEMON - RULE_BOX_POKEMON else 4
        return 2

    def score(item: tuple[int, dict[str, Any]]) -> tuple[int, int, int]:
        index, option = item
        card_id = _option_card_id(option, select, current)

        value = 0
        if context == 1:  # SETUP_ACTIVE_POKEMON
            value = {
                ABRA: 100,
                DUNSPARCE: 90,
                FEZANDIPITI_EX: 25,
                PSYDUCK: 5,
                SHAYMIN: 0,
            }.get(card_id, 0)
        elif context in {2, 5, 6}:  # SETUP_BENCH / TO_BENCH / TO_FIELD
            value = {
                ABRA: 100,
                DUNSPARCE: 90,
                FEZANDIPITI_EX: 25,
                PSYDUCK: 5,
                SHAYMIN: 0,
            }.get(card_id, 0)
            if card_id in {ABRA, DUNSPARCE} and bench_count >= 4:
                value -= 30
        elif context in {7, 9, 10}:  # TO_HAND / TO_DECK / TO_DECK_BOTTOM
            value = {
                ALAKAZAM: 100,
                KADABRA: 90,
                DUDUNSPARCE: 85,
                ABRA: 75,
                DUNSPARCE: 70,
                TELEPATH_ENERGY: 60,
                BASIC_PSYCHIC: 55,
                ENRICHING_ENERGY: 45,
            }.get(card_id, 20)
        elif context == 21:  # ATTACH_FROM, used by Wondrous Patch
            value = {BASIC_PSYCHIC: 100, TELEPATH_ENERGY: 20}.get(card_id, 0)
        elif context == 26:  # DISCARD_ENERGY_CARD, used by Enhanced Hammer
            value = {TELEPATH_ENERGY: 100, ENRICHING_ENERGY: 90}.get(card_id, 10)
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
        return category(card_id), -value, index

    ranked = sorted(enumerate(options), key=score)
    return [index for index, _ in ranked[:count]]


def _choose_energy_option(
    options: list[dict[str, Any]], current: dict[str, Any], player: dict[str, Any]
) -> list[int]:
    """Attach one useful Energy, with V2's special-energy discipline."""
    active = _active(player)
    hand = _hand_ids(player)
    early_setup = _own_turn_number(current) <= 1
    has_special_in_hand = TELEPATH_ENERGY in hand

    def score(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        index, option = item
        target = _pokemon_from_option(option, current)
        target_id = (target or {}).get("id")
        energy_id = _option_card_id(option, {"type": 0}, current)
        target_energy_count = _energy_count(target)

        # A second Energy on Alakazam is never part of the V2 plan. The
        # simulator should normally omit that option; a high score keeps it
        # as a last-resort legal fallback if it does not.
        if target_id in ENERGY_CAPPED_POKEMON and target_energy_count >= 1:
            return 90, index

        value = {
            ALAKAZAM: 100,
            KADABRA: 80,
            ABRA: 65,
            DUNSPARCE: 20,
            DUDUNSPARCE: 25,
        }.get(target_id, 10)
        if target is active and target_id == ALAKAZAM and _energy_count(target) < 1:
            value += 30
        if early_setup and has_special_in_hand and energy_id == TELEPATH_ENERGY:
            value += 25
        if target_id in {ALAKAZAM, KADABRA, ABRA} and energy_id == BASIC_PSYCHIC:
            value -= 10 if early_setup and has_special_in_hand else 0
        if target_id == DUDUNSPARCE and energy_id == ENRICHING_ENERGY:
            value += 40
        return -value, index

    ranked = sorted(enumerate(options), key=score)
    return [ranked[0][0]] if ranked else []


def _prize_value(card_id: int | None) -> int:
    if card_id in {723}:
        return 3
    if card_id in RULE_BOX_POKEMON:
        return 2
    return 1


def _choose_switch_option(options: list[dict[str, Any]], current: dict[str, Any]) -> list[int]:
    your_index = _your_index(current)
    your_player = _player(current, your_index)
    active_id = (_active(your_player) or {}).get("id")
    hand_count = int(your_player.get("handCount", len(your_player.get("hand") or [])))
    opponent = _opponent_state(current)
    opponent_active = _active(opponent)
    damage = _estimated_attack_damage(active_id, hand_count)

    def score(item: tuple[int, dict[str, Any]]) -> tuple[int, int, int, int, int]:
        index, option = item
        target = _pokemon_from_option(option, current)
        target_id = (target or {}).get("id")
        owner = int(option.get("playerIndex", your_index))
        if owner != your_index:
            hp = int((target or {}).get("hp", 9999))
            knockout = 0 if hp <= damage and damage > 0 else 1
            active_bonus = 0 if opponent_active and target_id == opponent_active.get("id") else 1
            return knockout, active_bonus, -_prize_value(target_id), hp, index
        return (
            0,
            0,
            -{ALAKAZAM: 100, KADABRA: 70, ABRA: 45, DUNSPARCE: 20}.get(target_id, 10),
            0,
            index,
        )
    ranked = sorted(enumerate(options), key=score)
    return [ranked[0][0]] if ranked else []


def _choose_damage_target(options: list[dict[str, Any]], current: dict[str, Any]) -> list[int]:
    """Prefer a legal opponent target that can be taken as a prize now."""
    your_player = _player(current, _your_index(current))
    active_id = (_active(your_player) or {}).get("id")
    hand_count = int(your_player.get("handCount", len(your_player.get("hand") or [])))
    damage = _estimated_attack_damage(active_id, hand_count)
    your_index = _your_index(current)

    def score(item: tuple[int, dict[str, Any]]) -> tuple[int, int, int, int]:
        index, option = item
        target = _pokemon_from_option(option, current)
        target_id = (target or {}).get("id")
        owner = int(option.get("playerIndex", your_index))
        hp = int((target or {}).get("hp", 9999))
        if owner == your_index:
            return 2, 0, hp, index
        knockout = 0 if damage > 0 and hp <= damage else 1
        return knockout, -_prize_value(target_id), hp, index

    ranked = sorted(enumerate(options), key=score)
    return [ranked[0][0]] if ranked else []


def _estimated_attack_damage(active_id: int | None, hand_count: int) -> int:
    if active_id == ALAKAZAM:
        return hand_count * 20
    return ATTACK_DAMAGE.get(active_id, 0)


def _opponent_active(current: dict[str, Any]) -> dict[str, Any] | None:
    return _active(_opponent_state(current))


def _can_knockout(current: dict[str, Any], player: dict[str, Any]) -> bool:
    target = _opponent_active(current)
    active_id = (_active(player) or {}).get("id")
    if target is None:
        return False
    damage = _estimated_attack_damage(
        active_id,
        int(player.get("handCount", len(player.get("hand") or []))),
    )
    return damage > 0 and int(target.get("hp", 9999)) <= damage


def _has_attack_option(options: list[dict[str, Any]]) -> bool:
    return any(option.get("type") == 13 for option in options)


def _has_attack_line(player: dict[str, Any]) -> bool:
    ids = _card_ids([*_cards(player, "active"), *_bench(player)])
    return bool(set(ids) & {ABRA, KADABRA, ALAKAZAM})


def _main_draw_gain(
    option: dict[str, Any],
    card_id: int | None,
    player: dict[str, Any],
) -> int:
    option_type = option.get("type")
    if option_type == 10:
        return DRAW_ABILITY_GAIN.get(card_id, 0)
    if option_type == 9:
        return {ALAKAZAM: 2, KADABRA: 1}.get(card_id, 0)
    if option_type == 8:
        return 3 if card_id == ENRICHING_ENERGY else -1
    if option_type != 7:
        return 0
    if card_id == DAWN:
        return 2
    if card_id == HILDA:
        return 1
    if card_id == LANAS_AID:
        return 2
    if card_id == RARE_CANDY:
        hand = set(_hand_ids(player))
        field_ids = _card_ids([*_cards(player, "active"), *_bench(player)])
        if ALAKAZAM in hand and any(card in field_ids for card in {ABRA, KADABRA}):
            return 1
        return -1
    if card_id in {POKE_PAD, NIGHT_STRETCHER}:
        return 0
    return 0


def _can_knockout_after_gain(current: dict[str, Any], player: dict[str, Any], gain: int) -> bool:
    target = _opponent_active(current)
    active_id = (_active(player) or {}).get("id")
    if target is None:
        return False
    hand_count = int(player.get("handCount", len(player.get("hand") or [])))
    damage = _estimated_attack_damage(active_id, max(0, hand_count + gain))
    return damage > 0 and int(target.get("hp", 9999)) <= damage


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

    if context in {13, 14, 15}:  # DAMAGE_COUNTER / DAMAGE target
        chosen = _choose_damage_target(options, current)
        _advance_effect(select)
        return chosen[: max(1, min_count)]

    if context == 22:
        # Wondrous Patch first chooses a Basic Psychic Energy, then a
        # Benched Psychic Pokémon. The latter is a field target, not a card
        # from the hand/deck/discard selection list.
        chosen = _choose_energy_option(options, current, player)
        _advance_effect(select)
        return chosen[: max(1, min_count)]

    if context in {1, 2, 5, 6, 7, 8, 9, 10, 21, 26, 29}:
        effect_id = _effect_id(select)
        known = [_option_card_id(option, select, current) for option in options]
        useful = [card_id for card_id in known if card_id is not None]
        if min_count:
            count = min_count
        elif effect_id in {POFFIN, TELEPATH_ENERGY}:
            wanted = {ABRA, DUNSPARCE}
            if effect_id == TELEPATH_ENERGY:
                wanted = {ABRA}
            count = min(max_count, sum(card_id in wanted for card_id in useful))
        elif effect_id == LANAS_AID:
            count = min(max_count, len(useful))
        elif context in {8, 9, 10, 21, 26}:
            count = min(max_count, len(useful))
        else:
            count = min(max_count, 1 if useful else 0)
        chosen = _choose_card_option(options, select, current, player, context, count=count)
        _advance_effect(select)
        return chosen

    if select_type == 4:  # ENERGY
        count = max(1, min_count)
        chosen = _choose_energy_option(options, current, player)[:count]
        _advance_effect(select)
        return chosen

    if context in {3, 4}:  # SWITCH / TO_ACTIVE
        chosen = _choose_switch_option(options, current)
        _advance_effect(select)
        return chosen

    if select_type == 8:  # COUNT
        # Powerful draw/search effects benefit from the largest legal value.
        ranked = sorted(enumerate(options), key=lambda item: -(item[1].get("number", 0)))
        count = min_count if min_count else 1
        chosen = [index for index, _ in ranked[:count]]
        _advance_effect(select)
        return chosen

    # For yes/no effects, accept the effect when it is available. For other
    # required selections, first legal is safer than inventing hidden state.
    if select_type == 9 or all(option.get("type") in {1, 2} for option in options):
        yes = [index for index, option in enumerate(options) if option.get("type") == 1]
        if yes and min_count:
            chosen = yes[:min_count]
            _advance_effect(select)
            return chosen
        if not min_count:
            chosen = yes[:1]
            _advance_effect(select)
            return chosen
    count = min_count if min_count else 0
    chosen = list(range(min(count, max_count, len(options))))
    _advance_effect(select)
    return chosen


def _main_action(obs: dict[str, Any]) -> list[int]:
    select = obs["select"]
    options = select.get("option") or []
    current, player = _your_state(obs)
    hand = _hand_ids(player)
    active = _active(player)
    active_id = (active or {}).get("id")
    bench = _bench(player)
    bench_space = int(player.get("benchMax", 5)) - len(bench)
    hand_count = int(player.get("handCount", len(hand)))
    deck_count = int(player.get("deckCount", 0))
    own_turn = _own_turn_number(current)
    early_setup = own_turn <= 1
    attack_path = _has_usable_attack_path(current, player)
    attack_is_legal = _has_attack_option(options)
    attack_is_ko = attack_is_legal and _can_knockout(current, player)
    attack_line_exists = _has_attack_line(player)
    setup_needed = not attack_line_exists or _needs_attack_line(player)
    ready_alakazam = _has_ready_alakazam(player)
    recovery_need = _recovery_need(player)

    def score(item: tuple[int, dict[str, Any]]) -> tuple[int, int, int]:
        index, option = item
        option_type = option.get("type")
        card_id = _option_card_id(option, select, current)

        if option_type == 13:
            # A knockout is the first-class turn objective. A non-knockout
            # attack still beats optional draw once an attack path is ready.
            if attack_is_ko:
                return (0, 0, index)
            return (
                28 if early_setup and not attack_is_ko else 6 if attack_path else 42,
                -_estimated_attack_damage(active_id, hand_count),
                index,
            )

        if option_type == 10:
            gain = _main_draw_gain(option, card_id, player)
            if gain > 0 and _can_knockout_after_gain(current, player, gain):
                return (1, -gain, index)
            if card_id == DUDUNSPARCE and deck_count <= max(1, gain):
                # Run Away Draw is optional. Avoid spending the last few deck
                # cards unless the draw crosses the attack threshold.
                return (34, 0, index)
            if card_id in DRAW_CARDS or card_id == FEZANDIPITI_EX:
                # After the attack is already secured, draw only prepares the
                # next attacker; it must not displace the current attack.
                if attack_is_legal and attack_path:
                    return (18, -gain, index)
                if card_id == DUDUNSPARCE and ready_alakazam and hand_count >= 10:
                    return (13, -gain, index)
                return (8, -gain, index)
            if card_id == PSYDUCK:
                # Damp is matchup-dependent utility. There is no visible
                # opponent card-text signal in the observation, so keep it
                # behind setup, draw, and a legal attack by default.
                return (75, 0, index)
            return (18, 0, index)

        if option_type == 9:
            evolved_id = card_id
            if evolved_id == ALAKAZAM:
                return (2, 0, index)
            if evolved_id == KADABRA:
                # Rare Candy + Alakazam is scored ahead of this option by the
                # main-action branch below, so a dead-end Kadabra evolution
                # cannot replace an attack this turn.
                return (8 if attack_path else 3, 0, index)
            if evolved_id == DUDUNSPARCE:
                return (11, 0, index)
            return (20, 0, index)

        if option_type == 8:
            energy_id = card_id
            target = _pokemon_from_option(option, current)
            target_id = (target or {}).get("id")
            gain = _main_draw_gain(option, energy_id, player)
            if gain > 0 and _can_knockout_after_gain(current, player, gain):
                return (4, -gain, index)
            if energy_id == TELEPATH_ENERGY:
                if target_id == ALAKAZAM and _energy_count(active) == 0:
                    return (5, 0, index)
                return (9 if target_id in {ALAKAZAM, KADABRA, ABRA} else 16, 0, index)
            if energy_id == BASIC_PSYCHIC:
                return (6 if target_id in {ALAKAZAM, KADABRA, ABRA} else 17, 0, index)
            if energy_id == ENRICHING_ENERGY:
                return (7 if target_id == DUDUNSPARCE else 22, 0, index)
            return (19, 0, index)

        if option_type == 7:
            gain = _main_draw_gain(option, card_id, player)
            if card_id == BOSS_ORDERS:
                # Gust only when it creates a knockout or exposes a strictly
                # better prize. Otherwise preserving the Supporter is part of
                # the plan, even when it is currently playable.
                if _boss_changes_prize_race(current, player, attack_is_legal):
                    return (4, 0, index)
                return (40, 0, index)
            if card_id in {DAWN, HILDA, LANAS_AID} and gain > 0:
                if _can_knockout_after_gain(current, player, gain):
                    return (4, -gain, index)
                return (10 if setup_needed else 16, -gain, index)
            if card_id == RARE_CANDY and ALAKAZAM in hand:
                if active_id in {ABRA, KADABRA} and attack_path:
                    return (1, 0, index)
                return (7, 0, index)
            if card_id == POFFIN:
                return (10 if bench_space > 0 and (early_setup or setup_needed) else 30, 0, index)
            if card_id == POKE_PAD:
                return (14 if setup_needed else 21, 0, index)
            if card_id == BATTLE_CAGE:
                return (16 if early_setup or recovery_need else 28, 0, index)
            if card_id == NIGHT_STRETCHER:
                return (15 if setup_needed or recovery_need else 35, 0, index)
            if card_id == WONDROUS_PATCH:
                patch_ready = BASIC_PSYCHIC in _discard_ids(player) and attack_line_exists
                return (15 if patch_ready else 34, 0, index)
            if card_id == SACRED_ASH:
                return (15 if recovery_need >= 2 else 35, 0, index)
            priorities = {
                ENHANCED_HAMMER: 23,
                ERI: 24,
            }
            return priorities.get(card_id, 25), 0, index
        if option_type == 12:
            # Do not discard attack Energy merely to change positions. Retreat
            # is reserved for a clear handoff to an energized Alakazam.
            return (12 if _retreat_improves_attack(player) else 99, 0, index)
        if option_type == 14:
            return (99, 0, index)
        return (50, 0, index)

    ranked = sorted(enumerate(options), key=score)
    return [ranked[0][0]] if ranked else []


def agent(obs_dict: dict[str, Any]) -> list[int]:
    """Return one legal action for the Alakazam V2 policy."""
    if obs_dict.get("select") is None:
        return read_deck_csv()

    select = obs_dict["select"]
    if int(select.get("type", 0)) == 0 and int(select.get("context", 0)) == 0:
        return _main_action(obs_dict)
    return _select_effect(obs_dict)
