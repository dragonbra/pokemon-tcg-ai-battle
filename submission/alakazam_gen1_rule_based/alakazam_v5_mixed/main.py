"""Alakazam V5: a deterministic Alakazam policy refined from the V5 review.

V5 keeps the V4 legal-option handling while making the direct Abra -> Alakazam
line explicit, treating Mist Energy as a real damage blocker in this runtime,
and using Xerosic's Machinations to pressure opposing Alakazam decks.
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

XEROSIC = 1197
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

# Every member of the Abra line only needs one Psychic Energy to attack. Do
# not spend a second Energy on a member of that line when another legal target
# can use it.
ENERGY_CAPPED_POKEMON = ATTACK_LINE

# Dunsparce's first attack is the zero-damage Trading Places handoff. The
# runtime exposes the attack id in the main-action option.
TRADING_PLACES_ATTACK = 423

ATTACK_DAMAGE = {
    ABRA: 10,
    DUNSPARCE: 20,
    DUDUNSPARCE: 90,
    KADABRA: 30,
}

ATTACK_ENERGY_COUNT = {
    ABRA: 1,
    DUNSPARCE: 1,
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
BASIC_ENERGY_IDS = set(range(1, 10))
MIST_ENERGY = 11
HAND_DRAW_STOP = 20
DECK_DRAW_STOP = 10
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


def _field_count(player: dict[str, Any], card_ids: set[int]) -> int:
    """Count matching Pokémon in Active plus Bench, regardless of position."""
    return sum(card.get("id") in card_ids for card in _field_pokemon(player))


def _attack_line_count(player: dict[str, Any]) -> int:
    return _field_count(player, ATTACK_LINE)


def _has_any_attack_line(player: dict[str, Any]) -> bool:
    return _attack_line_count(player) > 0


def _attack_line_target_reached(player: dict[str, Any]) -> bool:
    return _attack_line_count(player) >= 3


def _prize_count(player: dict[str, Any]) -> int:
    """Return the number of prizes still in our prize cards when visible."""
    prizes = player.get("prize")
    if isinstance(prizes, list):
        return len(prizes)
    return int(player.get("prizeCount", 6))


def _safe_deck_floor(player: dict[str, Any]) -> int:
    """Keep both V5's hard floor and the visible prize-card reserve."""
    return max(DECK_DRAW_STOP, _prize_count(player) + 1)


def _attack_continuity(current: dict[str, Any], player: dict[str, Any]) -> dict[str, int | bool]:
    """Summarize the visible current/next/future Alakazam attack chain.

    A field count is still useful for deciding how much to bench, but it is not
    enough to distinguish three unenergized Abras from a ready Alakazam plus a
    prepared replacement.  This state intentionally uses only visible cards;
    it is a priority hint, not hidden-information search.
    """
    active = _active(player)
    field_line = _field_pokemon(player)
    ready_active = _has_attack_ready_pokemon(player)
    ready_bench = sum(
        pokemon.get("id") in ATTACK_LINE and _has_psychic_energy(pokemon)
        for pokemon in _bench(player)
    )
    field_count = sum(pokemon.get("id") in ATTACK_LINE for pokemon in field_line)
    hand_count = sum(card_id in ATTACK_LINE for card_id in _hand_ids(player))
    unenergized = sum(
        pokemon.get("id") in ATTACK_LINE and not _has_psychic_energy(pokemon)
        for pokemon in field_line
    )
    direct_path = bool(
        active
        and _active_evolution_is_legal(current, player)
        and active.get("id") in {ABRA, KADABRA}
        and ALAKAZAM in _hand_ids(player)
        and (_has_psychic_energy(active) or _psychic_energy_in_hand(player))
        and (active.get("id") == KADABRA or RARE_CANDY in _hand_ids(player))
    )
    natural_path = bool(
        active
        and _active_evolution_is_legal(current, player)
        and active.get("id") == KADABRA
        and ALAKAZAM in _hand_ids(player)
        and _has_psychic_energy(active)
    )
    recovery_count = _discard_attack_line_count(player)
    next_attack_path = bool(
        ready_active
        or ready_bench
        or direct_path
        or natural_path
        or recovery_count > 0
    )
    dunsparce_count = _field_count(player, {DUNSPARCE})
    dudunsparce_count = _field_count(player, {DUDUNSPARCE})
    can_run_away_draw = any(
        pokemon.get("id") == DUDUNSPARCE
        and not pokemon.get("abilityUsed", False)
        for pokemon in field_line
    )
    return {
        "field_count": field_count,
        "future_count": field_count + hand_count,
        "ready_active": ready_active,
        "ready_bench": ready_bench,
        "next_attack_path": next_attack_path,
        "unenergized_count": unenergized,
        "recovery_count": recovery_count,
        "dunsparce_count": dunsparce_count,
        "dudunsparce_count": dudunsparce_count,
        "can_run_away_draw": can_run_away_draw,
    }


def _discard_attack_line_count(player: dict[str, Any]) -> int:
    return sum(card in ATTACK_LINE for card in _discard_ids(player))


def _has_attack_recovery(player: dict[str, Any]) -> bool:
    return _discard_attack_line_count(player) > 0


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


def _active_evolution_is_legal(current: dict[str, Any], player: dict[str, Any]) -> bool:
    """Whether the Active can evolve this turn under the runtime's rules."""
    active = _active(player)
    return bool(
        active
        and _own_turn_number(current) > 1
        and not active.get("appearThisTurn", False)
    )


def _active_direct_alakazam_route(current: dict[str, Any], player: dict[str, Any]) -> bool:
    """Whether the Active has a legal immediate Abra/Kadabra -> Alakazam line."""
    return _active_evolution_is_legal(current, player) and _active_can_become_alakazam_now(
        player
    )


def _active_hilda_rare_candy_route(current: dict[str, Any], player: dict[str, Any]) -> bool:
    """Whether Hilda can set up a legal Rare Candy attack this turn."""
    return _active_evolution_is_legal(current, player) and _active_can_use_hilda_rare_candy(
        player
    )


def _active_hilda_target_route(current: dict[str, Any], player: dict[str, Any]) -> bool:
    """Same route after Hilda has left hand and its search is in progress."""
    active = _active(player)
    hand_ids = set(_hand_ids(player))
    return bool(
        _active_evolution_is_legal(current, player)
        and active
        and active.get("id") == ABRA
        and RARE_CANDY in hand_ids
        and (_has_psychic_energy(active) or _psychic_energy_in_hand(player))
    )


def _active_natural_alakazam_route(current: dict[str, Any], player: dict[str, Any]) -> bool:
    """Whether a non-Rare-Candy Active Kadabra can evolve and attack now."""
    active = _active(player)
    return bool(
        _active_evolution_is_legal(current, player)
        and active
        and active.get("id") == KADABRA
        and ALAKAZAM in _hand_ids(player)
        and _has_psychic_energy(active)
    )


def _has_evolvable_abra(current: dict[str, Any], player: dict[str, Any]) -> bool:
    """Whether a field Abra can become Kadabra during this turn."""
    return bool(
        _own_turn_number(current) > 1
        and any(
            pokemon.get("id") == ABRA and not pokemon.get("appearThisTurn", False)
            for pokemon in _field_pokemon(player)
        )
    )


def _has_unenergized_attack_line(player: dict[str, Any]) -> bool:
    """Whether any Abra-line Pokémon still needs its first Psychic Energy."""
    return any(
        pokemon.get("id") in ATTACK_LINE and not _has_psychic_energy(pokemon)
        for pokemon in _field_pokemon(player)
    )


def _can_use_enriching_for_draw(player: dict[str, Any]) -> bool:
    """Whether Enriching Energy can safely convert Dudunsparce into extra draw."""
    active = _active(player)
    continuity = _attack_continuity({}, player)
    return bool(
        active
        and active.get("id") == ALAKAZAM
        and _has_psychic_energy(active)
        and not continuity["unenergized_count"]
        and continuity["ready_active"]
        and continuity["dudunsparce_count"]
        and continuity["can_run_away_draw"]
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
    damage = _attack_damage_against(
        (active or {}).get("id"),
        int(player.get("handCount", len(player.get("hand") or []))),
        _opponent_active(current),
    )
    opponent = _opponent_state(current)
    target = _opponent_active(current)
    if not target:
        return False
    active_ko = damage > 0 and int(target.get("hp", 9999)) <= damage
    active_prize = _prize_value(target.get("id"))
    return not active_ko and any(
        int(pokemon.get("hp", 9999))
        <= _attack_damage_against(
            (active or {}).get("id"),
            int(player.get("handCount", len(player.get("hand") or []))),
            pokemon,
        )
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


def _attack_security(player: dict[str, Any]) -> bool:
    """Whether this turn and the next attack have a visible attacker."""
    return (
        _has_attack_ready_pokemon(player)
        or _has_ready_alakazam(player)
        or _has_alakazam_attack_path(player)
    )


def _previous_turn_had_knockout(
    current: dict[str, Any], logs: list[dict[str, Any]] | None = None
) -> bool:
    """Detect our Pokémon being Knocked Out during the opponent's last turn."""
    your_index = _your_index(current)
    for log in reversed(logs or current.get("logs") or []):
        if (
            log.get("type") == 6
            and log.get("playerIndex") == your_index
            and log.get("fromArea") in {4, 5}
            and log.get("toArea") == 3
        ):
            return True
    return False


def _fezandipiti_needed(
    current: dict[str, Any],
    player: dict[str, Any],
    logs: list[dict[str, Any]] | None = None,
) -> bool:
    """Whether Flip the Script supplies a necessary source of progress."""
    if not _previous_turn_had_knockout(current, logs):
        return False
    if _can_knockout(current, player):
        return False
    if _active_hilda_rare_candy_route(current, player):
        return False
    if _active_direct_alakazam_route(current, player):
        return False
    active = _active(player)
    if active and active.get("id") == ALAKAZAM and _psychic_energy_in_hand(player):
        return False
    if _draw_changes_knockout(current, player, DRAW_ABILITY_GAIN[FEZANDIPITI_EX]):
        return True
    hand_count = int(player.get("handCount", len(player.get("hand") or [])))
    # A legal Kadabra attack for 30 damage is only a fallback. After a KO,
    # Flip the Script is worth trying while the hand is still small unless an
    # Alakazam attack is already secured.
    return hand_count <= 8


def _fezandipiti_play_is_safe(
    current: dict[str, Any], player: dict[str, Any], logs: list[dict[str, Any]] | None = None
) -> bool:
    """Avoid exposing a two-Prize Fez when an ordinary Bench already exists."""
    if not _bench(player):
        return True
    opponent = _opponent_state(current)
    if len(opponent.get("prize") or []) <= 2:
        return False
    return _fezandipiti_needed(current, player, logs)


def _telepath_requires_abra_first(player: dict[str, Any]) -> bool:
    """Whether playing a hand Abra must precede a Telepath attachment."""
    return (
        ABRA in _hand_ids(player)
        and not _has_field_card(player, ATTACK_LINE)
        and len(_bench(player)) < int(player.get("benchMax", 5))
    )


def _trading_places_improves_attack(player: dict[str, Any]) -> bool:
    """Whether Dunsparce can put a prepared Alakazam Active for an attack."""
    active = _active(player)
    return bool(
        active
        and active.get("id") == DUNSPARCE
        and any(
            pokemon.get("id") == ALAKAZAM and _has_psychic_energy(pokemon)
            for pokemon in _bench(player)
        )
    )


def _recovery_need(player: dict[str, Any]) -> int:
    """Score how much the discard pile can rebuild the next attack line."""
    discard = set(_discard_ids(player))
    return len(discard & {ABRA, KADABRA, ALAKAZAM, DUNSPARCE, DUDUNSPARCE}) + len(
        discard & {BASIC_PSYCHIC}
    )


def _recovery_card_value(card_id: int | None, player: dict[str, Any]) -> int:
    """Rank a recovered card by the shortest visible route to the next attack."""
    if card_id not in ATTACK_LINE:
        return 20
    field_ids = _field_ids(player)
    hand_ids = set(_hand_ids(player))
    active_id = (_active(player) or {}).get("id")

    # Abra is a hard prerequisite for a Rare Candy route.  Do not take an
    # already attractive Alakazam while leaving the board with no base.
    if not field_ids & ATTACK_LINE and card_id == ABRA:
        return 145
    if active_id == KADABRA and ALAKAZAM not in hand_ids and card_id == ALAKAZAM:
        return 140
    if (
        field_ids & {ABRA}
        and KADABRA not in hand_ids
        and ALAKAZAM in hand_ids
        and card_id == KADABRA
    ):
        return 138
    if card_id == ALAKAZAM:
        return 130
    if card_id == KADABRA:
        return 120
    return 105


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


def _attached_energy_ids(pokemon: dict[str, Any] | None) -> list[int]:
    return _card_ids((pokemon or {}).get("energyCards") or [])


def _special_energy_ids(pokemon: dict[str, Any] | None) -> list[int]:
    """Return attached special-energy IDs visible in the observation."""
    return [
        card_id
        for card_id in _attached_energy_ids(pokemon)
        if card_id not in BASIC_ENERGY_IDS
    ]


def _has_special_energy(pokemon: dict[str, Any] | None) -> bool:
    return bool(_special_energy_ids(pokemon))


def _has_mist_energy(pokemon: dict[str, Any] | None) -> bool:
    return MIST_ENERGY in _attached_energy_ids(pokemon)


def _attack_damage_against(
    active_id: int | None, hand_count: int, target: dict[str, Any] | None
) -> int:
    """Estimate damage using the official runtime's Mist interaction.

    In this engine Alakazam's Powerful Hand is implemented as DamageCounter,
    so Mist Energy prevents it even though the printed card text distinguishes
    damage from effects. The replay evidence makes this a strategy-level rule.
    """
    if active_id == ALAKAZAM and _has_mist_energy(target):
        return 0
    return _estimated_attack_damage(active_id, hand_count)


def _hammer_changes_attack(current: dict[str, Any], player: dict[str, Any]) -> bool:
    """Whether discarding one Hammer can turn the current attack effective."""
    target = _opponent_active(current)
    active = _active(player)
    if not target or not active or not _has_special_energy(target):
        return False
    if active.get("id") != ALAKAZAM:
        return False
    hand_count = int(player.get("handCount", len(player.get("hand") or [])))
    # Playing Enhanced Hammer costs the Supporter/action card from hand.  The
    # only runtime blocker relevant to Powerful Hand is Mist Energy; after it
    # is removed the remaining hand still has to reach the target HP.
    if not _has_mist_energy(target):
        return False
    return (hand_count - 1) * 20 >= int(target.get("hp", 9999))


def _psychic_energy_in_hand(player: dict[str, Any]) -> bool:
    return bool(set(_hand_ids(player)) & {TELEPATH_ENERGY, BASIC_PSYCHIC})


def _active_can_become_alakazam_now(player: dict[str, Any]) -> bool:
    """Whether the current hand already contains a direct Alakazam line."""
    active = _active(player)
    if not active or active.get("id") not in {ABRA, KADABRA}:
        return False
    hand_ids = set(_hand_ids(player))
    if ALAKAZAM not in hand_ids:
        return False
    if not (_has_psychic_energy(active) or _psychic_energy_in_hand(player)):
        return False
    if active.get("id") == ABRA and RARE_CANDY not in hand_ids:
        return False
    return True


def _active_can_use_hilda_rare_candy(player: dict[str, Any]) -> bool:
    """Whether Hilda can fetch Alakazam for a Rare Candy attack this turn."""
    active = _active(player)
    hand_ids = set(_hand_ids(player))
    if not active or active.get("id") != ABRA:
        return False
    return (
        HILDA in hand_ids
        and RARE_CANDY in hand_ids
        and (_has_psychic_energy(active) or _psychic_energy_in_hand(player))
    )


def _has_bench_unenergized_alakazam(player: dict[str, Any]) -> bool:
    return any(
        pokemon.get("id") == ALAKAZAM and not _has_psychic_energy(pokemon)
        for pokemon in _bench(player)
    )


def _opponent_is_alakazam_deck(current: dict[str, Any]) -> bool:
    opponent = _opponent_state(current)
    return bool(_field_ids(opponent) & ATTACK_LINE)


def _xerosic_changes_opponent_attack(current: dict[str, Any], player: dict[str, Any]) -> bool:
    """Whether Xerosic can visibly change the opponent's next attack.

    Xerosic leaves the opponent with three cards.  We only have hand size, not
    card identity, so the high-confidence case is an opposing Active
    Alakazam whose current Powerful Hand is lethal against our Active but its
    three-card version is not.  A mirror with a large hand is a weaker but
    still useful fallback when we have an Alakazam exchange to protect.
    """
    opponent = _opponent_state(current)
    opponent_active = _active(opponent)
    if _own_turn_number(current) < 2 or not opponent_active:
        return False
    hand_count = int(opponent.get("handCount", len(opponent.get("hand") or [])))
    if hand_count < 4 or opponent_active.get("id") != ALAKAZAM:
        return False
    our_active = _active(player)
    if not our_active:
        return False
    target_hp = int(our_active.get("hp", 9999))
    current_damage = hand_count * 20
    reduced_damage = 3 * 20
    if current_damage >= target_hp > reduced_damage:
        return True
    return bool(
        hand_count >= 8
        and our_active.get("id") == ALAKAZAM
        and _has_psychic_energy(our_active)
    )


def _xerosic_is_worth_playing(current: dict[str, Any], player: dict[str, Any]) -> bool:
    opponent = _opponent_state(current)
    hand_count = int(opponent.get("handCount", 0))
    if hand_count < 4 or _own_turn_number(current) < 2:
        return False
    return _xerosic_changes_opponent_attack(current, player)


def _hand_index_score(card_id: int, hand: list[int], player: dict[str, Any]) -> int:
    """Return a discard score; higher means a more acceptable discard."""
    counts = {card: hand.count(card) for card in set(hand)}
    if card_id in {ENRICHING_ENERGY, TELEPATH_ENERGY, BASIC_PSYCHIC}:
        return 3 if counts.get(card_id, 0) > 1 else 0
    if card_id in {ENHANCED_HAMMER, SACRED_ASH, WONDROUS_PATCH, BATTLE_CAGE}:
        return 3
    if card_id in {POFFIN, POKE_PAD, NIGHT_STRETCHER, LANAS_AID}:
        return 2 if counts.get(card_id, 0) > 1 else 1
    if card_id in EVOLUTION:
        return 1 if counts.get(card_id, 0) > 1 else 0
    if card_id in {
        RARE_CANDY,
        HILDA,
        DAWN,
        BOSS_ORDERS,
        XEROSIC,
        ABRA,
        KADABRA,
        ALAKAZAM,
        DUNSPARCE,
        DUDUNSPARCE,
    }:
        return 0
    if card_id in {FEZANDIPITI_EX, SHAYMIN, PSYDUCK}:
        return 2
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
    known_ids = [_option_card_id(option, select, current) for option in options]

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
            if effect_step == 0:
                if _active_hilda_rare_candy_route(current, player) or (
                    effect_id == HILDA and _active_hilda_target_route(current, player)
                ):
                    wanted = {ALAKAZAM}
                elif (_active(player) or {}).get("id") == ABRA:
                    wanted = {KADABRA}
                elif _attack_line_target_reached(player) and _attack_security(player):
                    wanted = {DUDUNSPARCE}
                else:
                    wanted = EVOLUTION
            else:
                wanted = (
                    {ENRICHING_ENERGY}
                    if _attack_line_target_reached(player) and _attack_security(player)
                    else {TELEPATH_ENERGY, BASIC_PSYCHIC, ENRICHING_ENERGY}
                )
            if any(_option_card_id(option, select, current) in wanted for option in options):
                return 0 if card_id in wanted else 4
        if effect_id == POFFIN:
            # Buddy-Buddy Poffin is limited to Basic Pokémon with 70 HP or
            # less. Complete the three-Pokémon attack line before expanding
            # the Dunsparce engine.
            continuity = _attack_continuity(current, player)
            wanted = {ABRA} if continuity["future_count"] < 3 else {DUNSPARCE}
            return 0 if card_id in wanted else 4
        if effect_id == TELEPATH_ENERGY:
            return 0 if card_id == ABRA else 4
        if effect_id in {NIGHT_STRETCHER, LANAS_AID}:
            if card_id == ABRA and not _has_field_card(player, {ABRA}):
                return -1
            return 0 if card_id in ATTACK_LINE else 1 if card_id == BASIC_PSYCHIC else 4
        if effect_id == POKE_PAD:
            active = _active(player)
            if _active_direct_alakazam_route(current, player) or _active_hilda_rare_candy_route(
                current, player
            ):
                wanted = {ALAKAZAM}
            elif active and active.get("id") == KADABRA:
                wanted = {ALAKAZAM}
            elif _own_turn_number(current) <= 1:
                # Nothing can evolve on the first turn. Search a setup Basic
                # instead of putting a dead Kadabra into hand.
                wanted = {DUNSPARCE} if DUNSPARCE in known_ids else {ABRA}
            elif active and active.get("id") == ABRA:
                wanted = {KADABRA}
            elif any(option_id == KADABRA for option_id in known_ids):
                wanted = {KADABRA}
            elif bench_count < int(player.get("benchMax", 5)):
                wanted = {DUNSPARCE}
            else:
                wanted = {ABRA, DUNSPARCE}
            if any(option_id in wanted for option_id in known_ids):
                return 0 if card_id in wanted else 4
        if effect_id == XEROSIC and context in {8, 29}:
            return 0
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
            if effect_id in {NIGHT_STRETCHER, LANAS_AID} and card_id in ATTACK_LINE:
                value = _recovery_card_value(card_id, player)
            else:
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
        elif context in {26, 30}:  # Enhanced Hammer energy selection
            value = {
                MIST_ENERGY: 200,
                TELEPATH_ENERGY: 100,
                ENRICHING_ENERGY: 90,
            }.get(card_id, 10)
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
    """Attach one useful Energy without overloading a single attacker."""
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
        owner = int(option.get("playerIndex", _your_index(current)))
        area = option.get("inPlayArea", option.get("area"))

        # Enhanced Hammer uses SelectType.ENERGY (context 30 in the current
        # API), not the attached-card context used by older policy comments.
        # The active Pokémon is the default target, with Mist taking priority
        # when it is blocking Powerful Hand.
        if owner != _your_index(current) and energy_id not in BASIC_ENERGY_IDS:
            if area == 4:
                if energy_id == MIST_ENERGY and (_active(player) or {}).get("id") == ALAKAZAM:
                    return -320, index
                return -260, index
            return -180 if energy_id == MIST_ENERGY else -140, index

        # The simulator should normally omit these options; a high score keeps
        # them as last-resort legal fallbacks if it does not.
        if target_id in ENERGY_CAPPED_POKEMON and target_energy_count >= 1:
            return 90, index

        if energy_id == TELEPATH_ENERGY:
            # Telepath's search only works on a Psychic Pokémon. Prefer an
            # unenergized Abra-line target, and never use a second attachment
            # merely to obtain a search that the first attachment already
            # made possible.
            if target_id in ATTACK_LINE:
                if _has_psychic_energy(target):
                    return 80, index
                return (0 if area == 4 else 1), index
            if target_id in {DUNSPARCE, DUDUNSPARCE}:
                return 35, index
            return 30, index

        if target_id in ATTACK_LINE and not _has_psychic_energy(target):
            value = 0 if area == 4 else 1
            if target is active:
                value -= 1
            return value, index

        if target is active and target_id in {FEZANDIPITI_EX, SHAYMIN}:
            if any(
                pokemon.get("id") == ALAKAZAM and _has_psychic_energy(pokemon)
                for pokemon in _bench(player)
            ):
                # One attached Energy pays the one-card retreat cost and
                # hands the turn to the prepared Alakazam.
                return -2, index

        value = {
            ALAKAZAM: 100,
            KADABRA: 80,
            ABRA: 65,
            DUNSPARCE: 20,
            DUDUNSPARCE: 25,
        }.get(target_id, 10)
        if target is active and target_id == ALAKAZAM and _energy_count(target) < 1:
            value += 30
        if target_id in {ALAKAZAM, KADABRA, ABRA} and energy_id == BASIC_PSYCHIC:
            value -= 10 if early_setup and has_special_in_hand else 0
        if target_id == DUDUNSPARCE and energy_id == ENRICHING_ENERGY:
            value = -5 if _can_use_enriching_for_draw(player) else 25
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

    def score(item: tuple[int, dict[str, Any]]) -> tuple[int, int, int, int, int]:
        index, option = item
        target = _pokemon_from_option(option, current)
        target_id = (target or {}).get("id")
        owner = int(option.get("playerIndex", your_index))
        if owner != your_index:
            hp = int((target or {}).get("hp", 9999))
            target_damage = _attack_damage_against(active_id, hand_count, target)
            knockout = 0 if hp <= target_damage and target_damage > 0 else 1
            active_bonus = 0 if option.get("area") == 4 else 1
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
    your_index = _your_index(current)

    def score(item: tuple[int, dict[str, Any]]) -> tuple[int, int, int, int]:
        index, option = item
        target = _pokemon_from_option(option, current)
        target_id = (target or {}).get("id")
        owner = int(option.get("playerIndex", your_index))
        hp = int((target or {}).get("hp", 9999))
        if owner == your_index:
            return 2, 0, hp, index
        damage = _attack_damage_against(active_id, hand_count, target)
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
    damage = _attack_damage_against(
        active_id,
        int(player.get("handCount", len(player.get("hand") or []))),
        target,
    )
    return damage > 0 and int(target.get("hp", 9999)) <= damage


def _has_attack_option(options: list[dict[str, Any]]) -> bool:
    return any(option.get("type") == 13 for option in options)


def _has_attack_line(player: dict[str, Any]) -> bool:
    """Whether the three-Pokémon Abra attack-line setup target is reached."""
    return _attack_line_target_reached(player)


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
    damage = _attack_damage_against(active_id, max(0, hand_count + gain), target)
    return damage > 0 and int(target.get("hp", 9999)) <= damage


def _draw_changes_knockout(current: dict[str, Any], player: dict[str, Any], gain: int) -> bool:
    """Return true only when the draw creates a knockout that was not present."""
    return (
        gain > 0
        and not _can_knockout(current, player)
        and _can_knockout_after_gain(current, player, gain)
    )


def _draw_is_blocked(
    current: dict[str, Any], player: dict[str, Any], gain: int = 0
) -> bool:
    """Apply hand/deck/prize guards unless this draw directly creates lethal.

    The 20-card hand and 10-card deck limits remain V5's hard guards.  The
    visible prize reserve adds context above those limits: drawing three from
    a deck of eleven is not safe when it leaves fewer cards than the remaining
    prizes.  The only first-class exception is a draw that creates this
    turn's knockout, which is the same P1 gate used by the public agents.
    """
    if _draw_changes_knockout(current, player, gain):
        return False
    hand_count = int(player.get("handCount", len(player.get("hand") or [])))
    deck_count = int(player.get("deckCount", 0))
    if _can_knockout(current, player):
        return True
    if hand_count > HAND_DRAW_STOP or deck_count <= DECK_DRAW_STOP:
        return True
    if gain > 0 and deck_count - gain < _safe_deck_floor(player):
        return True
    return False


def _choose_evolution_target(
    options: list[dict[str, Any]], current: dict[str, Any], player: dict[str, Any]
) -> list[int]:
    """Choose a legal Rare Candy target while preserving a direct attack route."""
    hand_ids = set(_hand_ids(player))
    active = _active(player)

    def score(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        index, option = item
        target = _pokemon_from_option(option, current)
        target_id = (target or {}).get("id")
        area = option.get("inPlayArea", option.get("area"))
        if target_id == ABRA and area == 4 and ALAKAZAM in hand_ids:
            return 0, index
        if target_id == ABRA:
            return 1, index
        if active and target is active:
            return 2, index
        return 3, index

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

    if context == 37:  # Rare Candy chooses the Pokémon to evolve.
        chosen = _choose_evolution_target(options, current, player)
        _advance_effect(select)
        return chosen[: max(1, min_count)]

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
    direct_hand_route = _active_direct_alakazam_route(current, player)
    direct_hilda_route = _active_hilda_rare_candy_route(current, player)
    natural_alakazam_route = _active_natural_alakazam_route(current, player)
    attack_path = _has_usable_attack_path(current, player) or direct_hilda_route
    attack_is_legal = _has_attack_option(options)
    attack_is_ko = attack_is_legal and _can_knockout(current, player)
    opponent_active = _opponent_active(current)
    mist_blocks_attack = (
        active_id == ALAKAZAM
        and _has_mist_energy(opponent_active)
        and attack_is_legal
    )
    attack_line_count = _attack_line_count(player)
    attack_line_exists = _has_any_attack_line(player)
    continuity = _attack_continuity(current, player)
    dunsparce_engine_ready = bool(continuity["can_run_away_draw"])
    attack_line_target_reached = attack_line_count >= 3
    # Keep the field-count objective for early benching, but continue setup if
    # the visible line has no ready attacker or no next-turn handoff.
    setup_needed = bool(
        attack_line_count < 3
        or not continuity["next_attack_path"]
        or continuity["unenergized_count"] > 1
    )
    ready_alakazam = _has_ready_alakazam(player)
    recovery_need = _recovery_need(player)
    attack_recovery_available = _has_attack_recovery(player)
    telepath_requires_abra = _telepath_requires_abra_first(player)
    fezandipiti_needed = _fezandipiti_play_is_safe(current, player, obs.get("logs") or [])

    def score(item: tuple[int, dict[str, Any]]) -> tuple[int, int, int]:
        index, option = item
        option_type = option.get("type")
        card_id = _option_card_id(option, select, current)

        if option_type == 13:
            # A knockout is the first-class turn objective. A non-knockout
            # attack yields to a useful setup/draw action while the deck is
            # safe, but remains a fallback once draw is guarded.
            if attack_is_ko:
                if (
                    active_id == DUNSPARCE
                    and option.get("attackId") == TRADING_PLACES_ATTACK
                    and _trading_places_improves_attack(player)
                ):
                    return (4, 0, index)
                return (0, 0, index)
            if (
                active_id == DUNSPARCE
                and option.get("attackId") == TRADING_PLACES_ATTACK
                and _trading_places_improves_attack(player)
            ):
                # Trading Places is a deliberate handoff to the prepared
                # Alakazam, not an ordinary low-damage attack.
                return (4, 0, index)
            if mist_blocks_attack:
                # If no Hammer or useful Boss target exists, ending is safer
                # than repeatedly putting zero damage into Mist Energy.
                return (110, 0, index)
            if _draw_is_blocked(current, player):
                return (6, -_estimated_attack_damage(active_id, hand_count), index)
            return (
                30 if not attack_line_target_reached else 24 if attack_path else 42,
                -_estimated_attack_damage(active_id, hand_count),
                index,
            )

        if option_type == 10:
            gain = _main_draw_gain(option, card_id, player)
            if attack_is_ko:
                # P0: an existing valid knockout is the end of the turn's
                # objective.  Do not spend an ability on speculative setup.
                return (40, 0, index)
            if _draw_changes_knockout(current, player, gain):
                return (1, -gain, index)
            if card_id == DUNSPARCE:
                return (4 if _trading_places_improves_attack(player) else 24, 0, index)
            if card_id == FEZANDIPITI_EX and not fezandipiti_needed:
                return (130, 0, index)
            if card_id == DUDUNSPARCE and deck_count <= max(1, gain):
                # Run Away Draw is optional. Avoid spending the last few deck
                # cards unless the draw crosses the attack threshold.
                return (34, 0, index)
            if card_id == DUDUNSPARCE and not dunsparce_engine_ready:
                return (130, 0, index)
            if card_id in DRAW_CARDS or card_id == FEZANDIPITI_EX:
                if _draw_is_blocked(current, player, gain):
                    return (130, 0, index)
                # After the attack is already secured, draw only prepares the
                # next attacker; it must not displace the current attack.
                if attack_is_legal and attack_path:
                    return (18, -gain, index)
                if not continuity["next_attack_path"]:
                    return (11 if setup_needed else 24, -gain, index)
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
            if attack_is_ko:
                return (40, 0, index)
            evolved_id = card_id
            target = _pokemon_from_option(option, current)
            target_id = (target or {}).get("id")
            target_area = option.get("inPlayArea", option.get("area"))
            if evolved_id == ALAKAZAM:
                if natural_alakazam_route and target_area == 4:
                    return (0, 0, index)
                return (1 if target_area == 4 else 3, 0, index)
            if evolved_id == KADABRA:
                # If the Active Abra still has a Rare Candy + Alakazam route,
                # evolve a Benched Abra into Kadabra instead. This preserves
                # the direct attack route on the Active Pokémon.
                if target_id == ABRA and target_area == 4 and (
                    direct_hand_route or direct_hilda_route
                ):
                    return (18, 0, index)
                if target_id == ABRA and target_area == 5 and (
                    direct_hand_route or direct_hilda_route
                ):
                    return (2, 0, index)
                if target_id == ABRA:
                    return (3 if target_area == 4 else 5, 0, index)
                return (8 if attack_path else 4, 0, index)
            if evolved_id == DUDUNSPARCE:
                return (7 if attack_line_target_reached else 13, 0, index)
            return (20, 0, index)

        if option_type == 8:
            if attack_is_ko:
                return (40, 0, index)
            energy_id = card_id
            target = _pokemon_from_option(option, current)
            target_id = (target or {}).get("id")
            target_area = option.get("inPlayArea", option.get("area"))
            gain = _main_draw_gain(option, energy_id, player)
            if _draw_changes_knockout(current, player, gain):
                return (4, -gain, index)
            if energy_id == TELEPATH_ENERGY:
                if telepath_requires_abra:
                    return (40, 0, index)
                if target_id in ATTACK_LINE:
                    if _has_psychic_energy(target):
                        return (28, 0, index)
                    return (0 if target_area == 4 else 1, 0, index)
                return (18, 0, index)
            if energy_id == BASIC_PSYCHIC:
                if target_id in ATTACK_LINE:
                    if _has_psychic_energy(target):
                        return (28, 0, index)
                    return (0 if target_area == 4 else 1, 0, index)
                if active_id in {FEZANDIPITI_EX, SHAYMIN} and target is active and any(
                    pokemon.get("id") == ALAKAZAM and _has_psychic_energy(pokemon)
                    for pokemon in bench
                ):
                    return (2, 0, index)
                return (17, 0, index)
            if energy_id == ENRICHING_ENERGY:
                if target_id == DUDUNSPARCE and _can_use_enriching_for_draw(player):
                    return (5, 0, index)
                return (22, 0, index)
            return (19, 0, index)

        if option_type == 7:
            if attack_is_ko:
                return (40, 0, index)
            gain = _main_draw_gain(option, card_id, player)
            if card_id == ABRA and telepath_requires_abra:
                return (0, 0, index)
            if card_id == FEZANDIPITI_EX:
                return (0 if fezandipiti_needed else 130, 0, index)
            if card_id == XEROSIC:
                if _xerosic_is_worth_playing(current, player) and not attack_is_ko:
                    return (6, 0, index)
                return (120, 0, index)
            if card_id == BOSS_ORDERS:
                # Gust only when it creates a knockout or exposes a strictly
                # better prize. Otherwise preserving the Supporter is part of
                # the plan, even when it is currently playable.
                if _boss_changes_prize_race(current, player, attack_is_legal):
                    return (4, 0, index)
                if mist_blocks_attack:
                    return (130, 0, index)
                return (40, 0, index)
            if card_id in {DAWN, HILDA, LANAS_AID} and gain > 0:
                if _draw_changes_knockout(current, player, gain):
                    return (4, -gain, index)
                if card_id == DAWN and _has_evolvable_abra(current, player) and not (
                    direct_hand_route or direct_hilda_route
                ):
                    # Dawn can fetch Kadabra and immediately let the Bench
                    # Abra use Psychic Draw; it is more valuable here than a
                    # blind Poké Pad search.
                    return (4, 0, index)
                if card_id == HILDA:
                    if direct_hilda_route:
                        return (1, 0, index)
                    if attack_line_target_reached and _attack_security(player):
                        return (8, 0, index)
                    return (10 if setup_needed else 12, 0, index)
                if card_id == LANAS_AID:
                    return (7 if attack_recovery_available and not attack_is_ko else 18, 0, index)
                return (9 if setup_needed else 16, -gain, index)
            if card_id == RARE_CANDY and ALAKAZAM in hand:
                if natural_alakazam_route:
                    return (8, 0, index)
                if active_id in {ABRA, KADABRA} and attack_path:
                    return (1, 0, index)
                return (7, 0, index)
            if card_id == POFFIN:
                return (
                    10 if bench_space > 0 and continuity["future_count"] < 3 else 12
                    if bench_space > 0
                    else 30,
                    0,
                    index,
                )
            if card_id == POKE_PAD:
                if direct_hilda_route:
                    return (9, 0, index)
                if active_id == ABRA and KADABRA not in hand:
                    return (8, 0, index)
                if active_id == KADABRA and ALAKAZAM not in hand:
                    return (7, 0, index)
                return (14 if setup_needed else 21, 0, index)
            if card_id == BATTLE_CAGE:
                return (16 if early_setup or recovery_need else 28, 0, index)
            if card_id == NIGHT_STRETCHER:
                return (
                    6 if attack_recovery_available and not attack_is_ko else 14
                    if setup_needed
                    else 30,
                    0,
                    index,
                )
            if card_id == WONDROUS_PATCH:
                patch_ready = BASIC_PSYCHIC in _discard_ids(player) and attack_line_exists
                return (15 if patch_ready else 34, 0, index)
            if card_id == SACRED_ASH:
                return (15 if recovery_need >= 2 else 35, 0, index)
            priorities = {
                ENHANCED_HAMMER: (
                    0
                    if mist_blocks_attack
                    else 3
                    if _hammer_changes_attack(current, player) and not attack_is_ko
                    else 9
                    if any(
                        _has_special_energy(pokemon)
                        for pokemon in _bench(_opponent_state(current))
                    )
                    and not attack_is_ko
                    else 30
                ),
            }
            return priorities.get(card_id, 25), 0, index
        if option_type == 12:
            # Do not discard attack Energy merely to change positions. Retreat
            # is reserved for a clear handoff to an energized Alakazam.
            return (
                12 if own_turn > 1 and _retreat_improves_attack(player) else 120,
                0,
                index,
            )
        if option_type == 14:
            return (99, 0, index)
        return (50, 0, index)

    ranked = sorted(enumerate(options), key=score)
    return [ranked[0][0]] if ranked else []


def agent(obs_dict: dict[str, Any]) -> list[int]:
    """Return one legal action for the deterministic Alakazam V5 policy."""
    if obs_dict.get("select") is None:
        return read_deck_csv()

    select = obs_dict["select"]
    if int(select.get("type", 0)) == 0 and int(select.get("context", 0)) == 0:
        return _main_action(obs_dict)
    return _select_effect(obs_dict)
