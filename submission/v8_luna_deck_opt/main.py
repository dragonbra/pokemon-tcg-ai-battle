"""Alakazam V7: a rule-semantic strategy rebuild on the V6 runtime.

The deck and simulator bridge stay fixed. V7 adds explicit turn memory and a
pre-attack preparation gate before the remaining deterministic score ordering,
so attack is never treated as an ordinary post-setup action when a required
preparation option is visible.
"""

from __future__ import annotations

from collections import Counter
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
BUDEW = 235

IONO_POKEMON = {265, 268, 269, 270, 271}
IONO_LOW_PRIZE_POKEMON = {265, 268, 270}


ENRICHING_ENERGY = 13
ENRICHING_DRAW_COUNT = 4
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
NIGHTTIME_MINE = 1266

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
TELEPORTATION_ATTACK = 1070
ITCHY_POLLEN_ATTACK = 323
POWERFUL_HAND_ATTACK = 1072

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
# Prize values for rule-box Pokémon present in the registered opponent pool.
# Mega Pokémon ex are worth three Prizes; ordinary Pokémon ex are worth two.
THREE_PRIZE_POKEMON = {678, 723, 756}
TWO_PRIZE_POKEMON = {
    30,
    40,
    63,
    75,
    80,
    96,
    108,
    121,
    130,
    140,
    150,
    153,
    154,
    176,
    184,
    190,
    207,
    210,
    269,
    306,
    320,
    337,
    340,
    389,
    481,
    990,
    997,
    1071,
}
BASIC_ENERGY_IDS = set(range(1, 10))
MIST_ENERGY = 11
ROCK_FIGHTING_ENERGY = 20
PROTECTIVE_DAMAGE_ENERGIES = {MIST_ENERGY, ROCK_FIGHTING_ENERGY}
FIGHTING_ENERGY_TYPE = 6
_POKEMON_ENERGY_TYPES: dict[int, int] | None = None
HAND_DRAW_STOP = 20
DECK_DRAW_STOP = 10
DECK_DRAW_WATCH = 15
SUPPORTER_IDS = {XEROSIC, HILDA, DAWN, LANAS_AID, BOSS_ORDERS}
ITEM_IDS = {
    ENHANCED_HAMMER,
    NIGHT_STRETCHER,
    POFFIN,
    POKE_PAD,
    RARE_CANDY,
    WONDROUS_PATCH,
    SACRED_ASH,
    BATTLE_CAGE,
}
TRACKED_RESOURCE_IDS = {ABRA, KADABRA, ALAKAZAM, DUNSPARCE, DUDUNSPARCE, BASIC_PSYCHIC}
_EFFECT_PROGRESS: dict[int, int] = {}


class TurnMemory:
    """Record timing facts that a single observation cannot reconstruct."""

    def __init__(self) -> None:
        self.turn_key: tuple[int, int] | None = None
        self.turn_start_serials: set[int] = set()
        self.played_at_turn: dict[int, int] = {}
        self.evolved_at_turn: dict[int, int] = {}
        self.evolved_this_turn: set[int] = set()
        self.known_field_serials: set[int] = set()
        self.supporter_used = False
        self.manual_energy_used = False
        self.retreat_used = False
        self.attack_submitted = False
        self.end_submitted = False
        self.item_lock_turn_key: tuple[int, int] | None = None
        self.seen_item_lock_events: set[tuple[int, int, int, int]] = set()
        self.last_main_options: list[dict[str, Any]] = []
        self.resource_ledger: dict[str, Counter[int]] = {}
        self.resource_unknown: Counter[int] = Counter()
        self.reset()

    def reset(self) -> None:
        self.turn_key = None
        self.turn_start_serials.clear()
        self.played_at_turn.clear()
        self.evolved_at_turn.clear()
        self.evolved_this_turn.clear()
        self.known_field_serials.clear()
        self.supporter_used = False
        self.manual_energy_used = False
        self.retreat_used = False
        self.attack_submitted = False
        self.end_submitted = False
        self.item_lock_turn_key = None
        self.seen_item_lock_events.clear()
        self.last_main_options.clear()
        self.resource_ledger.clear()
        self.resource_unknown.clear()

    @staticmethod
    def _field_pokemon(player: dict[str, Any]) -> list[dict[str, Any]]:
        return [card for card in [*(player.get("active") or []), *(player.get("bench") or [])] if card]

    @staticmethod
    def _serial(card: dict[str, Any]) -> int | None:
        value = card.get("serial")
        return int(value) if value is not None else None

    def sync(
        self,
        current: dict[str, Any],
        player: dict[str, Any],
        *,
        logs: list[dict[str, Any]] | None = None,
    ) -> None:
        turn_key = (int(current.get("turn", -1)), int(current.get("yourIndex", 0)))
        field_pokemon = self._field_pokemon(player)
        field_serials = {
            serial
            for serial in (self._serial(card) for card in field_pokemon)
            if serial is not None
        }
        if turn_key != self.turn_key:
            if self.item_lock_turn_key != turn_key:
                self.item_lock_turn_key = None
            self.turn_key = turn_key
            self.turn_start_serials = {
                serial
                for serial in (self._serial(card) for card in self._field_pokemon(player))
                if serial is not None
            }
            self.evolved_this_turn.clear()
            self.supporter_used = bool(current.get("supporterPlayed", False))
            self.manual_energy_used = bool(current.get("energyAttached", False))
            self.retreat_used = bool(current.get("retreated", False))
            self.attack_submitted = False
            self.end_submitted = False
            self.known_field_serials = set(field_serials)
        else:
            # EVOLVE replaces the source card with a new serial. The next
            # observation therefore carries the only stable bridge we have:
            # the new card's preEvolution serial. Mark both serials so a
            # newly evolved Kadabra cannot be skipped before Psychic Draw.
            for card in field_pokemon:
                serial = self._serial(card)
                if serial is None or serial in self.known_field_serials:
                    continue
                previous = card.get("preEvolution") or []
                if previous:
                    previous_serials = {
                        int(item.get("serial"))
                        for item in previous
                        if isinstance(item, dict) and item.get("serial") is not None
                    }
                    self.record_evolution(serial, previous_serials)
            self.known_field_serials.update(field_serials)

        opponent_index = 1 - int(current.get("yourIndex", 0))
        for log in logs or current.get("logs") or []:
            if not (
                log.get("type") == 15
                and int(log.get("playerIndex", -1)) == opponent_index
                and int(log.get("cardId", -1)) == BUDEW
                and int(log.get("attackId", -1)) == ITCHY_POLLEN_ATTACK
            ):
                continue
            event_key = (
                opponent_index,
                BUDEW,
                ITCHY_POLLEN_ATTACK,
                int(log.get("serial", -1) or -1),
            )
            if event_key not in self.seen_item_lock_events:
                self.seen_item_lock_events.add(event_key)
                self.item_lock_turn_key = turn_key
                break

        turn = turn_key[0]
        for card in self._field_pokemon(player):
            serial = self._serial(card)
            if serial is not None:
                self.played_at_turn.setdefault(serial, turn)

        zones: dict[str, Counter[int]] = {}
        for zone in ("hand", "deck", "discard"):
            zones[zone] = Counter(
                int(card["id"])
                for card in (player.get(zone) or [])
                if card and card.get("id") in TRACKED_RESOURCE_IDS
            )
        zones["active"] = Counter(
            int(card["id"])
            for card in (player.get("active") or [])
            if card and card.get("id") in TRACKED_RESOURCE_IDS
        )
        zones["bench"] = Counter(
            int(card["id"])
            for card in (player.get("bench") or [])
            if card and card.get("id") in TRACKED_RESOURCE_IDS
        )
        zones["prize"] = Counter(
            int(card["id"])
            for card in (player.get("prize") or [])
            if isinstance(card, dict) and card.get("id") in TRACKED_RESOURCE_IDS
        )
        self.resource_ledger = zones
        totals = {
            ABRA: 4,
            KADABRA: 4,
            ALAKAZAM: 4,
            DUNSPARCE: 3,
            DUDUNSPARCE: 3,
            BASIC_PSYCHIC: 2,
        }
        self.resource_unknown = Counter(
            {
                card_id: max(0, total - sum(zone.get(card_id, 0) for zone in zones.values()))
                for card_id, total in totals.items()
            }
        )

    def was_in_play_at_turn_start(self, serial: int | None) -> bool:
        return serial is not None and serial in self.turn_start_serials

    def record_evolution(
        self, serial: int | None, previous_serials: Iterable[int] | None = None
    ) -> None:
        serials = {serial} if serial is not None else set()
        serials.update(int(item) for item in (previous_serials or []))
        if not serials:
            return
        turn = self.turn_key[0] if self.turn_key else -1
        for item in serials:
            self.evolved_at_turn[item] = turn
            self.evolved_this_turn.add(item)

    def record_main_action(
        self, option: dict[str, Any], current: dict[str, Any], player: dict[str, Any]
    ) -> None:
        option_type = int(option.get("type", -1))
        card_id = option.get("cardId")
        if card_id is None and option_type == 7:
            index = option.get("index")
            hand = player.get("hand") or []
            if isinstance(index, int) and 0 <= index < len(hand):
                card_id = hand[index].get("id")
        if card_id is not None:
            card_id = int(card_id)
        if option_type == 7 and card_id in SUPPORTER_IDS:
            self.supporter_used = True
        elif option_type == 8:
            self.manual_energy_used = True
        elif option_type == 12:
            self.retreat_used = True
        elif option_type == 13:
            self.attack_submitted = True
        elif option_type == 14:
            self.end_submitted = True
        elif option_type == 9:
            area = _field_option_area(option)
            index = option.get("inPlayIndex", option.get("index"))
            field = player.get("active") if area == 4 else player.get("bench") if area == 5 else []
            if isinstance(index, int) and 0 <= index < len(field or []):
                source_serial = self._serial(field[index])
                evolved_serial = option.get("serial")
                if evolved_serial is not None:
                    evolved_serial = int(evolved_serial)
                self.record_evolution(source_serial, [evolved_serial] if evolved_serial else None)


_TURN_MEMORY = TurnMemory()


def read_deck_csv() -> list[int]:
    values = [line.strip() for line in DECK_PATH.read_text(encoding="utf-8").splitlines()]
    deck = [int(value) for value in values if value]
    if len(deck) != 60:
        raise ValueError(f"{DECK_PATH.name} must contain 60 cards, got {len(deck)}")
    return deck


def _v7_terminal_prize_closure(
    current: dict[str, Any],
    player: dict[str, Any],
    gain: int = 0,
    *,
    options: list[dict[str, Any]] | None = None,
) -> bool:
    """Whether a visible handoff/draw can close the final Prize this turn."""
    remaining = len(player.get("prize") or [])
    target = _opponent_active(current)
    active = _active(player)
    if remaining <= 0 or not target or not active:
        return remaining <= 0

    dudunsparce_draw_available = active.get("id") == DUDUNSPARCE and options is None
    dudunsparce_retreat_available = False
    if options is not None:
        active_id = active.get("id")
        if active_id == DUDUNSPARCE:
            dudunsparce_draw_available = any(
                option.get("type") == 10
                and _option_card_id(option, {"type": 0}, current) == DUDUNSPARCE
                for option in options
            )
            dudunsparce_retreat_available = any(option.get("type") == 12 for option in options)
            has_route_option = dudunsparce_draw_available or dudunsparce_retreat_available
        elif active_id == ALAKAZAM:
            has_route_option = any(option.get("type") == 13 for option in options)
        else:
            has_route_option = any(option.get("type") == 12 for option in options)
        if not has_route_option:
            return False

    hand_count = int(player.get("handCount", len(player.get("hand") or [])))
    candidate_attackers: list[tuple[dict[str, Any], int]] = []
    if active.get("id") == ALAKAZAM and _has_psychic_energy(active):
        candidate_attackers.append((active, hand_count + gain))

    ready_bench = [
        pokemon
        for pokemon in _bench(player)
        if pokemon.get("id") == ALAKAZAM and _has_psychic_energy(pokemon)
    ]
    if active.get("id") == DUDUNSPARCE and dudunsparce_draw_available:
        draw_gain = DRAW_ABILITY_GAIN[DUDUNSPARCE]
        candidate_attackers.extend(
            (pokemon, hand_count + gain + draw_gain) for pokemon in ready_bench
        )
    elif (
        active.get("id") == DUDUNSPARCE
        and dudunsparce_retreat_available
        and _retreat_available(current)
    ):
        candidate_attackers.extend((pokemon, hand_count + gain) for pokemon in ready_bench)
    elif ready_bench and _retreat_available(current):
        candidate_attackers.extend((pokemon, hand_count + gain) for pokemon in ready_bench)

    for attacker, attack_hand_count in candidate_attackers:
        damage = _attack_damage_against(attacker.get("id"), attack_hand_count, target)
        if damage > 0 and int(target.get("hp", 9999)) <= damage:
            return remaining <= _prize_value(target.get("id"))
    return False


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


def _first_turn_setup_target_reached(player: dict[str, Any]) -> bool:
    """Whether two Abra and one Dunsparce are already on the field."""
    field_ids = _card_ids(_field_pokemon(player))
    return field_ids.count(ABRA) >= 2 and DUNSPARCE in field_ids


def _ready_attack_line_count(player: dict[str, Any]) -> int:
    """Count Abra-line Pokémon that can attack with their attached Energy."""
    return sum(
        pokemon.get("id") in ATTACK_LINE and _has_psychic_energy(pokemon)
        for pokemon in _field_pokemon(player)
    )


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
    if active_id in ATTACK_LINE:
        return _has_psychic_energy(active)
    return _energy_count(active) >= ATTACK_ENERGY_COUNT.get(active_id, 99)


def _has_alakazam_attack_path(
    player: dict[str, Any], current: dict[str, Any] | None = None
) -> bool:
    """Whether the Active can evolve into an energized Alakazam immediately."""
    active = _active(player)
    if not active or active.get("id") not in {ABRA, KADABRA}:
        return False
    if current is not None and active.get("id") == ABRA and _item_lock_active(current):
        return False
    hand_ids = _hand_ids(player)
    has_psychic = _has_psychic_energy(active)
    return has_psychic and ALAKAZAM in hand_ids and (
        RARE_CANDY in hand_ids or active.get("id") == KADABRA
    )


def _active_evolution_is_legal(current: dict[str, Any], player: dict[str, Any]) -> bool:
    """Whether the Active can evolve this turn under the runtime's rules."""
    active = _active(player)
    serial = (active or {}).get("serial")
    return bool(
        active
        and _own_turn_number(current) > 1
        and not active.get("appearThisTurn", False)
        and _TURN_MEMORY.was_in_play_at_turn_start(serial)
        and serial not in _TURN_MEMORY.evolved_this_turn
    )


def _item_lock_active(current: dict[str, Any]) -> bool:
    """Whether Itchy Pollen locks Item cards during this own turn."""
    turn_key = (int(current.get("turn", -1)), int(current.get("yourIndex", 0)))
    return _TURN_MEMORY.item_lock_turn_key == turn_key


def _rare_candy_route_available(current: dict[str, Any], player: dict[str, Any]) -> bool:
    """Rare Candy is a strategy route only when the current turn permits Items."""
    return not _item_lock_active(current) and RARE_CANDY in _hand_ids(player)


def _is_abra_attack(option: dict[str, Any], active_id: int | None) -> bool:
    return option.get("type") == 13 and active_id == ABRA


def _is_forbidden_terminal_attack(option: dict[str, Any], active_id: int | None) -> bool:
    return _is_abra_attack(option, active_id) or (
        option.get("type") == 13
        and active_id == DUNSPARCE
        and option.get("attackId") == TRADING_PLACES_ATTACK
    )


def _enriching_draw_route(current: dict[str, Any], player: dict[str, Any]) -> bool:
    """Whether Enriching Energy can feed the current Dunsparce draw engine."""
    if ENRICHING_ENERGY not in _hand_ids(player):
        return False
    if not any(
        pokemon.get("id") in {DUNSPARCE, DUDUNSPARCE}
        for pokemon in _field_pokemon(player)
    ):
        return False
    active = _active(player)
    if active and active.get("id") == ALAKAZAM and _has_psychic_energy(active):
        return True
    if active and active.get("id") == DUDUNSPARCE:
        return True
    return _item_lock_active(current)


def _poffin_dunsparce_draw_route(current: dict[str, Any], player: dict[str, Any]) -> bool:
    """Whether Poffin may reserve a slot for a concrete Enriching draw line.

    Dunsparce is a draw engine, not an Abra-line successor. Keep this route
    deliberately narrow: the field must already contain three Abra-line
    Pokémon, no Dunsparce may be present, Enriching Energy must be visible,
    and the Active must have a legal immediate Alakazam route. In every other
    state Poffin continues to prioritize Abra for Bench continuity.
    """
    if ENRICHING_ENERGY not in _hand_ids(player):
        return False
    if _attack_line_count(player) < 3:
        return False
    if any(
        pokemon.get("id") in {DUNSPARCE, DUDUNSPARCE}
        for pokemon in _field_pokemon(player)
    ):
        return False
    active = _active(player)
    if not active or active.get("id") not in {ABRA, KADABRA}:
        return False
    return _active_direct_alakazam_route(current, player)


def _active_direct_alakazam_route(current: dict[str, Any], player: dict[str, Any]) -> bool:
    """Whether the Active has a legal immediate Abra/Kadabra -> Alakazam line."""
    if _item_lock_active(current) and (_active(player) or {}).get("id") == ABRA:
        return False
    return _active_evolution_is_legal(current, player) and _active_can_become_alakazam_now(
        player, current
    )


def _active_hilda_rare_candy_route(current: dict[str, Any], player: dict[str, Any]) -> bool:
    """Whether Hilda can set up a legal Rare Candy attack this turn."""
    if not _rare_candy_route_available(current, player):
        return False
    return _active_evolution_is_legal(current, player) and _active_can_use_hilda_rare_candy(
        player
    )


def _active_hilda_target_route(current: dict[str, Any], player: dict[str, Any]) -> bool:
    """Same route after Hilda has left hand and its search is in progress."""
    if not _rare_candy_route_available(current, player):
        return False
    active = _active(player)
    hand_ids = set(_hand_ids(player))
    return bool(
        _active_evolution_is_legal(current, player)
        and active
        and active.get("id") == ABRA
        and RARE_CANDY in hand_ids
        and (_has_psychic_energy(active) or _psychic_energy_in_hand(player))
    )


def _bench_direct_alakazam_route(current: dict[str, Any], player: dict[str, Any]) -> bool:
    """Whether an energized Bench Abra is the current Rare Candy target."""
    if _item_lock_active(current):
        return False
    if not {ALAKAZAM, RARE_CANDY}.issubset(_hand_ids(player)):
        return False
    return any(
        pokemon.get("id") == ABRA
        and _has_psychic_energy(pokemon)
        and _evolution_target_is_legal(current, player, pokemon)
        for pokemon in _bench(player)
    )


def _has_unenergized_bench_abra(current: dict[str, Any], player: dict[str, Any]) -> bool:
    """Whether a Bench Abra can still be used for Kadabra's draw Ability."""
    return any(
        pokemon.get("id") == ABRA
        and not _has_psychic_energy(pokemon)
        and _evolution_target_is_legal(current, player, pokemon)
        for pokemon in _bench(player)
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
            pokemon.get("id") == ABRA
            and not pokemon.get("appearThisTurn", False)
            and _TURN_MEMORY.was_in_play_at_turn_start(pokemon.get("serial"))
            and pokemon.get("serial") not in _TURN_MEMORY.evolved_this_turn
            for pokemon in _field_pokemon(player)
        )
    )


def _has_unenergized_attack_line(player: dict[str, Any]) -> bool:
    """Whether any Abra-line Pokémon still needs its first Psychic Energy."""
    return any(
        pokemon.get("id") in ATTACK_LINE and not _has_psychic_energy(pokemon)
        for pokemon in _field_pokemon(player)
    )


def _can_use_enriching_for_draw(
    player: dict[str, Any], current: dict[str, Any] | None = None
) -> bool:
    """Whether the confirmed V6 Enriching Energy draw route is available."""
    if current is not None and _enriching_draw_route(current, player):
        return True
    active = _active(player)
    return bool(
        active
        and active.get("id") == ALAKAZAM
        and _has_psychic_energy(active)
        and ENRICHING_ENERGY in _hand_ids(player)
        and any(pokemon.get("id") == DUDUNSPARCE for pokemon in _field_pokemon(player))
    )


def _has_usable_attack_path(current: dict[str, Any], player: dict[str, Any]) -> bool:
    """Whether this turn has an attack now or a visible immediate evolution path."""
    return _has_attack_ready_pokemon(player) or _has_alakazam_attack_path(player, current)


def _boss_changes_prize_race(
    current: dict[str, Any], player: dict[str, Any], attack_is_legal: bool
) -> bool:
    """Whether Boss can expose a currently certain knockout."""
    if not attack_is_legal:
        return False
    active = _active(player)
    damage = _attack_damage_against(
        (active or {}).get("id"),
        int(player.get("handCount", len(player.get("hand") or []))),
        _opponent_active(current),
    )
    target = _opponent_active(current)
    if not target:
        return False
    active_ko = damage > 0 and int(target.get("hp", 9999)) <= damage
    return not active_ko and bool(_boss_ko_targets(current, player))


def _boss_ko_targets(
    current: dict[str, Any], player: dict[str, Any], *, hand_count: int | None = None
) -> list[dict[str, Any]]:
    """Return visible Bench targets the current attack can definitely KO."""
    active = _active(player)
    if not active:
        return []
    if hand_count is None:
        # Boss itself leaves the hand before Alakazam's damage is evaluated.
        hand_count = max(0, int(player.get("handCount", len(player.get("hand") or []))) - 1)
    active_id = active.get("id")
    return [
        pokemon
        for pokemon in _bench(_opponent_state(current))
        if (
            (damage := _attack_damage_against(active_id, hand_count, pokemon)) > 0
            and int(pokemon.get("hp", 9999)) <= damage
        )
    ]


def _boss_has_strictly_better_prize(
    current: dict[str, Any],
    player: dict[str, Any],
    attack_is_legal: bool,
    options: list[dict[str, Any]],
    select: dict[str, Any],
) -> bool:
    """Whether a visible Boss route wins more Prizes than the Active target."""
    if not attack_is_legal or current.get("supporterPlayed") or _TURN_MEMORY.supporter_used:
        return False
    if not any(
        option.get("type") == 7
        and _option_card_id(option, select, current) == BOSS_ORDERS
        for option in options
    ):
        return False
    target = _opponent_active(current)
    if not target:
        return False
    active_prizes = _prize_value(target.get("id"))
    post_boss_hand_count = max(
        0, int(player.get("handCount", len(player.get("hand") or []))) - 1
    )
    return any(
        _prize_value(pokemon.get("id")) > active_prizes
        for pokemon in _boss_ko_targets(
            current, player, hand_count=post_boss_hand_count
        )
    )


def _iono_boss_changes_prize_race(
    current: dict[str, Any], player: dict[str, Any], attack_is_legal: bool
) -> bool:
    """Allow a targeted Boss KO on a one-Prize Iono setup Pokémon."""
    if not attack_is_legal:
        return False
    opponent = _opponent_state(current)
    if not (_field_ids(opponent) & IONO_POKEMON):
        return False
    active = _active(player)
    target = _opponent_active(current)
    if not active or not target:
        return False
    post_boss_hand_count = max(
        0, int(player.get("handCount", len(player.get("hand") or []))) - 1
    )
    damage = _attack_damage_against(
        active.get("id"),
        post_boss_hand_count,
        target,
    )
    if int(target.get("hp", 9999)) <= damage:
        return False
    return any(
        pokemon.get("id") in IONO_LOW_PRIZE_POKEMON
        and int(pokemon.get("hp", 9999))
            <= _attack_damage_against(active.get("id"), post_boss_hand_count, pokemon)
        for pokemon in _bench(opponent)
    )


def _retreat_improves_attack(player: dict[str, Any]) -> bool:
    """Allow retreat only when an energized bench attacker can take over."""
    active = _active(player)
    if not active:
        return False
    if active.get("id") == ALAKAZAM:
        return False
    return any(
        pokemon.get("id") == ALAKAZAM and _has_psychic_energy(pokemon)
        for pokemon in _bench(player)
    )


def _retreat_available(current: dict[str, Any]) -> bool:
    return not bool(current.get("retreated", False)) and not _TURN_MEMORY.retreat_used


def _own_turn_number(current: dict[str, Any]) -> int:
    """Convert the simulator's shared turn counter into each player's turn."""
    turn = int(current.get("turn", 0))
    your_index = _your_index(current)
    # `turn` is shared by both players.  The first player acts on odd shared
    # turns and the second player on even shared turns, but the evaluator can
    # swap our physical index.  Use the observed firstPlayer marker instead of
    # assuming that player index 0 always starts; otherwise our second own
    # turn after a swap is misread as turn one and natural evolution is
    # incorrectly rejected.
    first_player = int(current.get("firstPlayer", 0))
    return (turn + (1 if your_index == first_player else 0)) // 2


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
    options: list[dict[str, Any]] | None = None,
) -> bool:
    """Whether a legal post-KO Flip the Script opportunity should be used."""
    if not _previous_turn_had_knockout(current, logs):
        return False
    # A non-terminal KO does not make this draw optional.  The three cards are
    # a recovery resource after losing a Pokémon, even when the current
    # Alakazam already has a legal attack or another Bench line exists.
    if _v6_terminal_prize_closure(
        current,
        player,
        gain=DRAW_ABILITY_GAIN[FEZANDIPITI_EX],
        options=options,
    ):
        return False
    if _v6_draw_is_blocked(
        current,
        player,
        gain=DRAW_ABILITY_GAIN[FEZANDIPITI_EX],
        deck_delta=DRAW_ABILITY_GAIN[FEZANDIPITI_EX],
        options=options,
    ):
        return False
    return True


def _fezandipiti_play_is_safe(
    current: dict[str, Any],
    player: dict[str, Any],
    logs: list[dict[str, Any]] | None = None,
    options: list[dict[str, Any]] | None = None,
) -> bool:
    """Keep the post-KO draw unless it would delay a terminal Prize closure."""
    return _fezandipiti_needed(current, player, logs, options)


def _telepath_requires_abra_first(player: dict[str, Any]) -> bool:
    """Whether playing a hand Abra must precede a Telepath attachment."""
    return (
        ABRA in _hand_ids(player)
        and not _has_field_card(player, ATTACK_LINE)
        and len(_bench(player)) < int(player.get("benchMax", 5))
    )


def _pokepad_kadabra_route_available(
    current: dict[str, Any], player: dict[str, Any]
) -> bool:
    """Whether Poké Pad can be treated as a visible Kadabra route.

    Poké Pad searches the deck for a non-Rule-Box Pokémon.  The observation
    normally hides the deck list, so use the turn resource ledger as the
    conservative visibility signal; a concrete deck list, when present in a
    fixture, is authoritative.  A known Item Lock or an empty deck removes
    this route.
    """
    if _item_lock_active(current) or POKE_PAD not in _hand_ids(player):
        return False
    deck = player.get("deck") or []
    if deck:
        return any(card.get("id") == KADABRA for card in deck if card)
    if int(player.get("deckCount", 0)) <= 0:
        return False
    return _TURN_MEMORY.resource_unknown.get(KADABRA, 0) > 0


def _dudunsparce_net_deck_change(removed_count: int) -> int:
    """Compute draw-three minus the cards returned by Run Away Draw."""
    return 3 - max(0, int(removed_count))


def _dudunsparce_deck_delta(
    player: dict[str, Any],
    option: dict[str, Any] | None = None,
    current: dict[str, Any] | None = None,
) -> int:
    """Return the deck delta for the Dudunsparce that will use its Ability.

    Run Away Draw shuffles the selected Dudunsparce, its evolution stack, and
    all cards attached to that instance back into the deck. When the
    observation identifies a field target, use it; otherwise retain the old
    first-visible fallback for lightweight callers that do not carry option
    coordinates.
    """
    dudunsparce = None
    if option is not None and current is not None:
        selected = _pokemon_from_option(option, current)
        if selected and selected.get("id") == DUDUNSPARCE:
            dudunsparce = selected
    if dudunsparce is None:
        dudunsparce = next(
            (pokemon for pokemon in _field_pokemon(player) if pokemon.get("id") == DUDUNSPARCE),
            None,
        )
    if dudunsparce is None:
        return 3
    stack_count = len(dudunsparce.get("preEvolution") or [])
    attached_count = len(dudunsparce.get("energyCards") or [])
    return _dudunsparce_net_deck_change(1 + stack_count + attached_count)


def _v6_terminal_prize_closure(
    current: dict[str, Any],
    player: dict[str, Any],
    gain: int = 0,
    *,
    options: list[dict[str, Any]] | None = None,
) -> bool:
    """Allow risky draw only when the same turn can take the final Prizes."""
    if _v7_terminal_prize_closure(current, player, gain, options=options):
        return True
    remaining = len(player.get("prize") or [])
    if remaining <= 0:
        return True
    attack_options = [
        option
        for option in (options or [])
        if option.get("type") == 13
        and not (
            (_active(player) or {}).get("id") == DUNSPARCE
            and option.get("attackId") == TRADING_PLACES_ATTACK
        )
    ]
    if options is not None and not attack_options:
        return False
    target = _opponent_active(current)
    active = _active(player)
    if not target or not active:
        return False
    hand_count = int(player.get("handCount", len(player.get("hand") or [])))
    post_draw_hand_count = max(0, hand_count + gain)
    damage = _attack_damage_against(active.get("id"), post_draw_hand_count, target)
    if damage > 0 and int(target.get("hp", 9999)) <= damage:
        return remaining <= _prize_value(target.get("id"))
    supporter_available = not (
        _TURN_MEMORY.supporter_used or bool(current.get("supporterPlayed", False))
    )
    boss_options = [
        option
        for option in (options or [])
        if option.get("type") == 7 and _option_card_id(option, {"type": 0}, current) == BOSS_ORDERS
    ]
    if options is not None and not boss_options:
        return False
    if not supporter_available or BOSS_ORDERS not in _hand_ids(player):
        return False
    boss_targets = _boss_ko_targets(
        current, player, hand_count=max(0, post_draw_hand_count - 1)
    )
    if boss_targets:
        boss_value = max(_prize_value(item.get("id")) for item in boss_targets)
        return remaining <= boss_value
    return False


def _v6_draw_is_blocked(
    current: dict[str, Any],
    player: dict[str, Any],
    *,
    gain: int = 0,
    deck_delta: int = 0,
    options: list[dict[str, Any]] | None = None,
) -> bool:
    """Apply V6's time-ordered deck reserve and hand-value guard."""
    if _v6_terminal_prize_closure(current, player, gain, options=options):
        return False
    hand_count = int(player.get("handCount", len(player.get("hand") or [])))
    deck_count = int(player.get("deckCount", 0))
    deck_after = deck_count - int(deck_delta)
    if hand_count >= HAND_DRAW_STOP or deck_after <= DECK_DRAW_STOP:
        return True
    if deck_after <= DECK_DRAW_WATCH and not _attack_security(player):
        return True
    return False


def _recovery_needs(current: dict[str, Any], player: dict[str, Any]) -> tuple[bool, bool]:
    """Return the Pokémon and Psychic Energy gaps for a real attack route."""
    field = _field_pokemon(player)
    hand_ids = set(_hand_ids(player))
    field_has_line = any(pokemon.get("id") in ATTACK_LINE for pokemon in field)
    # A Stage 1/Stage 2 in hand is not a playable source by itself. Recovery
    # must still find a Basic Abra unless an Abra-line Pokémon is already in
    # play, where the missing resource is only the next evolution or Energy.
    hand_has_basic_source = ABRA in hand_ids
    needs_pokemon = not field_has_line and not hand_has_basic_source

    def has_visible_abra_evolution_route(pokemon: dict[str, Any]) -> bool:
        """Whether an unenergized Abra can become a planned attacker next turn."""
        if pokemon.get("id") != ABRA:
            return True
        if KADABRA in hand_ids:
            return True
        return (
            ALAKAZAM in hand_ids
            and RARE_CANDY in hand_ids
            and not _item_lock_active(current)
        )

    field_has_attack_route = any(
        pokemon.get("id") in {KADABRA, ALAKAZAM}
        or (
            pokemon.get("id") == ABRA
            and has_visible_abra_evolution_route(pokemon)
        )
        for pokemon in field
    )
    needs_energy = any(
        pokemon.get("id") in {KADABRA, ALAKAZAM}
        and not _has_psychic_energy(pokemon)
        or (
            pokemon.get("id") == ABRA
            and not _has_psychic_energy(pokemon)
            and has_visible_abra_evolution_route(pokemon)
        )
        for pokemon in field
    )
    if (field_has_attack_route or hand_has_basic_source) and not _psychic_energy_in_hand(player):
        needs_energy = True
    elif needs_pokemon and not _psychic_energy_in_hand(player):
        # A recovery-only route still needs its first Psychic Energy; Lana's
        # Aid can bring both pieces back in the same Supporter action.
        needs_energy = True
    return needs_pokemon, needs_energy


def _recovery_can_complete_route(
    current: dict[str, Any], player: dict[str, Any], effect_id: int
) -> bool:
    """Only spend recovery when it fills a visible current/next attack route."""
    discard = set(_discard_ids(player))
    needs_pokemon, needs_energy = _recovery_needs(current, player)
    if not (needs_pokemon or needs_energy):
        return False
    can_recover_pokemon = (
        ABRA in discard if needs_pokemon else bool(discard & ATTACK_LINE)
    )
    can_recover_energy = BASIC_PSYCHIC in discard
    if effect_id == LANAS_AID:
        return (not needs_pokemon or can_recover_pokemon) and (
            not needs_energy or can_recover_energy
        )
    if effect_id == NIGHT_STRETCHER:
        if needs_pokemon and needs_energy:
            return False
        return (not needs_pokemon or can_recover_pokemon) and (
            not needs_energy or can_recover_energy
        )
    return False


def _recovery_requires_pokemon_and_energy(player: dict[str, Any]) -> bool:
    """Whether Lana's Aid can replace a two-card recovery sequence."""
    discard = set(_discard_ids(player))
    needs_pokemon, needs_energy = _recovery_needs({}, player)
    return (
        needs_pokemon
        and needs_energy
        and bool(discard & ATTACK_LINE)
        and BASIC_PSYCHIC in discard
    )


def _newly_evolved_bench_kadabra(player: dict[str, Any]) -> bool:
    return any(
        pokemon.get("id") == KADABRA
        and pokemon.get("serial") in _TURN_MEMORY.evolved_this_turn
        for pokemon in _bench(player)
    )


def _recovery_need(player: dict[str, Any]) -> int:
    """Score how much the discard pile can rebuild the next attack line."""
    discard = set(_discard_ids(player))
    return len(discard & {ABRA, KADABRA, ALAKAZAM, DUNSPARCE, DUDUNSPARCE}) + len(
        discard & {BASIC_PSYCHIC}
    )


def _pokemon_from_option(option: dict[str, Any], current: dict[str, Any]) -> dict[str, Any] | None:
    owner = int(option.get("playerIndex", _your_index(current)))
    area = _field_option_area(option)
    index = _field_option_index(option)
    player = _player(current, owner)
    if area == 4 and isinstance(index, int) and 0 <= index < len(_cards(player, "active")):
        active = _cards(player, "active")
        return active[index]
    if area == 5 and isinstance(index, int) and 0 <= index < len(_bench(player)):
        bench = _bench(player)
        return bench[index]
    return None


def _field_option_area(option: dict[str, Any]) -> int | None:
    """Resolve a field area when the runtime includes a null alias field."""
    for key in ("inPlayArea", "area"):
        value = option.get(key)
        if isinstance(value, int):
            return value
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
    option_type = option.get("type")
    area = option.get("area")
    owner = int(option.get("playerIndex", _your_index(current)))
    player = _player(current, owner)
    if index is None and area not in {4, 5}:
        return None

    if option_type == 7:  # PLAY always indexes your hand.
        hand = player.get("hand") or []
        hand_index = option.get("indexInArea", index)
        return (
            hand[hand_index].get("id")
            if (
                owner == _your_index(current)
                and isinstance(hand_index, int)
                and 0 <= hand_index < len(hand)
            )
            else None
        )
    if area == 1:  # Search effects index the cards in select.deck.
        deck = select.get("deck") or []
        return deck[index].get("id") if 0 <= index < len(deck) else None
    if area == 2:  # Opponent hand cards are hidden and cannot be inferred.
        hand = player.get("hand") or []
        hand_index = option.get("indexInArea", index)
        return (
            hand[hand_index].get("id")
            if (
                owner == _your_index(current)
                and isinstance(hand_index, int)
                and 0 <= hand_index < len(hand)
            )
            else None
        )
    if area == 3:
        discard = player.get("discard") or []
        discard_index = option.get("indexInArea", index)
        if isinstance(discard_index, int) and 0 <= discard_index < len(discard):
            return discard[discard_index].get("id")
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
    if area in {4, 5}:
        index = _field_option_index(option)
        if not isinstance(index, int):
            return None
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


def _field_option_index(option: dict[str, Any]) -> int | None:
    """Resolve a field target when the runtime emits explicit null indexes."""
    for key in ("inPlayIndex", "indexInArea", "index"):
        value = option.get(key)
        if isinstance(value, int):
            return value
    return None


def _is_empty_bench_run_away_draw(
    option: dict[str, Any], card_id: int | None, player: dict[str, Any]
) -> bool:
    """Never wash the only Active Dudunsparce back into an empty Bench."""
    return (
        option.get("type") == 10
        and card_id == DUDUNSPARCE
        and (_active(player) or {}).get("id") == DUDUNSPARCE
        and not _bench(player)
    )


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


def _pokemon_energy_type(card_id: int | None) -> int | None:
    """Resolve a Pokémon's printed type from the bundled engine card table."""
    global _POKEMON_ENERGY_TYPES
    if card_id is None:
        return None
    if _POKEMON_ENERGY_TYPES is None:
        try:
            from cg.api import all_card_data

            _POKEMON_ENERGY_TYPES = {
                int(card.cardId): int(card.energyType)
                for card in all_card_data()
                if int(card.cardType) == 0
            }
        except Exception:
            # The strategy must remain usable in lightweight trace/unit-test
            # loaders. Unknown Rock targets are handled conservatively below.
            _POKEMON_ENERGY_TYPES = {}
    return _POKEMON_ENERGY_TYPES.get(card_id)


def _has_protective_damage_energy(pokemon: dict[str, Any] | None) -> bool:
    """Whether attached protection blocks Alakazam's DamageCounter effect.

    Mist Energy protects every Pokémon. Rock Fighting Energy protects only a
    Fighting Pokémon; when a reduced test loader cannot expose the card table,
    an unknown Rock target is treated as potentially protected for KO safety.
    """
    attached = set(_attached_energy_ids(pokemon))
    if MIST_ENERGY in attached:
        return True
    if ROCK_FIGHTING_ENERGY not in attached:
        return False
    target_type = _pokemon_energy_type((pokemon or {}).get("id"))
    return target_type in {FIGHTING_ENERGY_TYPE, None}


def _attack_damage_against(
    active_id: int | None, hand_count: int, target: dict[str, Any] | None
) -> int:
    """Estimate damage using the official runtime's protective-energy rules.

    In this engine Alakazam's Powerful Hand is implemented as DamageCounter,
    so Mist Energy and effective Rock Fighting Energy prevent it even though
    the printed card text describes the protection as preventing attack
    effects. The engine behavior makes this a strategy-level rule.
    """
    if active_id == ALAKAZAM and _has_protective_damage_energy(target):
        return 0
    return _estimated_attack_damage(active_id, hand_count)


def _psychic_energy_in_hand(player: dict[str, Any]) -> bool:
    return bool(set(_hand_ids(player)) & {TELEPATH_ENERGY, BASIC_PSYCHIC})


def _active_can_become_alakazam_now(
    player: dict[str, Any], current: dict[str, Any] | None = None
) -> bool:
    """Whether the current hand already contains a direct Alakazam line."""
    active = _active(player)
    if not active or active.get("id") not in {ABRA, KADABRA}:
        return False
    hand_ids = set(_hand_ids(player))
    if ALAKAZAM not in hand_ids:
        return False
    if not (_has_psychic_energy(active) or _psychic_energy_in_hand(player)):
        return False
    if active.get("id") == ABRA and (
        RARE_CANDY not in hand_ids
        or (current is not None and _item_lock_active(current))
    ):
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


def _discarded_abra_successor_recovery_due(
    current: dict[str, Any],
    player: dict[str, Any],
    options: list[dict[str, Any]] | None = None,
) -> bool:
    """Whether Lana can build the missing Bench successor from discard.

    This is a deliberately concrete route: a charged Active is attacking,
    no Abra-line Pokémon is currently on the Bench, there is room for one,
    and both an Abra and Basic Psychic are visible in discard. It is not a
    generic recovery preference; if a ready or chargeable Abra-line target
    already exists, the ordinary handoff logic remains responsible.
    """
    active = _active(player)
    if (
        not active
        or active.get("id") not in {KADABRA, ALAKAZAM}
        or not _has_psychic_energy(active)
    ):
        return False
    bench_space = int(player.get("benchMax", 5)) - len(_bench(player))
    if bench_space <= 0:
        return False
    if any(pokemon.get("id") in ATTACK_LINE for pokemon in _bench(player)):
        return False
    discard = set(_discard_ids(player))
    if ABRA not in discard:
        return False
    if _has_visible_bench_handoff_route(current, player):
        return False
    if options is not None:
        if _v7_terminal_prize_closure(current, player, options=options):
            return False
        if not any(
            option.get("type") == 7
            and _option_card_id(option, {"type": 0}, current) == LANAS_AID
            for option in options
        ):
            return False
    return True


def _discarded_abra_successor_stretcher_due(
    current: dict[str, Any],
    player: dict[str, Any],
    options: list[dict[str, Any]] | None = None,
) -> bool:
    """Whether Night Stretcher can retrieve an Abra for a hand-held Psychic."""
    active = _active(player)
    if (
        not active
        or active.get("id") not in {KADABRA, ALAKAZAM}
        or not _has_psychic_energy(active)
        or _item_lock_active(current)
    ):
        return False
    if int(player.get("benchMax", 5)) - len(_bench(player)) <= 0:
        return False
    if any(pokemon.get("id") in ATTACK_LINE for pokemon in _bench(player)):
        return False
    if ABRA not in _discard_ids(player) or not _psychic_energy_in_hand(player):
        return False
    if _has_visible_bench_handoff_route(current, player):
        return False
    if options is not None:
        if _v7_terminal_prize_closure(current, player, options=options):
            return False
        main_card_ids = {
            _option_card_id(option, {"type": 0}, current)
            for option in options
            if option.get("type") == 7
        }
        if main_card_ids & {POFFIN, ABRA}:
            return False
        if LANAS_AID in main_card_ids and _discarded_abra_successor_recovery_due(
            current, player, options
        ):
            return False
        if not any(
            option.get("type") == 7
            and _option_card_id(option, {"type": 0}, current) == NIGHT_STRETCHER
            for option in options
        ):
            return False
    return True


def _bench_handoff_preparation_due(
    current: dict[str, Any],
    player: dict[str, Any],
    options: list[dict[str, Any]],
    select: dict[str, Any],
) -> bool:
    """Reserve the turn's Psychic resource for a visible Alakazam handoff.

    The normal Enriching Energy rule is useful while building the engine, but
    it must not consume the only attachment when the ready Active Kadabra or
    Alakazam is attacking and a Bench Kadabra/Alakazam can be made ready. This
    helper is deliberately option-driven: it only fires when the observation
    exposes either the actual attachment target or a Lana's Aid route that can
    recover the Psychic Energy.
    """
    active = _active(player)
    if (
        not active
        or active.get("id") not in {KADABRA, ALAKAZAM}
        or not _has_psychic_energy(active)
    ):
        return False
    # A direct Bench attachment is useful only when the current attack is
    # still part of the ongoing prize race. If this attack already closes
    # the final Prize, the attack ends the game and must not be delayed for a
    # successor that will never be needed.
    if options is not None and _v7_terminal_prize_closure(
        current, player, options=options
    ):
        return False

    hand_ids = set(_hand_ids(player))

    if _discarded_abra_successor_recovery_due(current, player, options):
        return True
    if _discarded_abra_successor_stretcher_due(current, player, options):
        return True

    def valid_target(target: dict[str, Any] | None) -> bool:
        if not target or target.get("id") not in {ABRA, KADABRA, ALAKAZAM}:
            return False
        if _has_psychic_energy(target):
            return False
        # Evolution timing and Energy attachment are separate rules. A
        # Kadabra/Alakazam that entered play this turn cannot evolve again,
        # but it can still receive Psychic now and become the next-turn
        # handoff. The same is true for a newly played Abra when its next
        # evolution is visible in hand: attach now, evolve next turn.
        if target.get("id") in {KADABRA, ALAKAZAM}:
            return True
        # Generic recovery/attachment callers may still target Abra before
        # Kadabra is visible; their route-specific predicates below decide
        # whether that action should outrank the current attack.
        return True

    def valid_direct_energy_target(
        target: dict[str, Any] | None, energy_id: int | None
    ) -> bool:
        """Require a concrete route for direct Psychic attachment to Abra."""
        if not valid_target(target):
            return False
        if target.get("id") != ABRA:
            return True
        # Telepath only searches Basic Psychic Pokémon.  It can attach to an
        # Abra without an evolution route, but that does not make the Abra a
        # concrete next-turn attacker; keep this gate aligned with the direct
        # Energy handoff predicate below.
        return KADABRA in hand_ids or _pokepad_kadabra_route_available(
            current, player
        ) or (
            ALAKAZAM in hand_ids
            and RARE_CANDY in hand_ids
            and not _item_lock_active(current)
        )

    def valid_patch_target(target: dict[str, Any] | None) -> bool:
        """Only Patch an Abra when its next conversion is visible.

        Patch can legally attach to any Benched Psychic Pokémon, but an
        unenergized Abra with no visible Kadabra or Rare Candy route does not
        turn that free attachment into a concrete handoff. Keep the broader
        ``valid_target`` rule for manual attachment and Lana's Aid; this
        narrower predicate is only for promoting Patch ahead of an attack.
        """
        if not valid_target(target):
            return False
        if target.get("id") != ABRA:
            return True
        return KADABRA in hand_ids or _pokepad_kadabra_route_available(
            current, player
        ) or (
            ALAKAZAM in hand_ids
            and RARE_CANDY in hand_ids
            and not _item_lock_active(current)
        )

    for option in options:
        option_type = option.get("type")
        target = _pokemon_from_option(option, current)
        energy_id = _option_card_id(option, select, current)
        if (
            option_type == 8
            and _field_option_area(option) == 5
            and energy_id in {BASIC_PSYCHIC, TELEPATH_ENERGY}
            and valid_direct_energy_target(target, energy_id)
        ):
            return True
        if (
            option_type == 7
            and _option_card_id(option, select, current) == LANAS_AID
            and BASIC_PSYCHIC in _discard_ids(player)
            and _recovery_can_complete_route(current, player, LANAS_AID)
            and any(valid_target(bench_target) for bench_target in _bench(player))
        ):
            return True

    # Wondrous Patch attaches a Basic Psychic from the discard pile directly
    # to a Benched Psychic Pokémon.  It does not consume this turn's manual
    # Energy attachment, so it is a valid handoff even when the Active
    # attacker is already charged.  The option itself is the only reliable
    # visibility signal; do not infer a Patch route from a card merely being
    # in hand.
    if (
        not _item_lock_active(current)
        and not _v7_terminal_prize_closure(current, player, options=options)
        and BASIC_PSYCHIC in _discard_ids(player)
        and any(
            option.get("type") == 7
            and _option_card_id(option, select, current) == WONDROUS_PATCH
            for option in options
        )
        and any(valid_patch_target(bench_target) for bench_target in _bench(player))
    ):
        return True
    return False


def _has_visible_bench_handoff_route(
    current: dict[str, Any], player: dict[str, Any]
) -> bool:
    """Whether Bench already contains a visible Abra-line successor.

    Dunsparce is deliberately excluded: it is a useful low-Prize buffer and
    draw engine, but it cannot take over the Psychic attack after Alakazam is
    Knocked Out.  An energized Abra counts only when the next evolution is
    visible in hand; otherwise it is still a continuity debt.
    """
    hand_ids = set(_hand_ids(player))
    item_lock = _item_lock_active(current)

    def abra_route_visible() -> bool:
        return KADABRA in hand_ids or _pokepad_kadabra_route_available(
            current, player
        ) or (
            ALAKAZAM in hand_ids and RARE_CANDY in hand_ids and not item_lock
        )

    for pokemon in _bench(player):
        pokemon_id = pokemon.get("id")
        if pokemon_id in {KADABRA, ALAKAZAM} and _has_psychic_energy(pokemon):
            return True
        if (
            pokemon_id == ABRA
            and _has_psychic_energy(pokemon)
            and abra_route_visible()
        ):
            return True
    return False


def _is_concrete_bench_psychic_handoff_option(
    current: dict[str, Any],
    player: dict[str, Any],
    option: dict[str, Any],
    select: dict[str, Any],
) -> bool:
    """Whether an Energy option can make a Bench attacker ready next turn."""
    if option.get("type") != 8 or _field_option_area(option) != 5:
        return False
    energy_id = _option_card_id(option, select, current)
    if energy_id not in {BASIC_PSYCHIC, TELEPATH_ENERGY}:
        return False
    target = _pokemon_from_option(option, current)
    if not target or _has_psychic_energy(target):
        return False
    target_id = target.get("id")
    if target_id in {KADABRA, ALAKAZAM}:
        return True
    if target_id != ABRA:
        return False
    hand_ids = set(_hand_ids(player))
    return KADABRA in hand_ids or _pokepad_kadabra_route_available(
        current, player
    ) or (
        ALAKAZAM in hand_ids
        and RARE_CANDY in hand_ids
        and not _item_lock_active(current)
    )


def _active_can_attack_after_psychic_attachment(
    current: dict[str, Any],
    player: dict[str, Any],
    options: list[dict[str, Any]],
    select: dict[str, Any],
) -> bool:
    """Whether Active should keep Psychic for an attack in this turn.

    The Bench handoff rule is a continuity rule, not permission to abandon a
    legal current-turn Alakazam attack.  This is especially important after
    Kadabra evolved into Alakazam earlier in the same turn: the new Alakazam
    can still receive Energy and attack before the turn ends.
    """
    if _own_turn_number(current) < 2:
        return False
    active = _active(player)
    active_id = (active or {}).get("id")
    if active_id not in {ABRA, KADABRA, ALAKAZAM}:
        return False
    has_active_psychic_option = any(
        option.get("type") == 8
        and _field_option_area(option) == 4
        and _option_card_id(option, select, current) in {BASIC_PSYCHIC, TELEPATH_ENERGY}
        and _pokemon_from_option(option, current) is not None
        for option in options
    )
    if not has_active_psychic_option:
        return False
    if active_id == ALAKAZAM:
        return True
    if not _active_evolution_is_legal(current, player):
        return False
    hand_ids = set(_hand_ids(player))
    if active_id == KADABRA:
        return ALAKAZAM in hand_ids
    return (
        ALAKAZAM in hand_ids
        and RARE_CANDY in hand_ids
        and not _item_lock_active(current)
    )


def _unenergized_active_has_bench_psychic_handoff(
    current: dict[str, Any],
    player: dict[str, Any],
    options: list[dict[str, Any]],
    select: dict[str, Any],
) -> bool:
    """Whether a currently unenergized Active can hand off to Bench instead."""
    active = _active(player)
    if not active or active.get("id") not in ATTACK_LINE:
        return False
    if _has_psychic_energy(active):
        return False
    if _active_can_attack_after_psychic_attachment(current, player, options, select):
        return False
    return any(
        _is_concrete_bench_psychic_handoff_option(current, player, option, select)
        for option in options
    )


def _bench_insurance_due(
    current: dict[str, Any],
    player: dict[str, Any],
    options: list[dict[str, Any]],
    select: dict[str, Any],
) -> bool:
    """Build a Bench anchor when no visible Abra-line handoff exists.

    The anchor is intentionally checked before the terminal attack score.  A
    Dunsparce-only Bench is not enough to cover the next KO, while a ready
    Abra-line successor or a direct Psychic attachment route is enough.
    """
    active = _active(player)
    if not active or active.get("id") not in {KADABRA, ALAKAZAM}:
        return False
    bench_space = int(player.get("benchMax", 5)) - len(_bench(player))
    if bench_space <= 0:
        return False
    if not any(option.get("type") == 13 for option in options):
        return False
    if _bench_handoff_preparation_due(current, player, options, select):
        return False
    if _own_turn_number(current) == 2 and any(
        pokemon.get("id") in ATTACK_LINE for pokemon in _bench(player)
    ):
        # A Bench line that was already present at the start of the turn is a
        # usable continuation anchor: it can receive the next manual Psychic
        # without spending this turn's attack.  The exception is a freshly
        # played/evolved, entirely unenergized line.  Those Pokémon cannot
        # evolve again this turn and do not yet represent a ready handoff;
        # when Poffin is visible, establish another Basic before attacking.
        unready_lines = [
            pokemon
            for pokemon in _bench(player)
            if pokemon.get("id") in ATTACK_LINE
        ]
        freshly_unready = bool(unready_lines) and all(
            pokemon.get("appearThisTurn", False)
            and not _has_psychic_energy(pokemon)
            for pokemon in unready_lines
        )
        if not freshly_unready:
            return False
    if _has_visible_bench_handoff_route(current, player):
        return False
    hand_ids = set(_hand_ids(player))

    def is_energy_anchor(option: dict[str, Any]) -> bool:
        if option.get("type") != 8:
            return False
        energy_id = _option_card_id(option, select, current)
        if energy_id == TELEPATH_ENERGY:
            # Telepath's search effect can create Abra on the Bench whether
            # it is attached to the Active or a Benched Psychic Pokémon.
            return True
        if energy_id != BASIC_PSYCHIC or _field_option_area(option) != 5:
            return False
        target = _pokemon_from_option(option, current)
        target_id = (target or {}).get("id")
        if target_id in {KADABRA, ALAKAZAM}:
            return True
        return target_id == ABRA and (
            KADABRA in hand_ids
            or (
                ALAKAZAM in hand_ids
                and RARE_CANDY in hand_ids
                and not _item_lock_active(current)
            )
        )

    has_anchor_option = any(
        (
            option.get("type") == 7
            and _option_card_id(option, select, current) in {POFFIN, ABRA, DUNSPARCE}
        )
        or is_energy_anchor(option)
        for option in options
    )
    if not has_anchor_option:
        return False
    # A final Prize closure is the one exception: the attack itself ends the
    # game before the empty Bench can become a liability. Do not let a merely
    # hypothetical Boss-plus-attack route suppress the Bench insurance; if
    # Boss is the right action, its own prize-race score must select it.
    terminal_closure = _v7_terminal_prize_closure(current, player, options=options)
    if not terminal_closure and _can_knockout(current, player):
        terminal_closure = len(player.get("prize") or []) <= _prize_value(
            (_opponent_active(current) or {}).get("id")
        )
    return not terminal_closure


def _bench_ready_alakazam_handoff_due(
    current: dict[str, Any],
    player: dict[str, Any],
    options: list[dict[str, Any]],
) -> bool:
    """Prepare an energized Bench Kadabra before a non-terminal KO."""
    active = _active(player)
    if (
        not active
        or active.get("id") != ALAKAZAM
        or not _has_psychic_energy(active)
        or ALAKAZAM not in _hand_ids(player)
        or not _can_knockout(current, player)
        or _v7_terminal_prize_closure(current, player, options=options)
    ):
        return False
    for option in options:
        if option.get("type") != 9:
            continue
        if _field_option_area(option) != 5:
            continue
        target = _pokemon_from_option(option, current)
        if (
            target
            and target.get("id") == KADABRA
            and _has_psychic_energy(target)
            and _evolution_target_is_legal(current, player, target)
            and _option_card_id(option, {"type": 0}, current) == ALAKAZAM
        ):
            return True
    return False


def _opponent_is_alakazam_deck(current: dict[str, Any]) -> bool:
    opponent = _opponent_state(current)
    return bool(_field_ids(opponent) & ATTACK_LINE)


def _xerosic_is_worth_playing(current: dict[str, Any], player: dict[str, Any]) -> bool:
    opponent = _opponent_state(current)
    active = _active(player)
    opponent_active = _opponent_active(current)
    hand_count = int(opponent.get("handCount", 0))
    if hand_count < 6 or _own_turn_number(current) < 2:
        return False
    if not active or active.get("id") != ALAKAZAM or not _has_psychic_energy(active):
        return False
    if not opponent_active or int(opponent_active.get("hp", 9999)) >= int(
        opponent_active.get("maxHp", opponent_active.get("hp", 9999))
    ):
        return False
    if _can_knockout(current, player) or _boss_ko_targets(current, player):
        return False
    return True


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
        if effect_id == SACRED_ASH:
            # Rebuild the Abra line from the bottom up before spending the
            # remaining Sacred Ash slots on the low-prize draw line. When
            # Rare Candy is available, Kadabra is less urgent than recovering
            # an Alakazam that can serve the direct Stage 2 route.
            order = {
                ABRA: 0,
                KADABRA: 1,
                ALAKAZAM: 2,
                DUNSPARCE: 3,
                DUDUNSPARCE: 4,
            }
            if RARE_CANDY in hand and not _item_lock_active(current):
                order[KADABRA], order[ALAKAZAM] = order[ALAKAZAM], order[KADABRA]
            return order.get(card_id, 5)
        if effect_id == DAWN:
            groups = (
                BASIC_SETUP,
                {KADABRA, DUDUNSPARCE},
                {ALAKAZAM},
            )
            group = groups[min(effect_step, len(groups) - 1)]
            if (
                effect_step == 0
                and not _has_field_card(player, {DUNSPARCE, DUDUNSPARCE})
                and DUNSPARCE in known_ids
            ):
                group = {DUNSPARCE}
            elif (
                effect_step == 1
                and active_id == DUNSPARCE
                and DUDUNSPARCE in known_ids
            ):
                group = {DUDUNSPARCE}
            if any(_option_card_id(option, select, current) in group for option in options):
                return 0 if card_id in group else 4
        if effect_id == HILDA:
            if effect_step == 0:
                if (
                    active_id == ALAKAZAM
                    and not _has_field_card(player, {ABRA})
                    and _has_field_card(player, {DUNSPARCE})
                ):
                    wanted = {DUDUNSPARCE, ENRICHING_ENERGY}
                elif _item_lock_active(current) and active_id == ABRA:
                    wanted = {KADABRA}
                elif _active_hilda_rare_candy_route(current, player) or (
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
                    if (
                        _item_lock_active(current)
                        and _has_field_card(player, {DUNSPARCE, DUDUNSPARCE})
                    )
                    or (
                        _attack_line_target_reached(player)
                        and _attack_security(player)
                    )
                    else {TELEPATH_ENERGY, BASIC_PSYCHIC, ENRICHING_ENERGY}
                )
            if any(_option_card_id(option, select, current) in wanted for option in options):
                return 0 if card_id in wanted else 4
        if effect_id == POFFIN:
            # Buddy-Buddy Poffin is limited to Basic Pokémon with 70 HP or
            # less. Complete the three-Pokémon attack line before expanding
            # the Dunsparce engine. Energy alone is not enough: an Abra with
            # no visible Kadabra/Rare Candy route is not a ready replacement
            # after a knockout, even when it already has Psychic attached.
            visible_handoff = _has_visible_bench_handoff_route(current, player)
            wanted = (
                {DUNSPARCE}
                if _poffin_dunsparce_draw_route(current, player)
                else {ABRA}
                if _attack_line_count(player) < 3
                or not visible_handoff
                else {DUNSPARCE}
            )
            return 0 if card_id in wanted else 4
        if effect_id == TELEPATH_ENERGY:
            return 0 if card_id == ABRA else 4
        if effect_id in {NIGHT_STRETCHER, LANAS_AID}:
            needs_pokemon, needs_energy = _recovery_needs(current, player)
            if effect_id == LANAS_AID and _discarded_abra_successor_recovery_due(
                current, player
            ):
                needs_pokemon = True
            if effect_id == NIGHT_STRETCHER and _discarded_abra_successor_stretcher_due(
                current, player
            ):
                needs_pokemon = True
            if effect_id == LANAS_AID:
                return {
                    ABRA: 0,
                    KADABRA: 1,
                    ALAKAZAM: 2,
                    BASIC_PSYCHIC: 3 if needs_energy and not needs_pokemon else 4,
                }.get(card_id, 5)
            if card_id == BASIC_PSYCHIC and needs_energy:
                return 0
            if card_id == ABRA and needs_pokemon:
                return 0
            return 4
        if effect_id == POKE_PAD:
            active = _active(player)
            if _active_direct_alakazam_route(current, player) or _active_hilda_rare_candy_route(
                current, player
            ):
                wanted = {ALAKAZAM}
            elif active and active.get("id") == DUNSPARCE and DUDUNSPARCE in known_ids:
                # Dudunsparce is a legal switch resource for an Active
                # Dunsparce, not merely another Stage 1 draw Pokémon.
                wanted = {DUDUNSPARCE}
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
                recovery_value = {ABRA: 90, KADABRA: 95, ALAKAZAM: 100}
                if card_id == ABRA and not _has_field_card(player, {ABRA}):
                    recovery_value[ABRA] = 125
                value = recovery_value.get(card_id, 20)
            elif effect_id == LANAS_AID and card_id == BASIC_PSYCHIC:
                value = 130
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
                ROCK_FIGHTING_ENERGY: 200,
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
        area = _field_option_area(option)

        # Enhanced Hammer uses SelectType.ENERGY (context 30 in the current
        # API), not the attached-card context used by older policy comments.
        # Protective energies are the first class of targets. Within that
        # class, an Active target is best because it removes the immediate
        # protection and the opponent's current attack resource together.
        if owner != _your_index(current) and energy_id not in BASIC_ENERGY_IDS:
            protective = energy_id in PROTECTIVE_DAMAGE_ENERGIES
            if area == 4:
                return (-400 if protective else -260), index
            return (-320 if protective else -180), index

        # The simulator should normally omit these options; a high score keeps
        # them as last-resort legal fallbacks if it does not.
        if target_id in ENERGY_CAPPED_POKEMON and _has_psychic_energy(target):
            return 90, index

        if _unenergized_active_has_bench_psychic_handoff(
            current, player, options, {"type": 0}
        ) and _is_concrete_bench_psychic_handoff_option(
            current, player, option, {"type": 0}
        ):
            # When the Active is not charged yet, a Bench Kadabra/Alakazam is
            # the better handoff if it can be made ready with this attachment.
            # The caller still falls back to Active when no such target exists.
            return -2, index

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
        if target_id in {DUNSPARCE, DUDUNSPARCE} and energy_id == ENRICHING_ENERGY:
            value = -5 if _enriching_draw_route(current, player) else 25
        return -value, index

    ranked = sorted(enumerate(options), key=score)
    return [ranked[0][0]] if ranked else []


def _prize_value(card_id: int | None) -> int:
    if card_id in THREE_PRIZE_POKEMON:
        return 3
    if card_id in TWO_PRIZE_POKEMON or card_id in RULE_BOX_POKEMON:
        return 2
    return 1


def _choose_switch_option(options: list[dict[str, Any]], current: dict[str, Any]) -> list[int]:
    your_index = _your_index(current)
    your_player = _player(current, your_index)
    active_id = (_active(your_player) or {}).get("id")
    hand_count = max(
        0, int(your_player.get("handCount", len(your_player.get("hand") or []))) - 1
    )

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
            # Boss's Orders chooses among certain KOs by remaining HP, high
            # to low, per the V6 target rule. A non-KO target stays behind
            # every certain KO regardless of its Prize value.
            return knockout, active_bonus, -hp, -_prize_value(target_id), index
        ready_handoff_exists = active_id == DUNSPARCE and any(
            pokemon.get("id") == ALAKAZAM and _has_psychic_energy(pokemon)
            for pokemon in _bench(your_player)
        )
        target_is_ready_handoff = target_id == ALAKAZAM and _has_psychic_energy(target)
        handoff_rank = 0 if not ready_handoff_exists or target_is_ready_handoff else 1
        return (
            handoff_rank,
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
        # The card draws four, but the attached card leaves hand first:
        # Alakazam's hand value therefore increases by three.
        return ENRICHING_DRAW_COUNT - 1 if card_id == ENRICHING_ENERGY else -1
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


def _can_kadabra_knockout_after_evolution(
    current: dict[str, Any], player: dict[str, Any]
) -> bool:
    """Whether an energized Active Abra can evolve and KO with Kadabra now."""
    active = _active(player)
    target = _opponent_active(current)
    if not active or active.get("id") != ABRA or not _has_psychic_energy(active) or not target:
        return False
    damage = _attack_damage_against(
        KADABRA,
        int(player.get("handCount", len(player.get("hand") or []))),
        target,
    )
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
    """Apply the hand/deck guards unless this draw directly creates lethal damage."""
    if _draw_changes_knockout(current, player, gain):
        return False
    hand_count = int(player.get("handCount", len(player.get("hand") or [])))
    deck_count = int(player.get("deckCount", 0))
    return hand_count > HAND_DRAW_STOP or deck_count <= DECK_DRAW_STOP or _can_knockout(
        current, player
    )


def _evolution_target_is_legal(
    current: dict[str, Any], player: dict[str, Any], target: dict[str, Any] | None
) -> bool:
    """Apply V6's evolution clock before trusting a simulator option."""
    if not target or target.get("id") not in {ABRA, KADABRA, DUNSPARCE}:
        return False
    serial = target.get("serial")
    return bool(
        _own_turn_number(current) > 1
        and not target.get("appearThisTurn", False)
        and _TURN_MEMORY.was_in_play_at_turn_start(serial)
        and serial not in _TURN_MEMORY.evolved_this_turn
    )


def _choose_evolution_target(
    options: list[dict[str, Any]], current: dict[str, Any], player: dict[str, Any]
) -> list[int]:
    """Choose a legal Rare Candy target while preserving a direct attack route."""
    hand_ids = set(_hand_ids(player))
    active = _active(player)
    direct_bench_route = _bench_direct_alakazam_route(current, player)

    def score(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        index, option = item
        target = _pokemon_from_option(option, current)
        target_id = (target or {}).get("id")
        area = _field_option_area(option)
        if target_id != ABRA or not _evolution_target_is_legal(current, player, target):
            return 100, index
        if target_id == ABRA and area == 4 and ALAKAZAM in hand_ids:
            return 0, index
        if target_id == ABRA and area == 5 and direct_bench_route:
            return (0 if _has_psychic_energy(target) else 2), index
        if target_id == ABRA:
            return 1, index
        if active and target is active:
            return 2, index
        return 3, index

    ranked = sorted(enumerate(options), key=score)
    legal = [item for item in ranked if score(item)[0] < 100]
    if not legal:
        return []
    ranked = legal
    return [ranked[0][0]] if ranked else []


def _select_effect(obs: dict[str, Any]) -> list[int]:
    select = obs["select"]
    options = select.get("option") or []
    min_count = int(select.get("minCount", 0))
    max_count = int(select.get("maxCount", len(options)))
    current, player = _your_state(obs)
    _TURN_MEMORY.sync(current, player, logs=obs.get("logs") or [])
    context = int(select.get("context", 0))
    select_type = int(select.get("type", 0))

    if not options:
        if min_count:
            raise ValueError("simulator returned a required selection with no options")
        return []

    if context == 37:  # Rare Candy chooses the Pokémon to evolve.
        chosen = _choose_evolution_target(options, current, player)
        if not chosen and min_count:
            # The simulator's effect options are authoritative. If local turn
            # memory cannot prove a target's history, keep the required effect
            # legal instead of returning an empty selection.
            chosen = [0]
        for option_index in chosen[: max(1, min_count)]:
            target = _pokemon_from_option(options[option_index], current)
            _TURN_MEMORY.record_evolution((target or {}).get("serial"))
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

    if context == 43 and select_type == 9:  # ACTIVATE an optional Ability.
        effect_id = _effect_id(select)
        context_card = select.get("contextCard") or {}
        effect_id = effect_id or context_card.get("id")
        gain = DRAW_ABILITY_GAIN.get(effect_id, 0)
        if gain and _v6_draw_is_blocked(
            current,
            player,
            gain=gain,
            deck_delta=gain,
            options=_TURN_MEMORY.last_main_options,
        ):
            no = [index for index, option in enumerate(options) if option.get("type") == 2]
            if no:
                chosen = no[: max(1, min_count)]
                _advance_effect(select)
                return chosen

    if context in {1, 2, 5, 6, 7, 8, 9, 10, 21, 26, 29}:
        effect_id = _effect_id(select)
        known = [_option_card_id(option, select, current) for option in options]
        useful = [card_id for card_id in known if card_id is not None]
        if effect_id == LANAS_AID:
            # Lana's Aid can recover up to three cards. Prefer the full visible
            # Abra line for a successor; take Basic Psychic only when it is the
            # remaining concrete attack gap.
            needs_pokemon, needs_energy = _recovery_needs(current, player)
            if _discarded_abra_successor_recovery_due(current, player):
                needs_pokemon = True
            pokemon_count = sum(card_id in ATTACK_LINE for card_id in useful)
            if needs_pokemon and pokemon_count:
                count = min(max_count, pokemon_count)
            elif needs_energy and BASIC_PSYCHIC in useful:
                count = min(max_count, 1)
            else:
                count = min_count
        elif effect_id == SACRED_ASH:
            count = min(max_count, len(useful))
        elif min_count:
            count = min_count
        elif effect_id in {POFFIN, TELEPATH_ENERGY}:
            wanted = {ABRA, DUNSPARCE}
            if effect_id == TELEPATH_ENERGY:
                wanted = {ABRA}
            count = min(max_count, sum(card_id in wanted for card_id in useful))
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
    _TURN_MEMORY.sync(current, player, logs=obs.get("logs") or [])
    _TURN_MEMORY.last_main_options = list(options)
    hand = _hand_ids(player)
    active = _active(player)
    active_id = (active or {}).get("id")
    bench = _bench(player)
    bench_space = int(player.get("benchMax", 5)) - len(bench)
    hand_count = int(player.get("handCount", len(hand)))
    deck_count = int(player.get("deckCount", 0))
    own_turn = _own_turn_number(current)
    early_setup = own_turn <= 1
    item_lock_active = _item_lock_active(current)
    direct_hand_route = _active_direct_alakazam_route(current, player)
    direct_hilda_route = _active_hilda_rare_candy_route(current, player)
    direct_bench_route = _bench_direct_alakazam_route(current, player)
    natural_alakazam_route = _active_natural_alakazam_route(current, player)
    attack_path = _has_usable_attack_path(current, player) or direct_hilda_route
    attack_is_legal = _has_attack_option(options)
    attack_is_ko = attack_is_legal and _can_knockout(current, player)
    opponent_active = _opponent_active(current)
    protective_energy_blocks_attack = (
        active_id == ALAKAZAM
        and _has_protective_damage_energy(opponent_active)
        and attack_is_legal
    )
    active_special_energy = _has_special_energy(opponent_active)
    attack_line_count = _attack_line_count(player)
    attack_line_exists = _has_any_attack_line(player)
    attack_line_target_reached = attack_line_count >= 3
    setup_needed = not attack_line_target_reached
    ready_alakazam = _has_ready_alakazam(player)
    recovery_need = _recovery_need(player)
    attack_recovery_available = _has_attack_recovery(player)
    telepath_requires_abra = _telepath_requires_abra_first(player)
    fezandipiti_needed = _fezandipiti_play_is_safe(
        current, player, obs.get("logs") or [], options
    )
    bench_handoff_preparation_due = _bench_handoff_preparation_due(
        current, player, options, select
    )
    bench_insurance_due = _bench_insurance_due(current, player, options, select)
    bench_ready_handoff_due = _bench_ready_alakazam_handoff_due(
        current, player, options
    )

    def preparation_is_due() -> bool:
        """Keep terminal attack behind actions that are still required now."""
        if bench_insurance_due:
            return True
        if bench_handoff_preparation_due:
            return True
        if bench_ready_handoff_due:
            return True
        for option in options:
            option_type = option.get("type")
            card_id = _option_card_id(option, select, current)
            if option_type == 9:
                target = _pokemon_from_option(option, current)
                target_area = _field_option_area(option)
                if target is not None and _evolution_target_is_legal(current, player, target):
                    if target_area == 4 or target.get("id") in {ABRA, DUNSPARCE}:
                        return True
            elif option_type == 10:
                gain = _main_draw_gain(option, card_id, player)
                deck_delta = (
                    _dudunsparce_deck_delta(player, option, current)
                    if card_id == DUDUNSPARCE
                    else gain
                )
                if card_id in DRAW_CARDS or card_id == FEZANDIPITI_EX:
                    if _v6_draw_is_blocked(
                        current, player, gain=gain, deck_delta=deck_delta, options=options
                    ):
                        continue
                if (
                    card_id == KADABRA
                    and active_id == ALAKAZAM
                    and _newly_evolved_bench_kadabra(player)
                ):
                    return True
                if _draw_changes_knockout(current, player, gain):
                    return True
                if card_id == DUDUNSPARCE:
                    return True
            elif option_type == 8:
                target = _pokemon_from_option(option, current)
                if (
                    card_id == ENRICHING_ENERGY
                    and (target or {}).get("id") == DUDUNSPARCE
                    and not _v6_draw_is_blocked(
                        current,
                        player,
                        gain=ENRICHING_DRAW_COUNT - 1,
                        deck_delta=ENRICHING_DRAW_COUNT,
                        options=options,
                    )
                ):
                    return True
            elif option_type == 7:
                if card_id == DAWN and _has_evolvable_abra(current, player):
                    # Dawn can supply Kadabra for a Bench Abra's Psychic Draw
                    # before the current Alakazam submits a non-terminal KO.
                    return True
        return False

    preparation_due = preparation_is_due()
    second_turn_boss_prize_route = (
        _boss_changes_prize_race(current, player, attack_is_legal)
        or _iono_boss_changes_prize_race(current, player, attack_is_legal)
    )
    second_turn_powerful_hand_priority = (
        active_id == ALAKAZAM
        and own_turn == 2
        and attack_is_legal
        and any(
            option.get("type") == 13
            and option.get("attackId") == POWERFUL_HAND_ATTACK
            for option in options
        )
        and not preparation_due
        and not second_turn_boss_prize_route
    )

    def score(item: tuple[int, dict[str, Any]]) -> tuple[int, int, int]:
        index, option = item
        option_type = option.get("type")
        card_id = _option_card_id(option, select, current)

        if option_type == 13:
            # Attack is a terminal submission. Even a knockout waits until
            # the current turn's necessary evolution, draw, and handoff work
            # has been completed.
            if active_id == DUNSPARCE and option.get("attackId") == TRADING_PLACES_ATTACK:
                return (120, 0, index)
            if _is_abra_attack(option, active_id):
                return (130, 0, index)
            if preparation_due:
                return (60, 0, index)
            if (
                second_turn_powerful_hand_priority
                and option.get("attackId") == POWERFUL_HAND_ATTACK
            ):
                return (1, 0, index)
            if attack_is_ko:
                return (2, 0, index)
            if protective_energy_blocks_attack:
                # If no Hammer or useful Boss target exists, ending is safer
                # than repeatedly putting zero damage into a protective
                # Energy target.
                return (110, 0, index)
            if _v6_draw_is_blocked(current, player, gain=0):
                return (72, -_estimated_attack_damage(active_id, hand_count), index)
            return (78, -_estimated_attack_damage(active_id, hand_count), index)

        if option_type == 10:
            if _is_empty_bench_run_away_draw(option, card_id, player):
                # Run Away Draw is only a handoff when another Pokémon can
                # receive the Active position. With an empty Bench it leaves
                # no Pokémon in play and is an avoidable self-loss.
                return (130, 0, index)
            gain = _main_draw_gain(option, card_id, player)
            deck_delta = (
                _dudunsparce_deck_delta(player, option, current)
                if card_id == DUDUNSPARCE
                else gain
            )
            if card_id in DRAW_CARDS or card_id == FEZANDIPITI_EX:
                if _v6_draw_is_blocked(
                    current, player, gain=gain, deck_delta=deck_delta, options=options
                ):
                    return (100, 0, index)
            if _draw_changes_knockout(current, player, gain):
                return (1, -gain, index)
            if (
                card_id == KADABRA
                and active_id == ALAKAZAM
                and _newly_evolved_bench_kadabra(player)
            ):
                return (1, -gain, index)
            if card_id == DUNSPARCE:
                return (24, 0, index)
            if card_id == FEZANDIPITI_EX and not fezandipiti_needed:
                return (130, 0, index)
            if card_id in DRAW_CARDS or card_id == FEZANDIPITI_EX:
                if card_id == DUDUNSPARCE and _can_use_enriching_for_draw(
                    player, current
                ):
                    return (5, -gain, index)
                if attack_is_legal and attack_path:
                    return (18, -gain, index)
                return (8, -gain, index)
            if card_id == PSYDUCK:
                # Damp is matchup-dependent utility. There is no visible
                # opponent card-text signal in the observation, so keep it
                # behind setup, draw, and a legal attack by default.
                return (75, 0, index)
            return (18, 0, index)

        if option_type == 9:
            evolved_id = card_id
            target = _pokemon_from_option(option, current)
            target_id = (target or {}).get("id")
            target_area = _field_option_area(option)
            if target is not None and not _evolution_target_is_legal(current, player, target):
                return (140, 0, index)
            if evolved_id == ALAKAZAM:
                if natural_alakazam_route and target_area == 4:
                    return (0, 0, index)
                if (
                    bench_ready_handoff_due
                    and target_area == 5
                    and target_id == KADABRA
                ):
                    return (0, 0, index)
                return (1 if target_area == 4 else 3, 0, index)
            if evolved_id == KADABRA:
                if (
                    target_id == ABRA
                    and target_area == 5
                    and active_id == ALAKAZAM
                    and _has_psychic_energy(active)
                    and target is not None
                    and not target.get("appearThisTurn", False)
                ):
                    return (1, 0, index)
                # If the Active Abra still has a Rare Candy + Alakazam route,
                # evolve a Benched Abra into Kadabra instead. This preserves
                # the direct attack route on the Active Pokémon.
                if target_id == ABRA and target_area == 4 and (
                    direct_hand_route or direct_hilda_route
                ):
                    return (18, 0, index)
                if (
                    target_id == ABRA
                    and target_area == 4
                    and KADABRA in hand
                    and _has_psychic_energy(active)
                    and _can_kadabra_knockout_after_evolution(current, player)
                ):
                    # A natural Active Abra -> Kadabra evolution is a
                    # complete terminal route when Kadabra can immediately
                    # take the prize.  Do not spend this turn evolving only a
                    # Bench Abra and leave the Active Abra unable to attack.
                    return (0, 0, index)
                if target_id == ABRA and target_area == 5:
                    if direct_bench_route and _has_psychic_energy(target):
                        return (18, 0, index)
                    return (1 if not _has_psychic_energy(target) else 2, 0, index)
                if target_id == ABRA:
                    return (3 if target_area == 4 else 5, 0, index)
                return (8 if attack_path else 4, 0, index)
            if evolved_id == DUDUNSPARCE:
                if item_lock_active or _enriching_draw_route(current, player):
                    return (1, 0, index)
                return (7 if attack_line_target_reached else 13, 0, index)
            return (20, 0, index)

        if option_type == 8:
            if _TURN_MEMORY.manual_energy_used or bool(current.get("energyAttached", False)):
                return (130, 0, index)
            if _v7_terminal_prize_closure(current, player, options=options):
                # The attack is already a complete final-Prize route. Any
                # optional Energy attachment would only consume the last
                # action window before the terminal attack.
                return (100, 0, index)
            energy_id = card_id
            target = _pokemon_from_option(option, current)
            target_id = (target or {}).get("id")
            target_area = _field_option_area(option)
            gain = _main_draw_gain(option, energy_id, player)
            if (
                _unenergized_active_has_bench_psychic_handoff(
                    current, player, options, select
                )
                and _is_concrete_bench_psychic_handoff_option(
                    current, player, option, select
                )
            ):
                # The Active cannot attack without this Energy. If a
                # concrete Bench attacker can receive it instead, preserve
                # the next turn's handoff rather than charging a disposable
                # Active that may be Knocked Out immediately.
                return (-2, 0, index)
            if (
                energy_id == TELEPATH_ENERGY
                and bench_insurance_due
                and target_id in ATTACK_LINE
                and bench_space > 0
            ):
                # Telepath is an Energy attachment and a Psychic-Basic Bench
                # search in one action. Prefer it to a non-terminal attack
                # when no visible Abra-line handoff exists.
                direct_anchor_available = any(
                    option.get("type") == 7
                    and _option_card_id(option, select, current)
                    in {POFFIN, ABRA, DUNSPARCE}
                    for option in options
                )
                if target_area == 5 or not direct_anchor_available:
                    return (-9, 0, index)
            if energy_id == ENRICHING_ENERGY and target_id in {DUNSPARCE, DUDUNSPARCE}:
                if bench_handoff_preparation_due:
                    return (20, 0, index)
                if not _enriching_draw_route(current, player):
                    return (100, 0, index)
                if _v6_draw_is_blocked(
                    current,
                    player,
                    gain=ENRICHING_DRAW_COUNT - 1,
                    deck_delta=ENRICHING_DRAW_COUNT,
                    options=options,
                ):
                    return (100, 0, index)
                if not attack_is_ko:
                    return (0, -gain, index)
            if _draw_changes_knockout(current, player, gain):
                return (4, -gain, index)
            if energy_id == TELEPATH_ENERGY:
                if telepath_requires_abra:
                    return (40, 0, index)
                if target_id in ATTACK_LINE:
                    if _has_psychic_energy(target):
                        return (28, 0, index)
                    if (
                        target_area == 4
                        and bench_space > 0
                        and not _has_visible_bench_handoff_route(current, player)
                    ):
                        # When the Active still needs its first Psychic
                        # Energy, Telepath can also search Basic Psychic
                        # Pokémon onto an open Bench. Prefer that concrete
                        # continuity gain over Basic Psychic, which only
                        # charges the Active.
                        return (-1, 0, index)
                    return (0 if target_area == 4 else 1, 0, index)
                return (18, 0, index)
            if energy_id == BASIC_PSYCHIC:
                if target_id in ATTACK_LINE:
                    if (
                        target_id == ABRA
                        and target_area == 5
                        and not (
                            KADABRA in _hand_ids(player)
                            or (
                                ALAKAZAM in _hand_ids(player)
                                and RARE_CANDY in _hand_ids(player)
                                and not item_lock_active
                            )
                        )
                    ):
                        # A Basic Psychic attachment does not establish a
                        # concrete Abra handoff without visible evolution
                        # evidence. Preserve the resource for the current
                        # turn instead of treating a future draw as certain.
                        return (100, 0, index)
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
                    return (2, 0, index)
                if target_id in ATTACK_LINE and not _has_psychic_energy(target):
                    # Enriching Energy only supplies Colorless. It cannot
                    # make an Abra-line Pokémon ready for its Psychic attack.
                    return (100, 0, index)
                return (22, 0, index)
            return (19, 0, index)

        if option_type == 7:
            gain = _main_draw_gain(option, card_id, player)
            if card_id in SUPPORTER_IDS and (
                _TURN_MEMORY.supporter_used or bool(current.get("supporterPlayed", False))
            ):
                return (140, 0, index)
            if item_lock_active and card_id in ITEM_IDS:
                return (140, 0, index)
            if card_id == RARE_CANDY and not _rare_candy_route_available(current, player):
                return (140, 0, index)
            if (
                card_id == LANAS_AID
                and bench_handoff_preparation_due
                and (
                    _recovery_can_complete_route(current, player, LANAS_AID)
                    or _discarded_abra_successor_recovery_due(current, player)
                )
            ):
                return (0, 0, index)
            if card_id == ABRA and telepath_requires_abra:
                return (0, 0, index)
            if card_id == FEZANDIPITI_EX:
                if _v7_terminal_prize_closure(current, player, options=options):
                    # Flip the Script is optional setup; a legal Powerful Hand
                    # that closes the final Prize must be submitted first.
                    return (130, 0, index)
                return (0 if fezandipiti_needed else 130, 0, index)
            if card_id == NIGHTTIME_MINE:
                # Our deck has no Tera Pokémon. The Stadium can therefore be
                # played immediately to replace an opposing Stadium, and it
                # also adds an attack-cost tax when the opponent uses Tera.
                return (-20, 0, index)
            if card_id == ABRA and bench_insurance_due:
                # A Basic played from hand is a valid Bench anchor even when
                # Poffin is unavailable. The next evolution/attachment is a
                # separate legal action and must be planned on the next
                # observation.
                return (-9, 0, index)
            if card_id == DUNSPARCE and bench_insurance_due:
                return (-8, 0, index)
            if card_id == XEROSIC:
                if _xerosic_is_worth_playing(current, player) and not attack_is_ko:
                    return (6, 0, index)
                return (35, 0, index)
            if card_id == BOSS_ORDERS:
                # Gust only when it creates a knockout or exposes a strictly
                # better prize. Otherwise preserving the Supporter is part of
                # the plan, even when it is currently playable.
                if _boss_changes_prize_race(
                    current, player, attack_is_legal
                ) or _iono_boss_changes_prize_race(current, player, attack_is_legal):
                    return (4, 0, index)
                if protective_energy_blocks_attack:
                    return (130, 0, index)
                return (100, 0, index)
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
                    if bench_handoff_preparation_due and (
                        _recovery_can_complete_route(current, player, LANAS_AID)
                        or _discarded_abra_successor_recovery_due(current, player)
                    ):
                        return (0, 0, index)
                    if _recovery_can_complete_route(current, player, LANAS_AID):
                        if _recovery_requires_pokemon_and_energy(player) and not _boss_changes_prize_race(
                            current, player, attack_is_legal
                        ):
                            return (3, 0, index)
                        return (7 if not attack_is_ko else 18, 0, index)
                    return (110, 0, index)
                return (9 if setup_needed else 16, -gain, index)
            if card_id == RARE_CANDY and ALAKAZAM in hand:
                if item_lock_active:
                    return (140, 0, index)
                if natural_alakazam_route:
                    return (8, 0, index)
                if direct_hand_route or direct_bench_route:
                    # Preserve an unenergized Bench Abra for Kadabra's draw
                    # Ability before spending the direct Rare Candy route.
                    return (2 if _has_unenergized_bench_abra(current, player) else 1, 0, index)
                if active_id in {ABRA, KADABRA} and attack_path:
                    return (1, 0, index)
                return (7, 0, index)
            if card_id == POFFIN:
                if bench_insurance_due:
                    # The insurance is a hard preparation gate. A draw
                    # Pokémon such as Fezandipiti may otherwise tie at score
                    # zero and win by stable option order, leaving Alakazam
                    # alone despite a legal Poffin.
                    return (-10, 0, index)
                return (
                    10
                    if bench_space > 0 and not attack_line_target_reached
                    else 12
                    if bench_space > 0
                    else 30,
                    0,
                    index,
                )
            if card_id == POKE_PAD:
                if early_setup and _first_turn_setup_target_reached(player):
                    # Two Abra plus Dunsparce already establish the first-turn
                    # board. Preserve Poké Pad for an evolution or handoff
                    # target instead of adding a redundant Basic.
                    return (130, 0, index)
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
                if _discarded_abra_successor_stretcher_due(current, player, options):
                    return (0, 0, index)
                if _recovery_can_complete_route(current, player, NIGHT_STRETCHER):
                    if _recovery_requires_pokemon_and_energy(player):
                        return (12, 0, index)
                    return (6 if not attack_is_ko else 16, 0, index)
                return (110, 0, index)
            if card_id == WONDROUS_PATCH:
                patch_ready = BASIC_PSYCHIC in _discard_ids(player) and attack_line_exists
                if bench_handoff_preparation_due and patch_ready:
                    # Patch directly completes the visible Bench handoff and
                    # preserves both the manual Energy attachment and the
                    # Supporter slot.  A generic search Supporter must not
                    # outrank this concrete resource conversion.
                    return (0, 0, index)
                if (
                    patch_ready
                    and active_id in {KADABRA, ALAKAZAM}
                    and _has_psychic_energy(active)
                    and not _v7_terminal_prize_closure(current, player, options=options)
                ):
                    # A Patch with no concrete Bench conversion route is
                    # legal but does not improve continuity.  Keep the
                    # current attack ahead of spending Psychic on an Abra
                    # whose next evolution is not visible.
                    return (90, 0, index)
                return (15 if patch_ready else 34, 0, index)
            if card_id == SACRED_ASH:
                return (15 if recovery_need >= 2 else 35, 0, index)
            priorities = {
                ENHANCED_HAMMER: (
                    0
                    if active_special_energy
                    or any(
                        _has_special_energy(pokemon)
                        for pokemon in _bench(_opponent_state(current))
                    )
                    else 30
                ),
            }
            return priorities.get(card_id, 25), 0, index
        if option_type == 12:
            # Do not discard attack Energy merely to change positions. Retreat
            # is reserved for a clear handoff to an energized Alakazam.
            return (
                # END is also scored at 99 below.  Give a dead retreat a
                # strictly worse score so the stable option order cannot
                # turn a first-turn/no-attack position change into a choice.
                12
                if own_turn > 1 and _retreat_available(current) and _retreat_improves_attack(player)
                else 100,
                0,
                index,
            )
        if option_type == 14:
            return (99, 0, index)
        return (50, 0, index)

    rankable = [
        item
        for item in enumerate(options)
        if not _is_forbidden_terminal_attack(item[1], active_id)
    ]
    if not rankable:
        # A simulator state with only a forbidden attack is malformed for the
        # intended policy. Never submit Trading Places as a strategy action;
        # return END when the runtime offers it, otherwise return no action
        # rather than fabricating a forbidden attack choice.
        rankable = [
            item for item in enumerate(options) if item[1].get("type") == 14
        ]
    ranked = sorted(rankable, key=score)
    if not ranked:
        return []
    chosen_index = ranked[0][0]
    _TURN_MEMORY.record_main_action(options[chosen_index], current, player)
    return [chosen_index]


DECK = read_deck_csv()


def agent(obs_dict: dict[str, Any]) -> list[int]:
    """Return one legal action for the deterministic Alakazam V7 policy."""
    if obs_dict.get("select") is None:
        # The evaluator reuses this module across games. Effect serials are
        # allocated by the engine per battle, so progress from the previous
        # game must not leak into a new Dawn/Hilda/Poffin sequence.
        _EFFECT_PROGRESS.clear()
        _TURN_MEMORY.reset()
        return read_deck_csv()

    select = obs_dict["select"]
    if int(select.get("type", 0)) == 0 and int(select.get("context", 0)) == 0:
        return _main_action(obs_dict)
    return _select_effect(obs_dict)
