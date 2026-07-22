from __future__ import annotations

from collections import Counter
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .cards import DeckSpec
from .memory import GameMemory
from .model import Area, PlayerFacts, PokemonRef, ResourceLedger, TurnBudget, TurnFacts


def own_turn_number(current: Mapping[str, Any]) -> int:
    turn = int(current.get("turn", 0))
    your_index = int(current.get("yourIndex", 0))
    first_player = int(current.get("firstPlayer", 0))
    return (turn + (1 if your_index == first_player else 0)) // 2


def _card_ids(cards: Iterable[dict[str, Any] | None]) -> tuple[int, ...]:
    return tuple(
        int(card["id"])
        for card in cards
        if card is not None and card.get("id") is not None
    )


def _pokemon_ref(
    card: dict[str, Any],
    *,
    owner: int,
    area: Area,
    index: int,
    memory: GameMemory,
    own_turn: int,
) -> PokemonRef:
    serial = card.get("serial")
    serial = int(serial) if serial is not None else None
    previous = tuple(item for item in (card.get("preEvolution") or []) if isinstance(item, dict))
    return PokemonRef(
        card_id=int(card.get("id", -1)),
        serial=serial,
        owner=owner,
        area=area,
        index=index,
        hp=int(card.get("hp", 0)),
        max_hp=int(card.get("maxHp", card.get("hp", 0))),
        energy_types=tuple(int(value) for value in (card.get("energies") or [])),
        attached_energy_ids=_card_ids(card.get("energyCards") or []),
        appear_this_turn=bool(card.get("appearThisTurn", False)),
        pre_evolution_ids=_card_ids(previous),
        pre_evolution_serials=tuple(
            int(item["serial"]) for item in previous if item.get("serial") is not None
        ),
        can_evolve=memory.can_evolve(
            serial,
            bool(card.get("appearThisTurn", False)),
            own_turn,
        ),
    )


def _player_facts(
    player: dict[str, Any],
    *,
    index: int,
    your_index: int,
    memory: GameMemory,
    own_turn: int,
) -> PlayerFacts:
    active_cards = player.get("active") or []
    active = (
        _pokemon_ref(
            active_cards[0],
            owner=index,
            area=Area.ACTIVE,
            index=0,
            memory=memory,
            own_turn=own_turn,
        )
        if active_cards and active_cards[0]
        else None
    )
    bench = tuple(
        _pokemon_ref(
            card,
            owner=index,
            area=Area.BENCH,
            index=bench_index,
            memory=memory,
            own_turn=own_turn,
        )
        for bench_index, card in enumerate(player.get("bench") or [])
        if card
    )
    return PlayerFacts(
        index=index,
        active=active,
        bench=bench,
        hand_ids=_card_ids(player.get("hand") or []) if index == your_index else (),
        hand_count=int(player.get("handCount", len(player.get("hand") or []))),
        deck_count=int(player.get("deckCount", 0)),
        discard_ids=_card_ids(player.get("discard") or []),
        prize_count=len(player.get("prize") or []),
        bench_max=int(player.get("benchMax", 5)),
    )


def _field_stack_ids(player: dict[str, Any]) -> tuple[int, ...]:
    result: list[int] = []
    for card in [*(player.get("active") or []), *(player.get("bench") or [])]:
        if not card:
            continue
        if card.get("id") is not None:
            result.append(int(card["id"]))
        result.extend(_card_ids(card.get("preEvolution") or []))
    return tuple(result)


def _readonly_counter(values: Iterable[int]) -> Mapping[int, int]:
    return MappingProxyType(dict(Counter(values)))


def _resource_ledger(player: dict[str, Any], deck_spec: DeckSpec) -> ResourceLedger:
    zones: dict[str, Mapping[int, int]] = {
        "hand": _readonly_counter(_card_ids(player.get("hand") or [])),
        "field": _readonly_counter(_field_stack_ids(player)),
        "discard": _readonly_counter(_card_ids(player.get("discard") or [])),
        "deck": _readonly_counter(_card_ids(player.get("deck") or [])),
        "prize": _readonly_counter(_card_ids(player.get("prize") or [])),
    }
    known_in_deck = zones["deck"]
    known_in_prize = zones["prize"]
    unknown = {
        card_id: max(
            0,
            deck_spec.count(card_id)
            - sum(zone.get(card_id, 0) for zone in zones.values()),
        )
        for card_id in deck_spec.card_ids
    }
    return ResourceLedger(
        visible_by_zone=MappingProxyType(zones),
        known_in_deck=known_in_deck,
        known_in_prize=known_in_prize,
        unknown_deck_or_prize=MappingProxyType(unknown),
    )


def build_turn_facts(
    obs: dict[str, Any], memory: GameMemory, deck_spec: DeckSpec
) -> TurnFacts:
    current = obs.get("current") or {}
    players = current.get("players") or []
    your_index = int(current.get("yourIndex", 0))
    opponent_index = 1 - your_index
    your_player = players[your_index] if your_index < len(players) else {}
    opponent_player = players[opponent_index] if opponent_index < len(players) else {}
    logs = list(obs.get("logs") or current.get("logs") or [])
    memory.sync(current, your_player, logs)
    own_turn = own_turn_number(current)
    return TurnFacts(
        turn=int(current.get("turn", 0)),
        own_turn=own_turn,
        your_index=your_index,
        opponent_index=opponent_index,
        first_player=int(current.get("firstPlayer", 0)),
        yours=_player_facts(
            your_player,
            index=your_index,
            your_index=your_index,
            memory=memory,
            own_turn=own_turn,
        ),
        opponent=_player_facts(
            opponent_player,
            index=opponent_index,
            your_index=your_index,
            memory=memory,
            own_turn=own_turn,
        ),
        budget=TurnBudget(
            supporter_used=memory.supporter_used,
            stadium_used=memory.stadium_used,
            energy_used=memory.energy_used,
            retreat_used=memory.retreat_used,
            attack_submitted=memory.attack_submitted,
        ),
        resources=_resource_ledger(your_player, deck_spec),
        item_lock=memory.item_lock_turn_key == memory.turn_key,
        previous_opponent_turn_had_ko=memory.post_ko_turn_key == memory.turn_key,
        stadium_ids=_card_ids(current.get("stadium") or []),
        logs=tuple(logs),
    )
