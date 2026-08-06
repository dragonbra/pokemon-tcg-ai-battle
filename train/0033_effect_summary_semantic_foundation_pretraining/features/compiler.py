"""Compile truthful current facts and typed relations from an official observation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..contracts.fields import ACTOR_KEYS, ENERGY_TYPE_COUNT, SCHEMA_VERSION
from ..domain.prototypes import FieldState, PrototypeIndex
from ..knowledge.state import CausalSnapshot
from .relations import card_id, integer


ZONE = {
    "active": 1,
    "bench": 2,
    "hand": 3,
    "discard": 4,
    "stadium": 5,
    "looking": 6,
    "select_deck": 7,
    "energy": 8,
    "tool": 9,
    "evolution": 10,
    "known_opponent_hand": 11,
}


def _items(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _player(current: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    players = current.get("players")
    if not isinstance(players, Sequence) or len(players) != 2 or not isinstance(players[index], Mapping):
        raise ValueError("canonical observation requires two player states")
    return players[index]


def _status_bits(player: Mapping[str, Any]) -> int:
    return sum(1 << index for index, name in enumerate(
        ("asleep", "burned", "confused", "paralyzed", "poisoned")
    ) if player.get(name))


def _relative_owner(player_index: int, actor: int) -> int:
    return 1 if player_index == actor else (2 if player_index in (0, 1) else 3)


def _knowledge_code(value: Any) -> int:
    states = ("unobserved", "visible_now", "remembered", "inferred_exact", "bounded", "unknown")
    raw = getattr(value, "value", value)
    return states.index(raw) + 1 if raw in states else 0


def _deck(row: Mapping[str, Any]) -> list[int]:
    output: list[int] = []
    for identity, count in row["deck_manifest"]["counts"]:
        output.extend([int(identity)] * int(count))
    if len(output) != 60:
        raise ValueError("canonical deck manifest must contain exactly 60 cards")
    return output


def _state(present: bool, applicable: bool = True) -> int:
    if not applicable:
        return int(FieldState.NOT_APPLICABLE)
    return int(FieldState.PRESENT if present else FieldState.UNKNOWN)


def _raw_int(item: Mapping[str, Any], name: str) -> tuple[float, int]:
    value = item.get(name)
    present = isinstance(value, (int, float)) and not isinstance(value, bool)
    return (float(value) if present else 0.0, _state(present))


def _categorical_bool(item: Mapping[str, Any], name: str) -> int:
    value = item.get(name)
    return int(value) + 1 if isinstance(value, bool) else 0


def _categorical_int(item: Mapping[str, Any], name: str, *, maximum: int) -> int:
    value = item.get(name)
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < maximum:
        return value + 1
    return 0


def _energy_histogram(values: Sequence[Any]) -> list[float]:
    """Count the engine's exact EnergyTypeIndex values without collapsing multiplicity."""
    histogram = [0.0] * ENERGY_TYPE_COUNT
    for value in values:
        code = integer(value, -1)
        if not 0 <= code < ENERGY_TYPE_COUNT:
            raise ValueError(f"observation contains invalid EnergyTypeIndex: {value!r}")
        histogram[code] += 1.0
    return histogram


def compile_canonical_row(
    row: Mapping[str, Any], snapshot: CausalSnapshot, prototypes: PrototypeIndex
) -> dict[str, Any]:
    observation = row.get("actor_observation")
    if not isinstance(observation, Mapping):
        raise ValueError("canonical row has no actor observation")
    current, select = observation.get("current"), observation.get("select")
    if not isinstance(current, Mapping) or not isinstance(select, Mapping):
        raise ValueError("canonical observation has no current/select state")
    actor = integer(current.get("yourIndex"), -1)
    if actor not in (0, 1) or snapshot.perspective_actor != actor:
        raise ValueError("canonical actor and causal snapshot disagree")
    opponent = 1 - actor
    own, other = _player(current, actor), _player(current, opponent)
    options = _items(select.get("option"))
    action = [integer(value, -1) for value in _items(row.get("ordered_action"))]
    minimum = integer(select.get("minCount"), 0)
    maximum = integer(select.get("maxCount"), len(options))
    if not options or not 0 <= minimum <= maximum <= len(options):
        raise ValueError("invalid canonical legal-option bounds")
    if len(set(action)) != len(action) or any(value < 0 or value >= len(options) for value in action):
        raise ValueError("canonical target is not a unique legal option sequence")
    if not minimum <= len(action) <= maximum:
        raise ValueError("canonical target violates selection bounds")

    card_cat: list[list[int]] = []
    card_num: list[list[float]] = []
    card_state: list[list[int]] = []
    card_parent: list[int] = []
    locations: dict[tuple[int, int, int], int] = {}
    serial_locations: dict[int, int] = {}
    child_locations: dict[tuple[int, str, int], int] = {}

    def add_card(
        raw: Any,
        *,
        owner: int,
        zone: int,
        slot: int,
        in_play_pokemon: bool = False,
        status: int = 0,
        parent: int = -1,
        key: tuple[int, int, int] | None = None,
    ) -> int:
        identity = card_id(raw)
        if identity <= 0:
            return -1
        item = raw if isinstance(raw, Mapping) else {}
        hp, hp_state = _raw_int(item, "hp")
        maximum_hp, max_hp_state = _raw_int(item, "maxHp")
        appeared = item.get("appearThisTurn", item.get("appear"))
        if in_play_pokemon:
            appeared_state = int(appeared) + 1 if isinstance(appeared, bool) else 3
        else:
            appeared_state = 4
        energies_present = in_play_pokemon and isinstance(item.get("energies"), (list, tuple))
        histogram = _energy_histogram(_items(item.get("energies"))) if energies_present else [0.0] * ENERGY_TYPE_COUNT
        index = len(card_cat)
        card_cat.append([
            identity,
            owner,
            zone,
            min(slot + 1, 129),
            status + 1 if in_play_pokemon else 0,
            appeared_state,
        ])
        card_num.append([hp, maximum_hp, *histogram])
        histogram_state = (
            _state(True)
            if energies_present
            else (_state(False) if in_play_pokemon else _state(False, False))
        )
        card_state.append([
            hp_state if in_play_pokemon else _state(False, False),
            max_hp_state if in_play_pokemon else _state(False, False),
            *([histogram_state] * ENERGY_TYPE_COUNT),
        ])
        card_parent.append(parent + 1)
        if key is not None:
            locations[key] = index
        serial = item.get("serial")
        if isinstance(serial, int) and not isinstance(serial, bool):
            serial_locations[serial] = index
        return index

    def add_player_zone(player_index: int, name: str, area: int) -> None:
        player = _player(current, player_index)
        relative = _relative_owner(player_index, actor)
        for slot, raw in enumerate(_items(player.get(name))):
            parent = add_card(
                raw, owner=relative, zone=ZONE[name], slot=slot,
                in_play_pokemon=name in {"active", "bench"},
                status=_status_bits(player) if name == "active" else 0,
                key=(player_index, area, slot),
            )
            if parent < 0 or name not in {"active", "bench"} or not isinstance(raw, Mapping):
                continue
            for child_slot, child in enumerate(_items(raw.get("energyCards"))):
                child_index = add_card(
                    child, owner=relative, zone=ZONE["energy"], slot=child_slot,
                    parent=parent,
                )
                child_locations[(parent, "energy", child_slot)] = child_index
            for field, suffix in (("tools", "tool"), ("preEvolution", "evolution")):
                for child_slot, child in enumerate(_items(raw.get(field))):
                    child_index = add_card(
                        child, owner=relative, zone=ZONE[suffix], slot=child_slot,
                        parent=parent,
                    )
                    child_locations[(parent, field, child_slot)] = child_index

    for player_index in (actor, opponent):
        add_player_zone(player_index, "active", 4)
        add_player_zone(player_index, "bench", 5)
        if player_index == actor:
            add_player_zone(player_index, "hand", 2)
        add_player_zone(player_index, "discard", 3)
    for slot, raw in enumerate(_items(current.get("stadium"))):
        item = raw if isinstance(raw, Mapping) else {}
        owner_index = integer(item.get("playerIndex"), -1)
        add_card(raw, owner=_relative_owner(owner_index, actor), zone=ZONE["stadium"],
                 slot=slot, key=(-1, 7, slot))
    for slot, raw in enumerate(_items(current.get("looking"))):
        add_card(raw, owner=1, zone=ZONE["looking"], slot=slot, key=(actor, 12, slot))
    for slot, raw in enumerate(_items(select.get("deck"))):
        add_card(raw, owner=1, zone=ZONE["select_deck"], slot=slot, key=(actor, 1, slot))
    for slot, known in enumerate(snapshot.known_opponent_hand):
        if known.serial in serial_locations:
            continue
        add_card({"id": known.card_id, "serial": known.serial}, owner=2,
                 zone=ZONE["known_opponent_hand"], slot=slot)

    resource_cat: list[list[int]] = []
    resource_num: list[list[float]] = []
    resource_state: list[list[int]] = []
    for identity, entry in sorted(snapshot.self_ledger.items()):
        visible = entry.visible
        resource_cat.append([
            identity,
            _knowledge_code(entry.deck.state),
            _knowledge_code(entry.prize.state),
        ])
        resource_num.append([
            float(entry.initial), float(visible.get("playing", 0)),
            float(entry.deck.value or 0), float(entry.deck.lower), float(entry.deck.upper),
            float(entry.prize.value or 0), float(entry.prize.lower), float(entry.prize.upper),
            float(entry.deck.age), float(entry.prize.age),
        ])
        resource_state.append([
            int(FieldState.PRESENT), int(FieldState.PRESENT),
            _state(entry.deck.value is not None), int(FieldState.PRESENT), int(FieldState.PRESENT),
            _state(entry.prize.value is not None), int(FieldState.PRESENT), int(FieldState.PRESENT),
            int(FieldState.PRESENT), int(FieldState.PRESENT),
        ])

    event_cat: list[list[int]] = []
    event_num: list[list[float]] = []
    event_state: list[list[int]] = []
    event_source: list[int] = []
    event_target: list[int] = []
    newest = snapshot.recent_events[-1].source_event if snapshot.recent_events else -1
    for event in snapshot.recent_events:
        p = event.payload
        relative = 3 if event.actor is None else _relative_owner(event.actor, actor)
        target_id = max(0, integer(p.get("cardIdTarget")))
        event_cat.append([
            event.log_type + 1, relative, int(event.card_id or 0), target_id,
            max(0, integer(p.get("attackId"))), int(event.from_area if event.from_area is not None else -1) + 1,
            int(event.to_area if event.to_area is not None else -1) + 1,
            max(0, integer(p.get("cardIdActive"))), max(0, integer(p.get("cardIdBench"))),
            max(0, integer(p.get("cardIdBefore"))), max(0, integer(p.get("cardIdAfter"))),
            _categorical_bool(p, "isRecover"),
            _categorical_bool(p, "putDamageCounter"),
            _categorical_bool(p, "head"),
            _categorical_int(p, "specialConditionType", maximum=33),
            _categorical_int(p, "result", maximum=129),
            _categorical_int(p, "reason", maximum=129),
        ])
        numeric_names = ("value", "count", "number")
        numeric = [_raw_int(p, name) for name in numeric_names]
        event_num.append([float(newest - event.source_event), *[value for value, _ in numeric]])
        event_state.append([int(FieldState.PRESENT), *[state for _, state in numeric]])
        source_serial = integer(
            p.get("serial", p.get("serialActive", p.get("serialBefore"))), -1
        )
        target_serial = integer(
            p.get("serialTarget", p.get("serialBench", p.get("serialAfter"))), -1
        )
        event_source.append(serial_locations.get(source_serial, -1) + 1)
        event_target.append(serial_locations.get(target_serial, -1) + 1)

    context_card, effect_card = card_id(select.get("contextCard")), card_id(select.get("effect"))
    option_cat: list[list[int]] = []
    option_num: list[list[float]] = []
    option_state: list[list[int]] = []
    option_source: list[int] = []
    option_target: list[int] = []
    for value in options:
        option = value if isinstance(value, Mapping) else {}
        action_type = integer(option.get("type"), -1)
        source_player = integer(option.get("playerIndex"), actor)
        source_area = integer(option.get("area"), -1)
        if action_type == 7 and source_area < 0:
            source_area = 2
        source_slot = integer(option.get("index"), -1)
        source_index = locations.get((source_player, source_area, source_slot), -1)
        if source_index < 0:
            source_index = serial_locations.get(integer(option.get("serial"), -1), -1)
        if source_area == 7:
            source_index = locations.get((-1, 7, source_slot), source_index)
        if action_type in {12, 13} and source_index < 0:
            source_player, source_area = actor, 4
            source_index = locations.get((actor, 4, 0), -1)
        parent_source = source_index
        energy_index = integer(option.get("energyIndex"), -1)
        tool_index = integer(option.get("toolIndex"), -1)
        if energy_index >= 0 and parent_source >= 0:
            source_index = child_locations.get((parent_source, "energy", energy_index), source_index)
        elif tool_index >= 0 and parent_source >= 0:
            source_index = child_locations.get((parent_source, "tools", tool_index), source_index)
        source_identity = card_id(option.get("cardId"))
        if source_index >= 0:
            source_identity = card_cat[source_index][0]
        source_owner = (
            card_cat[source_index][1]
            if source_index >= 0
            else (
                _relative_owner(source_player, actor)
                if source_area >= 0 or "playerIndex" in option
                else 0
            )
        )

        target_player = integer(
            option.get("inPlayPlayerIndex", option.get("targetPlayerIndex")), actor
        )
        target_area, target_slot = integer(option.get("inPlayArea"), -1), integer(option.get("inPlayIndex"), -1)
        target_index = locations.get((target_player, target_area, target_slot), -1)
        if action_type == 13 and target_index < 0:
            target_player, target_area, target_index = opponent, 4, locations.get((opponent, 4, 0), -1)
        target_identity = card_cat[target_index][0] if target_index >= 0 else 0
        target_owner = (
            card_cat[target_index][1]
            if target_index >= 0
            else (
                _relative_owner(target_player, actor)
                if target_area >= 0
                or "inPlayPlayerIndex" in option
                or "targetPlayerIndex" in option
                else 0
            )
        )
        attack_id = max(0, integer(option.get("attackId")))
        number = _raw_int(option, "number")
        count = _raw_int(option, "count")
        option_cat.append([
            action_type + 1, source_owner, source_area + 1,
            target_owner, target_area + 1, source_identity,
            target_identity, attack_id, integer(option.get("specialConditionType"), -1) + 1,
            context_card, effect_card,
        ])
        facts = (number, count)
        option_num.append([fact[0] for fact in facts])
        option_state.append([fact[1] for fact in facts])
        option_source.append(source_index + 1)
        option_target.append(target_index + 1)

    global_names = (
        (current, "turn"), (current, "turnActionCount"), (own, "deckCount"), (other, "deckCount"),
        (other, "handCount"),
    )
    global_facts = [_raw_int(container, name) for container, name in global_names]
    global_facts.extend([
        (float(len(_items(own.get("prize")))), int(FieldState.PRESENT)),
        (float(len(_items(other.get("prize")))), int(FieldState.PRESENT)),
        _raw_int(own, "benchMax"), _raw_int(other, "benchMax"),
        _raw_int(select, "remainDamageCounter"), _raw_int(select, "remainEnergyCost"),
    ])
    first_player = integer(current.get("firstPlayer"), -1)
    relative_first_player = 1 if first_player == actor else (2 if first_player == opponent else 0)
    actor_record = {
        "global_cat": [
            integer(select.get("type"), -1) + 1, integer(select.get("context"), -1) + 1,
            relative_first_player,
            _categorical_bool(current, "supporterPlayed"),
            _categorical_bool(current, "stadiumPlayed"),
            _categorical_bool(current, "energyAttached"),
            _categorical_bool(current, "retreated"),
        ],
        "global_num": [value for value, _ in global_facts],
        "global_state": [state for _, state in global_facts],
        "card_cat": card_cat, "card_num": card_num, "card_state": card_state, "card_parent": card_parent,
        "resource_cat": resource_cat, "resource_num": resource_num, "resource_state": resource_state,
        "event_cat": event_cat, "event_num": event_num, "event_state": event_state,
        "event_source": event_source, "event_target": event_target,
        "option_cat": option_cat, "option_num": option_num, "option_state": option_state,
        "option_source": option_source, "option_target": option_target,
        "min_count": minimum, "max_count": maximum,
    }
    if set(actor_record) != ACTOR_KEYS:
        raise AssertionError(f"canonical actor contract drift: {sorted(set(actor_record) ^ ACTOR_KEYS)}")
    return {
        "schema_version": SCHEMA_VERSION, "actor": actor_record,
        "target": {"ordered_action": action, "termination": row.get("action_termination")},
        "audit": {"identity": row.get("identity"), "split": row.get("split")},
        "registered_deck": _deck(row),
    }


__all__ = ["compile_canonical_row"]
