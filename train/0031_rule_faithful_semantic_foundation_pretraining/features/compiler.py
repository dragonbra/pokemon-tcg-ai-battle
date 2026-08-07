"""Compile truthful current facts and typed relations from an official observation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..contracts.fields import ACTOR_KEYS, SCHEMA_VERSION
from ..domain.prototypes import FieldState, PrototypeIndex
from ..knowledge.state import CausalSnapshot
from .relations import card_id, integer


ZONE = {
    "self_active": 1, "self_bench": 2, "self_hand": 3, "self_discard": 4,
    "opponent_active": 5, "opponent_bench": 6, "opponent_discard": 7,
    "stadium": 8, "looking": 9, "select_deck": 10,
    "self_energy": 11, "opponent_energy": 12, "self_tool": 13,
    "opponent_tool": 14, "self_evolution": 15, "opponent_evolution": 16,
    "known_opponent_hand": 17,
    "self_playing": 18, "opponent_playing": 19,
    "self_resolved_energy": 20, "opponent_resolved_energy": 21,
    "known_self_deck_order": 22, "remembered_opponent_hidden": 23,
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


def _categorical_int(item: Mapping[str, Any], name: str) -> int:
    value = item.get(name)
    return int(value) + 1 if isinstance(value, int) and not isinstance(value, bool) else 0


def _categorical_bool(item: Mapping[str, Any], name: str) -> int:
    value = item.get(name)
    return int(value) + 1 if isinstance(value, bool) else 0


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
    raw_cards: list[Mapping[str, Any]] = []
    locations: dict[tuple[int, int, int], int] = {}
    serial_locations: dict[int, int] = {}
    child_locations: dict[tuple[int, str, int], int] = {}

    def add_card(raw: Any, *, owner: int, zone: int, slot: int, kind: int,
                 status: int = 0, parent: int = -1,
                 key: tuple[int, int, int] | None = None,
                 identity_knowledge: int = 1) -> int:
        identity = card_id(raw)
        if identity <= 0:
            return -1
        if not 0 <= slot < 256:
            raise ValueError(f"card zone slot outside exact categorical range: {slot}")
        item = raw if isinstance(raw, Mapping) else {}
        serial, serial_state = _raw_int(item, "serial")
        hp, hp_state = _raw_int(item, "hp")
        maximum_hp, max_hp_state = _raw_int(item, "maxHp")
        energy_cards = _items(item.get("energyCards"))
        energies = _items(item.get("energies"))
        tools = _items(item.get("tools"))
        evolution = _items(item.get("preEvolution"))
        appeared = item.get("appearThisTurn", item.get("appear"))
        appeared_present = isinstance(appeared, bool)
        is_pokemon = kind == 2
        index = len(card_cat)
        card_cat.append([
            identity, int(serial) + 1 if serial_state == int(FieldState.PRESENT) else 0,
            owner, zone, slot + 1, kind, status, 0,
            identity_knowledge,
        ])
        card_num.append([
            hp, maximum_hp, float(len(energy_cards)), float(len(energies)),
            float(len(tools)), float(len(evolution)),
            float(bool(appeared)) if appeared_present else 0.0,
        ])
        card_state.append([
            hp_state if is_pokemon else _state(False, False),
            max_hp_state if is_pokemon else _state(False, False),
            _state(True) if is_pokemon else _state(False, False),
            _state(True) if is_pokemon else _state(False, False),
            _state(True) if is_pokemon else _state(False, False),
            _state(True) if is_pokemon else _state(False, False),
            _state(appeared_present) if is_pokemon else _state(False, False),
        ])
        card_parent.append(parent + 1)
        raw_cards.append(item)
        if key is not None:
            locations[key] = index
        if serial_state == int(FieldState.PRESENT):
            serial_locations[int(serial)] = index
        return index

    def add_resolved_energy_unit(*, owner: int, parent: int, slot: int, value: Any) -> int:
        energy_type = integer(value, -1)
        if not 0 <= energy_type < 16:
            raise ValueError(f"resolved EnergyTypeIndex outside audited range: {value!r}")
        if not 0 <= slot < 256:
            raise ValueError(f"resolved Energy slot outside exact categorical range: {slot}")
        prefix = "self" if owner == 1 else "opponent"
        index = len(card_cat)
        card_cat.append([
            0, 0, owner, ZONE[f"{prefix}_resolved_energy"], slot + 1,
            9, 0, energy_type + 1, 4,
        ])
        card_num.append([0.0] * 7)
        card_state.append([int(FieldState.NOT_APPLICABLE)] * 7)
        card_parent.append(parent + 1)
        raw_cards.append({})
        return index

    def add_player_zone(player_index: int, name: str, area: int) -> None:
        player = _player(current, player_index)
        relative = _relative_owner(player_index, actor)
        prefix = "self" if relative == 1 else "opponent"
        for slot, raw in enumerate(_items(player.get(name))):
            parent = add_card(
                raw, owner=relative, zone=ZONE[f"{prefix}_{name}"], slot=slot,
                kind=2 if name in {"active", "bench"} else 1,
                status=_status_bits(player) if name == "active" else 0,
                key=(player_index, area, slot),
            )
            if parent < 0 or name not in {"active", "bench"} or not isinstance(raw, Mapping):
                continue
            for child_slot, child in enumerate(_items(raw.get("energyCards"))):
                child_owner_index = integer(
                    child.get("playerIndex") if isinstance(child, Mapping) else None,
                    player_index,
                )
                child_relative = _relative_owner(child_owner_index, actor)
                child_prefix = "self" if child_relative == 1 else "opponent"
                child_index = add_card(
                    child, owner=child_relative,
                    zone=ZONE[f"{child_prefix}_energy"], slot=child_slot,
                    kind=3, parent=parent,
                )
                child_locations[(parent, "energy", child_slot)] = child_index
            for field, suffix, kind in (("tools", "tool", 4), ("preEvolution", "evolution", 5)):
                for child_slot, child in enumerate(_items(raw.get(field))):
                    child_owner_index = integer(
                        child.get("playerIndex") if isinstance(child, Mapping) else None,
                        player_index,
                    )
                    child_relative = _relative_owner(child_owner_index, actor)
                    child_prefix = "self" if child_relative == 1 else "opponent"
                    child_index = add_card(
                        child, owner=child_relative,
                        zone=ZONE[f"{child_prefix}_{suffix}"], slot=child_slot,
                        kind=kind, parent=parent,
                    )
                    child_locations[(parent, field, child_slot)] = child_index
            for unit_slot, energy_type in enumerate(_items(raw.get("energies"))):
                add_resolved_energy_unit(
                    owner=relative, parent=parent, slot=unit_slot, value=energy_type
                )

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
                 slot=slot, kind=6, key=(-1, 7, slot))
    for slot, raw in enumerate(_items(current.get("looking"))):
        add_card(raw, owner=1, zone=ZONE["looking"], slot=slot, kind=7, key=(actor, 12, slot))
    for slot, raw in enumerate(_items(select.get("deck"))):
        add_card(raw, owner=1, zone=ZONE["select_deck"], slot=slot, kind=1, key=(actor, 1, slot))
    if not _items(select.get("deck")) and snapshot.deck_order_known:
        for slot, known in enumerate(snapshot.known_self_deck_order):
            if known.serial in serial_locations:
                continue
            add_card(
                {"id": known.card_id, "serial": known.serial}, owner=1,
                zone=ZONE["known_self_deck_order"], slot=slot, kind=1,
                identity_knowledge=2,
            )
    select_serials = {
        integer(value.get("serial"), -1)
        for value in (select.get("contextCard"), select.get("effect"))
        if isinstance(value, Mapping)
    }
    for slot, known in enumerate(snapshot.known_opponent_hand):
        if known.serial in serial_locations:
            continue
        add_card({"id": known.card_id, "serial": known.serial}, owner=2,
                 zone=ZONE["known_opponent_hand"], slot=slot, kind=8,
                 identity_knowledge=2)
    for slot, known in enumerate(snapshot.possible_opponent_hand):
        if known.serial in serial_locations:
            continue
        add_card(
            {"id": known.card_id, "serial": known.serial}, owner=2,
            zone=ZONE["known_opponent_hand"], slot=slot, kind=8,
            identity_knowledge=3,
        )
    for slot, known in enumerate(snapshot.remembered_opponent_cards):
        if known.serial in serial_locations or known.serial in select_serials:
            continue
        add_card(
            {"id": known.card_id, "serial": known.serial}, owner=2,
            zone=ZONE["remembered_opponent_hidden"], slot=slot, kind=10,
            identity_knowledge=2,
        )

    def ensure_select_card(raw: Any) -> int:
        if not isinstance(raw, Mapping) or card_id(raw) <= 0:
            return -1
        serial = integer(raw.get("serial"), -1)
        existing = serial_locations.get(serial, -1)
        if existing >= 0:
            return existing
        player_index = integer(raw.get("playerIndex"), actor)
        relative = _relative_owner(player_index, actor)
        prefix = "self" if relative == 1 else "opponent"
        return add_card(
            raw, owner=relative, zone=ZONE[f"{prefix}_playing"], slot=0, kind=7
        )

    context_index = ensure_select_card(select.get("contextCard"))
    effect_card_index = ensure_select_card(select.get("effect"))

    resource_cat: list[list[int]] = []
    resource_num: list[list[float]] = []
    resource_state: list[list[int]] = []
    for identity, entry in sorted(snapshot.self_ledger.items()):
        visible = entry.visible
        resource_cat.append([identity, _knowledge_code(entry.deck.state), _knowledge_code(entry.prize.state), int(snapshot.deck_order_known) + 1])
        resource_num.append([
            float(entry.initial), float(visible.get("active", 0)), float(visible.get("bench", 0)),
            float(visible.get("hand", 0)), float(visible.get("discard", 0)),
            float(visible.get("stadium", 0)), float(visible.get("playing", 0)),
            float(entry.deck.value or 0), float(entry.deck.lower), float(entry.deck.upper),
            float(entry.prize.value or 0), float(entry.prize.lower), float(entry.prize.upper),
            float(entry.deck.age), float(entry.prize.age),
        ])
        resource_state.append([
            int(FieldState.PRESENT), *([int(FieldState.PRESENT)] * 6),
            _state(entry.deck.value is not None), int(FieldState.PRESENT), int(FieldState.PRESENT),
            _state(entry.prize.value is not None), int(FieldState.PRESENT), int(FieldState.PRESENT),
            int(FieldState.PRESENT), int(FieldState.PRESENT),
        ])

    event_cat: list[list[int]] = []
    event_num: list[list[float]] = []
    event_state: list[list[int]] = []
    event_source: list[int] = []
    event_target: list[int] = []
    event_before: list[int] = []
    event_after: list[int] = []
    newest = snapshot.recent_events[-1].source_event if snapshot.recent_events else -1
    for event in snapshot.recent_events:
        p = event.payload
        relative = 3 if event.actor is None else _relative_owner(event.actor, actor)
        target_id = max(0, integer(p.get("cardIdTarget")))
        source_serial = integer(p.get("serial"), -1)
        target_serial = integer(p.get("serialTarget"), -1)
        active_serial = integer(p.get("serialActive"), -1)
        bench_serial = integer(p.get("serialBench"), -1)
        before_serial = integer(p.get("serialBefore"), -1)
        after_serial = integer(p.get("serialAfter"), -1)
        event_cat.append([
            event.log_type + 1, relative, int(event.card_id or 0), target_id,
            max(0, integer(p.get("attackId"))), int(event.from_area if event.from_area is not None else -1) + 1,
            int(event.to_area if event.to_area is not None else -1) + 1, int(event.identity_visible) + 1,
            int(isinstance(p.get("serial"), int)) + 1, int(isinstance(p.get("serialTarget"), int) or target_id > 0) + 1,
            max(0, integer(p.get("cardIdActive"))), max(0, integer(p.get("cardIdBench"))),
            max(0, integer(p.get("cardIdBefore"))), max(0, integer(p.get("cardIdAfter"))),
            _categorical_int(p, "serial"), _categorical_int(p, "serialTarget"),
            _categorical_int(p, "serialActive"), _categorical_int(p, "serialBench"),
            _categorical_int(p, "serialBefore"), _categorical_int(p, "serialAfter"),
            _categorical_bool(p, "isRecover"), _categorical_bool(p, "hasBasicPokemon"),
            _categorical_bool(p, "head"), _categorical_bool(p, "putDamageCounter"),
            _categorical_int(p, "result"), _categorical_int(p, "reason"),
            _categorical_int(p, "index"), _categorical_int(p, "energyIndex"),
            _categorical_int(p, "toolIndex"), _categorical_int(p, "inPlayArea"),
            _categorical_int(p, "inPlayIndex"),
        ])
        numeric_names = ("value", "count", "number")
        numeric = [_raw_int(p, name) for name in numeric_names]
        event_num.append([float(newest - event.source_event), *[value for value, _ in numeric]])
        event_state.append([int(FieldState.PRESENT), *[state for _, state in numeric]])
        if event.name == "switch":
            source_serial, target_serial = active_serial, bench_serial
        event_source.append(serial_locations.get(source_serial, -1) + 1)
        event_target.append(serial_locations.get(target_serial, -1) + 1)
        event_before.append(serial_locations.get(before_serial, -1) + 1)
        event_after.append(serial_locations.get(after_serial, -1) + 1)

    context_card, effect_card = card_id(select.get("contextCard")), card_id(select.get("effect"))
    option_cat: list[list[int]] = []
    option_num: list[list[float]] = []
    option_state: list[list[int]] = []
    option_source: list[int] = []
    option_target: list[int] = []
    option_context: list[int] = []
    option_effect_card: list[int] = []
    skill_ids: list[int] = []
    skill_roles: list[int] = []
    skill_parents: list[int] = []
    effect_ids: list[int] = []
    effect_roles: list[int] = []
    effect_parents: list[int] = []

    def related_skills(identity: int) -> list[tuple[int, int]]:
        card = prototypes.engine_cards.get(identity, {})
        return [(integer(card.get(name)), role) for role, name in enumerate(
            ("ability_skill_id", "play_skill_id", "delay_skill_id"), 1
        ) if integer(card.get(name)) > 0]

    for option_index, value in enumerate(options):
        option = value if isinstance(value, Mapping) else {}
        action_type = integer(option.get("type"), -1)
        source_player = integer(option.get("playerIndex"), actor)
        source_area = integer(option.get("area"), -1)
        if action_type == 7 and source_area < 0:
            source_area = 2
        source_slot = integer(option.get("index"), -1)
        source_index = locations.get((source_player, source_area, source_slot), -1)
        if source_area == 7:
            source_index = locations.get((-1, 7, source_slot), source_index)
        if action_type == 15:
            source_index = serial_locations.get(integer(option.get("serial"), -1), source_index)
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
            card_cat[source_index][2]
            if source_index >= 0 else _relative_owner(source_player, actor)
        )

        target_player = integer(
            option.get("inPlayPlayerIndex", option.get("targetPlayerIndex")), -1
        )
        if target_player < 0 and option.get("inPlayArea") is not None:
            target_player = actor
        target_area, target_slot = integer(option.get("inPlayArea"), -1), integer(option.get("inPlayIndex"), -1)
        target_index = locations.get((target_player, target_area, target_slot), -1)
        target_identity = card_cat[target_index][0] if target_index >= 0 else 0
        target_owner = card_cat[target_index][2] if target_index >= 0 else 3
        attack_id = max(0, integer(option.get("attackId")))
        attack = prototypes.engine_attacks.get(attack_id)
        number = _raw_int(option, "number")
        count = _raw_int(option, "count")
        option_cat.append([
            action_type + 1, source_owner, source_area + 1,
            target_owner, target_area + 1, source_identity,
            target_identity, attack_id, integer(option.get("specialConditionType"), -1) + 1,
            integer(select.get("type"), -1) + 1, integer(select.get("context"), -1) + 1,
            context_card, effect_card,
            option_index + 1, source_slot + 1, target_slot + 1,
            energy_index + 1, tool_index + 1,
            integer(option.get("serial"), -1) + 1,
        ])
        facts = (number, count)
        option_num.append([fact[0] for fact in facts])
        option_state.append([fact[1] for fact in facts])
        option_source.append(source_index + 1)
        option_target.append(target_index + 1)
        option_context.append(context_index + 1)
        option_effect_card.append(effect_card_index + 1)

        candidates: list[tuple[int, int]] = []
        for relation_role, identity in ((1, source_identity), (2, context_card), (3, effect_card)):
            candidates.extend((skill_id, (relation_role - 1) * 3 + role)
                              for skill_id, role in related_skills(identity))
        seen: set[tuple[int, int]] = set()
        for skill_id, role in candidates:
            if (skill_id, role) in seen:
                continue
            seen.add((skill_id, role))
            skill_ids.append(skill_id); skill_roles.append(role); skill_parents.append(option_index + 1)
            for effect_ref in prototypes.skill_effect_refs.get(skill_id, ()):
                effect_ids.append(effect_ref); effect_roles.append(1); effect_parents.append(option_index + 1)
        for effect_ref in prototypes.attack_effect_refs.get(attack_id, ()):
            effect_ids.append(effect_ref); effect_roles.append(2); effect_parents.append(option_index + 1)

    global_names = (
        (current, "turn"), (current, "turnActionCount"), (own, "deckCount"), (other, "deckCount"),
        (own, "handCount"), (other, "handCount"),
    )
    global_facts = [_raw_int(container, name) for container, name in global_names]
    global_facts.extend([
        (float(len(_items(own.get("prize")))), int(FieldState.PRESENT)),
        (float(len(_items(other.get("prize")))), int(FieldState.PRESENT)),
        (float(len(_items(own.get("bench")))), int(FieldState.PRESENT)),
        (float(len(_items(other.get("bench")))), int(FieldState.PRESENT)),
        (float(len(options)), int(FieldState.PRESENT)), (float(minimum), int(FieldState.PRESENT)),
        (float(maximum), int(FieldState.PRESENT)), _raw_int(select, "remainDamageCounter"),
        _raw_int(select, "remainEnergyCost"),
        (float(len(snapshot.known_opponent_hand)), int(FieldState.PRESENT)),
        (float(snapshot.unknown_opponent_hand), int(FieldState.PRESENT)),
        _raw_int(own, "benchMax"), _raw_int(other, "benchMax"),
    ])
    looking = current.get("looking")
    looking_known = isinstance(looking, (list, tuple))
    looking_values = list(looking) if looking_known else []
    looking_visible = sum(isinstance(item, Mapping) for item in looking_values)
    global_facts.extend([
        (float(len(looking_values)), _state(looking_known)),
        (float(looking_visible), _state(looking_known)),
        (float(len(snapshot.possible_opponent_hand)), int(FieldState.PRESENT)),
        (float(snapshot.possible_opponent_hand_known_lower), int(FieldState.PRESENT)),
        (float(snapshot.possible_opponent_hand_known_upper), int(FieldState.PRESENT)),
    ])
    if not looking_known:
        looking_visibility = 1
    elif any(item is None for item in looking_values):
        looking_visibility = 2
    else:
        looking_visibility = 3
    actor_record = {
        "global_cat": [
            integer(select.get("type"), -1) + 1, integer(select.get("context"), -1) + 1,
            1 if integer(current.get("firstPlayer"), -1) == actor else 2,
            int(bool(current.get("supporterPlayed"))) + 1, int(bool(current.get("stadiumPlayed"))) + 1,
            int(bool(current.get("energyAttached"))) + 1, int(bool(current.get("retreated"))) + 1,
            _status_bits(own) + 1, _status_bits(other) + 1,
            int(snapshot.deck_membership_known) + 1, int(snapshot.deck_order_known) + 1,
            looking_visibility,
        ],
        "global_num": [value for value, _ in global_facts],
        "global_state": [state for _, state in global_facts],
        "card_cat": card_cat, "card_num": card_num, "card_state": card_state, "card_parent": card_parent,
        "resource_cat": resource_cat, "resource_num": resource_num, "resource_state": resource_state,
        "event_cat": event_cat, "event_num": event_num, "event_state": event_state,
        "event_source": event_source, "event_target": event_target,
        "event_before": event_before, "event_after": event_after,
        "option_cat": option_cat, "option_num": option_num, "option_state": option_state,
        "option_source": option_source, "option_target": option_target,
        "option_context": option_context, "option_effect_card": option_effect_card,
        "option_skill_id": skill_ids, "option_skill_role": skill_roles, "option_skill_parent": skill_parents,
        "option_effect_id": effect_ids, "option_effect_role": effect_roles, "option_effect_parent": effect_parents,
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


def compile_canonical_layers(
    row: Mapping[str, Any], snapshot: CausalSnapshot, prototypes: PrototypeIndex
) -> dict[str, Any]:
    """Compile through the explicit research layers without changing the hot path."""
    from .layers import (
        assemble_canonical_record,
        compile_card_layer,
        compile_event_layer,
        compile_global_layer,
        compile_option_layer,
        compile_resource_layer,
    )

    cards = compile_card_layer(row, snapshot, prototypes)
    resources = compile_resource_layer(row, snapshot)
    events = compile_event_layer(row, snapshot, cards)
    options = compile_option_layer(row, snapshot, prototypes, cards)
    globals_ = compile_global_layer(row, snapshot)
    return assemble_canonical_record(row, cards, resources, events, options, globals_)


__all__ = ["compile_canonical_layers", "compile_canonical_row"]
