"""Fail-closed audits for canonical features against their source observation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from ..domain.prototypes import FieldState
from ..knowledge.state import CausalSnapshot
from .compiler import ZONE
from .relations import card_id, integer


ATTACH_ACTION_TYPE = 8
ENERGY_TYPE_COUNT = 12
CHILD_ZONES = frozenset({ZONE["energy"], ZONE["tool"], ZONE["evolution"]})
IN_PLAY_ZONES = frozenset({ZONE["active"], ZONE["bench"]})


def _items(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _player(current: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    players = current.get("players")
    if (
        not isinstance(players, Sequence)
        or len(players) != 2
        or not isinstance(players[index], Mapping)
    ):
        raise ValueError("feature audit requires two player states")
    return players[index]


def _relative_owner(player_index: int, actor: int) -> int:
    return 1 if player_index == actor else (2 if player_index in (0, 1) else 3)


def _trace_card_instances(
    source: Mapping[str, Any],
    actor_record: Mapping[str, Any],
    snapshot: CausalSnapshot,
) -> tuple[
    dict[tuple[int, int, int], int],
    dict[int, int],
    dict[tuple[int, str, int], int],
]:
    """Rebuild actor-visible instance addresses and compare every token in exact order."""
    observation = source["actor_observation"]
    current = observation["current"]
    select = observation["select"]
    actor = int(current["yourIndex"])
    card_cat = actor_record["card_cat"]
    card_parent = actor_record["card_parent"]
    locations: dict[tuple[int, int, int], int] = {}
    serial_locations: dict[int, int] = {}
    child_locations: dict[tuple[int, str, int], int] = {}
    cursor = 0

    def add_card(
        raw: Any,
        *,
        owner: int,
        zone: int,
        slot: int,
        parent: int = 0,
        key: tuple[int, int, int] | None = None,
    ) -> int:
        nonlocal cursor
        identity = card_id(raw)
        if identity <= 0:
            return 0
        if cursor >= len(card_cat):
            raise ValueError("compiled card sequence ended before the observation")
        expected = [identity, owner, zone, min(slot + 1, 129)]
        if card_cat[cursor][:4] != expected:
            raise ValueError(
                "compiled card instance identity/owner/zone/slot differs from observation"
            )
        if card_parent[cursor] != parent:
            raise ValueError("compiled card_parent differs from the exact observation parent")
        relation = cursor + 1
        if key is not None:
            locations[key] = relation
        item = raw if isinstance(raw, Mapping) else {}
        serial = item.get("serial")
        if isinstance(serial, int) and not isinstance(serial, bool):
            serial_locations[serial] = relation
        cursor += 1
        return relation

    for player_index in (actor, 1 - actor):
        player = _player(current, player_index)
        owner = _relative_owner(player_index, actor)
        for name, area in (("active", 4), ("bench", 5)):
            for slot, raw in enumerate(_items(player.get(name))):
                parent = add_card(
                    raw,
                    owner=owner,
                    zone=ZONE[name],
                    slot=slot,
                    key=(player_index, area, slot),
                )
                if parent <= 0 or not isinstance(raw, Mapping):
                    continue
                for child_slot, child in enumerate(_items(raw.get("energyCards"))):
                    relation = add_card(
                        child,
                        owner=owner,
                        zone=ZONE["energy"],
                        slot=child_slot,
                        parent=parent,
                    )
                    child_locations[(parent, "energy", child_slot)] = relation
                for field, zone_name in (("tools", "tool"), ("preEvolution", "evolution")):
                    for child_slot, child in enumerate(_items(raw.get(field))):
                        relation = add_card(
                            child,
                            owner=owner,
                            zone=ZONE[zone_name],
                            slot=child_slot,
                            parent=parent,
                        )
                        child_locations[(parent, field, child_slot)] = relation
        if player_index == actor:
            for slot, raw in enumerate(_items(player.get("hand"))):
                add_card(
                    raw,
                    owner=owner,
                    zone=ZONE["hand"],
                    slot=slot,
                    key=(player_index, 2, slot),
                )
        for slot, raw in enumerate(_items(player.get("discard"))):
            add_card(
                raw,
                owner=owner,
                zone=ZONE["discard"],
                slot=slot,
                key=(player_index, 3, slot),
            )

    for slot, raw in enumerate(_items(current.get("stadium"))):
        item = raw if isinstance(raw, Mapping) else {}
        add_card(
            raw,
            owner=_relative_owner(integer(item.get("playerIndex"), -1), actor),
            zone=ZONE["stadium"],
            slot=slot,
            key=(-1, 7, slot),
        )
    for slot, raw in enumerate(_items(current.get("looking"))):
        add_card(
            raw,
            owner=1,
            zone=ZONE["looking"],
            slot=slot,
            key=(actor, 12, slot),
        )
    for slot, raw in enumerate(_items(select.get("deck"))):
        add_card(
            raw,
            owner=1,
            zone=ZONE["select_deck"],
            slot=slot,
            key=(actor, 1, slot),
        )
    for slot, known in enumerate(snapshot.known_opponent_hand):
        if known.serial in serial_locations:
            continue
        add_card(
            {"id": known.card_id, "serial": known.serial},
            owner=2,
            zone=ZONE["known_opponent_hand"],
            slot=slot,
        )
    if cursor != len(card_cat):
        raise ValueError("compiled card sequence has instances absent from actor-visible state")
    return locations, serial_locations, child_locations


def _expected_histogram(values: list[Any]) -> list[float]:
    counts = Counter(int(value) for value in values)
    invalid = sorted(value for value in counts if not 0 <= value < ENERGY_TYPE_COUNT)
    if invalid:
        raise ValueError(f"invalid observed EnergyTypeIndex values: {invalid}")
    return [float(counts[index]) for index in range(ENERGY_TYPE_COUNT)]


def audit_compiled_feature_input(
    source: Mapping[str, Any],
    record: Mapping[str, Any],
    snapshot: CausalSnapshot,
) -> dict[str, int]:
    """Verify every retained dynamic relation and resolved Energy fact before shard write."""
    actor_record = record["actor"]
    observation = source["actor_observation"]
    current = observation["current"]
    actor = int(current["yourIndex"])
    card_cat = actor_record["card_cat"]
    card_num = actor_record["card_num"]
    card_state = actor_record["card_state"]
    card_parent = actor_record["card_parent"]
    locations, serial_locations, child_locations = _trace_card_instances(
        source,
        actor_record,
        snapshot,
    )

    in_play_count = 0
    for player_index in (actor, 1 - actor):
        relative_owner = 1 if player_index == actor else 2
        player = current["players"][player_index]
        for zone_name in ("active", "bench"):
            zone = ZONE[zone_name]
            for slot, pokemon in enumerate(player.get(zone_name) or []):
                if (
                    not isinstance(pokemon, Mapping)
                    or isinstance(pokemon.get("id"), bool)
                    or not isinstance(pokemon.get("id"), int)
                    or int(pokemon["id"]) <= 0
                ):
                    continue
                matches = [
                    index
                    for index, values in enumerate(card_cat)
                    if values[1] == relative_owner
                    and values[2] == zone
                    and values[3] == slot + 1
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"in-play Pokemon token mismatch: player={player_index}, "
                        f"zone={zone_name}, slot={slot}, matches={matches}"
                    )
                index = matches[0]
                expected = _expected_histogram(pokemon.get("energies") or [])
                if card_num[index][2:] != expected:
                    raise ValueError(
                        "compiled resolved-Energy histogram differs from observation"
                    )
                if card_state[index][2:] != [int(FieldState.PRESENT)] * ENERGY_TYPE_COUNT:
                    raise ValueError("in-play Pokemon Energy histogram is not marked PRESENT")
                in_play_count += 1

    child_count = 0
    current_parent = 0
    for index, (values, states, parent) in enumerate(
        zip(card_cat, card_state, card_parent, strict=True)
    ):
        if values[2] in IN_PLAY_ZONES:
            current_parent = index + 1
            if parent != 0:
                raise ValueError("in-play Pokemon cannot itself have card_parent")
            continue
        if values[2] not in CHILD_ZONES:
            current_parent = 0
            continue
        child_count += 1
        if parent != current_parent or current_parent <= 0:
            raise ValueError("attached child relation does not point to its exact Pokemon")
        if card_cat[parent - 1][1] != values[1]:
            raise ValueError("attached child and parent have different owners")
        if (
            values[2] == ZONE["energy"]
            and states[2:] != [int(FieldState.NOT_APPLICABLE)] * ENERGY_TYPE_COUNT
        ):
            raise ValueError("physical Energy card has an applicable Pokemon histogram")

    relation_count = 0
    for relation_name, identity_column in (("option_source", 5), ("option_target", 6)):
        for option_index, relation in enumerate(actor_record[relation_name]):
            if relation <= 0:
                continue
            relation_count += 1
            if relation > len(card_cat):
                raise ValueError(f"{relation_name} points outside the card sequence")
            related_card_id = card_cat[relation - 1][0]
            option_card_id = actor_record["option_cat"][option_index][identity_column]
            if related_card_id != option_card_id:
                raise ValueError(f"{relation_name} card identity disagrees with the option")

    attach_count = 0
    exact_option_relations = 0
    opponent = 1 - actor
    for index, raw_option in enumerate(observation["select"]["option"]):
        option = raw_option if isinstance(raw_option, Mapping) else {}
        action_type = integer(option.get("type"), -1)
        source_player = integer(option.get("playerIndex"), actor)
        source_area = integer(option.get("area"), -1)
        if action_type == 7 and source_area < 0:
            source_area = 2
        source_slot = integer(option.get("index"), -1)
        expected_source = locations.get((source_player, source_area, source_slot), 0)
        if expected_source <= 0:
            expected_source = serial_locations.get(integer(option.get("serial"), -1), 0)
        if source_area == 7:
            expected_source = locations.get((-1, 7, source_slot), expected_source)
        if action_type in {12, 13} and expected_source <= 0:
            expected_source = locations.get((actor, 4, 0), 0)
        parent_source = expected_source
        energy_index = integer(option.get("energyIndex"), -1)
        tool_index = integer(option.get("toolIndex"), -1)
        if energy_index >= 0 and parent_source > 0:
            expected_source = child_locations.get(
                (parent_source, "energy", energy_index), expected_source
            )
        elif tool_index >= 0 and parent_source > 0:
            expected_source = child_locations.get(
                (parent_source, "tools", tool_index), expected_source
            )

        target_player = integer(
            option.get("inPlayPlayerIndex", option.get("targetPlayerIndex")), actor
        )
        target_area = integer(option.get("inPlayArea"), -1)
        target_slot = integer(option.get("inPlayIndex"), -1)
        expected_target = locations.get((target_player, target_area, target_slot), 0)
        if action_type == 13 and expected_target <= 0:
            expected_target = locations.get((opponent, 4, 0), 0)

        if actor_record["option_source"][index] != expected_source:
            raise ValueError("option_source differs from the exact observation instance")
        if actor_record["option_target"][index] != expected_target:
            raise ValueError("option_target differs from the exact observation instance")
        exact_option_relations += int(expected_source > 0) + int(expected_target > 0)

        if action_type != ATTACH_ACTION_TYPE:
            continue
        attach_count += 1
        if actor_record["option_source"][index] <= 0:
            raise ValueError("Attach option has no source Energy relation")
        if actor_record["option_target"][index] <= 0:
            raise ValueError("Attach option has no target Pokemon relation")

    if len(snapshot.recent_events) != len(actor_record["event_cat"]):
        raise ValueError("compiled event sequence length differs from the causal snapshot")
    exact_event_relations = 0
    for index, event in enumerate(snapshot.recent_events):
        payload = event.payload
        source_serial = integer(
            payload.get("serial", payload.get("serialActive", payload.get("serialBefore"))),
            -1,
        )
        target_serial = integer(
            payload.get(
                "serialTarget",
                payload.get("serialBench", payload.get("serialAfter")),
            ),
            -1,
        )
        expected_source = serial_locations.get(source_serial, 0)
        expected_target = serial_locations.get(target_serial, 0)
        if actor_record["event_source"][index] != expected_source:
            raise ValueError("event_source differs from the exact visible serial instance")
        if actor_record["event_target"][index] != expected_target:
            raise ValueError("event_target differs from the exact visible serial instance")
        exact_event_relations += int(expected_source > 0) + int(expected_target > 0)

    return {
        "audited_decisions": 1,
        "in_play_pokemon": in_play_count,
        "attached_children": child_count,
        "attach_options": attach_count,
        "nonzero_option_relations": relation_count,
        "exact_option_relations": exact_option_relations,
        "exact_event_relations": exact_event_relations,
    }


__all__ = ["audit_compiled_feature_input"]
