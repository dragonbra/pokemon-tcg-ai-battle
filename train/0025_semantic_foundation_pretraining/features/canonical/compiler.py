"""Compile the canonical semantic actor directly from an official observation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ...knowledge.state import CausalSnapshot
from ..prototypes import FieldState, PrototypeIndex
from .resolver import attack_facts, card_id, integer
from .schema import ACTOR_KEYS, SCHEMA_VERSION


ZONE = {
    "self_active": 1,
    "self_bench": 2,
    "self_hand": 3,
    "self_discard": 4,
    "opponent_active": 5,
    "opponent_bench": 6,
    "opponent_discard": 7,
    "stadium": 8,
    "looking": 9,
    "select_deck": 10,
    "self_energy": 11,
    "opponent_energy": 12,
    "self_tool": 13,
    "opponent_tool": 14,
    "self_evolution": 15,
    "opponent_evolution": 16,
}
AREA_TO_ZONE = {1: "select_deck", 2: "self_hand", 3: "self_discard", 4: "self_active", 5: "self_bench", 7: "stadium", 12: "looking"}


def _items(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _player(current: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    players = current.get("players")
    if not isinstance(players, Sequence) or len(players) != 2 or not isinstance(players[index], Mapping):
        raise ValueError("canonical observation requires two player states")
    return players[index]


def _status_bits(player: Mapping[str, Any]) -> int:
    return sum(
        1 << index
        for index, name in enumerate(("asleep", "burned", "confused", "paralyzed", "poisoned"))
        if player.get(name)
    )


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


def compile_canonical_row(
    row: Mapping[str, Any], snapshot: CausalSnapshot, prototypes: PrototypeIndex
) -> dict[str, Any]:
    observation = row.get("actor_observation")
    if not isinstance(observation, Mapping):
        raise ValueError("canonical row has no actor observation")
    current = observation.get("current")
    select = observation.get("select")
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
    card_parent: list[int] = []
    raw_cards: list[Mapping[str, Any]] = []
    locations: dict[tuple[int, int, int], int] = {}

    def add_card(
        raw: Any,
        *,
        owner: int,
        zone: int,
        kind: int,
        status: int = 0,
        parent: int = -1,
        key: tuple[int, int, int] | None = None,
    ) -> int:
        identity = card_id(raw)
        if identity <= 0:
            return -1
        item = raw if isinstance(raw, Mapping) else {}
        current_hp = float(item.get("hp", 0) or 0)
        maximum_hp = max(current_hp, float(item.get("maxHp", current_hp) or current_hp))
        energy = _items(item.get("energyCards", item.get("energies")))
        tools = _items(item.get("tools"))
        evolution = _items(item.get("preEvolution"))
        index = len(card_cat)
        card_cat.append([identity, owner, zone, kind, status])
        card_num.append(
            [
                current_hp / 400.0,
                maximum_hp / 400.0,
                max(0.0, maximum_hp - current_hp) / 400.0,
                len(energy) / 10.0,
                len(tools) / 4.0,
                len(evolution) / 4.0,
                float(bool(item.get("appearThisTurn", item.get("appear", False)))),
            ]
        )
        card_parent.append(parent + 1)
        raw_cards.append(item)
        if key is not None:
            locations[key] = index
        return index

    def add_player_zone(player_index: int, name: str, area: int) -> None:
        player = _player(current, player_index)
        relative = _relative_owner(player_index, actor)
        prefix = "self" if relative == 1 else "opponent"
        zone = ZONE[f"{prefix}_{name}"]
        for slot, raw in enumerate(_items(player.get(name))):
            parent = add_card(
                raw,
                owner=relative,
                zone=zone,
                kind=2 if name in {"active", "bench"} else 1,
                status=_status_bits(player) if name == "active" else 0,
                key=(player_index, area, slot),
            )
            if parent < 0 or name not in {"active", "bench"}:
                continue
            children = (
                ("energyCards", "energies", "energy", 3),
                ("tools", "tools", "tool", 4),
                ("preEvolution", "preEvolution", "evolution", 5),
            )
            for primary, fallback, suffix, kind in children:
                for child in _items(raw.get(primary, raw.get(fallback))) if isinstance(raw, Mapping) else []:
                    add_card(
                        child,
                        owner=relative,
                        zone=ZONE[f"{prefix}_{suffix}"],
                        kind=kind,
                        parent=parent,
                    )

    for player_index in (actor, opponent):
        add_player_zone(player_index, "active", 4)
        add_player_zone(player_index, "bench", 5)
        if player_index == actor:
            add_player_zone(player_index, "hand", 2)
        add_player_zone(player_index, "discard", 3)
    for slot, raw in enumerate(_items(current.get("stadium"))):
        add_card(raw, owner=3, zone=ZONE["stadium"], kind=6, key=(-1, 7, slot))
    for slot, raw in enumerate(_items(current.get("looking"))):
        add_card(raw, owner=1, zone=ZONE["looking"], kind=7, key=(actor, 12, slot))
    for slot, raw in enumerate(_items(select.get("deck"))):
        add_card(raw, owner=1, zone=ZONE["select_deck"], kind=1, key=(actor, 1, slot))

    resource_cat: list[list[int]] = []
    resource_num: list[list[float]] = []
    for identity, entry in sorted(snapshot.self_ledger.items()):
        visible = entry.visible
        resource_cat.append(
            [identity, _knowledge_code(entry.deck.state), _knowledge_code(entry.prize.state), int(snapshot.deck_order_known) + 1]
        )
        resource_num.append(
            [
                float(entry.initial),
                float(visible.get("active", 0)),
                float(visible.get("bench", 0)),
                float(visible.get("hand", 0)),
                float(visible.get("discard", 0)),
                float(visible.get("stadium", 0)),
                float(visible.get("playing", 0)),
                float(entry.deck.value or 0),
                float(entry.deck.lower),
                float(entry.deck.upper),
                float(entry.prize.value or 0),
                float(entry.prize.lower),
                float(entry.prize.upper),
                float(entry.deck.age),
                float(entry.prize.age),
            ]
        )

    event_cat: list[list[int]] = []
    event_num: list[list[float]] = []
    newest = snapshot.recent_events[-1].source_event if snapshot.recent_events else -1
    for event in snapshot.recent_events:
        relative = 3 if event.actor is None else _relative_owner(event.actor, actor)
        event_cat.append(
            [
                event.log_type + 1,
                relative,
                int(event.card_id or 0),
                int(event.from_area if event.from_area is not None else -1) + 1,
                int(event.to_area if event.to_area is not None else -1) + 1,
                int(event.identity_visible) + 1,
                int("serial" in event.payload) + 1,
                int("cardIdTarget" in event.payload) + 1,
            ]
        )
        event_num.append(
            [
                float(newest - event.source_event) / 64.0,
                float(event.payload.get("value", 0) or 0) / 400.0,
                float(event.payload.get("putDamageCounter", 0) or 0) / 30.0,
                float(event.payload.get("head", 0) or 0) / 10.0,
            ]
        )

    context_card = card_id(select.get("contextCard"))
    effect_card = card_id(select.get("effect"))
    option_cat: list[list[int]] = []
    option_num: list[list[float]] = []
    option_state: list[list[int]] = []
    option_source: list[int] = []
    option_target: list[int] = []
    skill_ids: list[int] = []
    skill_roles: list[int] = []
    skill_parents: list[int] = []
    effect_ids: list[int] = []
    effect_roles: list[int] = []
    effect_parents: list[int] = []

    def related_skills(identity: int) -> list[tuple[int, int]]:
        card = prototypes.engine_cards.get(identity, {})
        return [
            (int(card.get(name, 0) or 0), role)
            for role, name in enumerate(("ability_skill_id", "play_skill_id", "delay_skill_id"), 1)
            if int(card.get(name, 0) or 0) > 0
        ]

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
        if action_type == 13 and source_index < 0:
            source_index = locations.get((actor, 4, 0), -1)
        source_identity = card_id(option.get("cardId"))
        if source_identity <= 0 and source_index >= 0:
            source_identity = card_cat[source_index][0]

        target_player = integer(option.get("inPlayPlayerIndex", option.get("targetPlayerIndex", source_player)), actor)
        target_area = integer(option.get("inPlayArea"), -1)
        target_slot = integer(option.get("inPlayIndex"), -1)
        target_index = locations.get((target_player, target_area, target_slot), -1)
        if action_type == 13 and target_index < 0:
            target_player, target_area, target_slot = opponent, 4, 0
            target_index = locations.get((opponent, 4, 0), -1)
        target_identity = card_cat[target_index][0] if target_index >= 0 else 0
        attack_id = max(0, integer(option.get("attackId")))
        source_raw = raw_cards[source_index] if source_index >= 0 else None
        target_raw = raw_cards[target_index] if target_index >= 0 else None
        facts = attack_facts(attack_id, source_raw, prototypes)
        target_hp = float(target_raw.get("hp", 0) or 0) if target_raw else 0.0
        target_max = max(target_hp, float(target_raw.get("maxHp", target_hp) or target_hp)) if target_raw else 0.0
        source_hp = float(source_raw.get("hp", 0) or 0) if source_raw else 0.0
        source_max = max(source_hp, float(source_raw.get("maxHp", source_hp) or source_hp)) if source_raw else 0.0
        after = max(0.0, target_hp - facts.base_damage)
        attack_present = attack_id > 0 and attack_id in prototypes.attacks
        hp_present = target_raw is not None
        selected_energy = 0
        source_prototype = prototypes.cards.get(source_identity)
        if source_prototype and isinstance(source_prototype.get("energy_type"), Mapping):
            energy_field = source_prototype["energy_type"]
            if int(energy_field.get("state", FieldState.UNKNOWN)) == int(FieldState.PRESENT):
                selected_energy = integer(energy_field.get("value"), -1) + 1
        energy_index = integer(option.get("energyIndex"), -1)
        if energy_index >= 0:
            energy_cards = _items(source_raw.get("energyCards")) if source_raw else []
            if energy_index < len(energy_cards):
                energy_proto = prototypes.cards.get(card_id(energy_cards[energy_index]))
                if energy_proto and isinstance(energy_proto.get("energy_type"), Mapping):
                    selected_energy = integer(energy_proto["energy_type"].get("value"), -1) + 1
        option_cat.append(
            [
                action_type + 1,
                _relative_owner(source_player, actor),
                source_area + 1,
                _relative_owner(target_player, actor),
                target_area + 1,
                source_identity,
                target_identity,
                attack_id,
                selected_energy,
                integer(option.get("specialConditionType"), -1) + 1,
                integer(select.get("type"), -1) + 1,
                integer(select.get("context"), -1) + 1,
                context_card,
                effect_card,
            ]
        )
        option_num.append(
            [
                float(integer(option.get("number"), 0)) / 128.0,
                float(integer(option.get("count"), 0)) / 16.0,
                float(select.get("remainDamageCounter", 0) or 0) / 300.0,
                float(select.get("remainEnergyCost", 0) or 0) / 10.0,
                facts.base_damage / 400.0,
                facts.required_energy_count / 10.0,
                facts.attached_energy_count / 10.0,
                facts.exact_energy_matches / 10.0,
                facts.typed_energy_deficit / 10.0,
                facts.total_energy_deficit / 10.0,
                target_hp / 400.0,
                target_max / 400.0,
                after / 400.0,
                float(bool(attack_present and target_hp > 0 and facts.base_damage >= target_hp)),
                source_hp / 400.0,
                source_max / 400.0,
            ]
        )
        option_state.append(
            [
                int(FieldState.PRESENT),
                int(FieldState.PRESENT),
                int(FieldState.PRESENT),
                int(FieldState.PRESENT),
                facts.base_damage_state if attack_present else int(FieldState.NOT_APPLICABLE),
                int(FieldState.PRESENT if attack_present else FieldState.NOT_APPLICABLE),
                int(FieldState.PRESENT if attack_present else FieldState.NOT_APPLICABLE),
                int(FieldState.PRESENT if attack_present else FieldState.NOT_APPLICABLE),
                int(FieldState.PRESENT if attack_present else FieldState.NOT_APPLICABLE),
                int(FieldState.PRESENT if attack_present else FieldState.NOT_APPLICABLE),
                int(FieldState.PRESENT if hp_present else FieldState.NOT_APPLICABLE),
                int(FieldState.PRESENT if hp_present else FieldState.NOT_APPLICABLE),
                int(FieldState.PRESENT if hp_present and attack_present else FieldState.NOT_APPLICABLE),
                int(FieldState.PRESENT if hp_present and attack_present else FieldState.NOT_APPLICABLE),
                int(FieldState.PRESENT if source_raw else FieldState.NOT_APPLICABLE),
                int(FieldState.PRESENT if source_raw else FieldState.NOT_APPLICABLE),
            ]
        )
        option_source.append(source_index + 1)
        option_target.append(target_index + 1)

        seen_skills: set[tuple[int, int]] = set()
        for relation_role, identity in ((1, source_identity), (2, context_card), (3, effect_card)):
            for skill_id, skill_role in related_skills(identity):
                key = (skill_id, (relation_role - 1) * 3 + skill_role)
                if key in seen_skills:
                    continue
                seen_skills.add(key)
                skill_ids.append(skill_id)
                skill_roles.append(key[1])
                skill_parents.append(option_index + 1)
                for effect_ref in prototypes.skill_effect_refs.get(skill_id, ()):
                    effect_ids.append(effect_ref)
                    effect_roles.append(1)
                    effect_parents.append(option_index + 1)
        for effect_ref in prototypes.attack_effect_refs.get(attack_id, ()):
            effect_ids.append(effect_ref)
            effect_roles.append(2)
            effect_parents.append(option_index + 1)

    actor_record = {
        "global_cat": [
            integer(select.get("type"), -1) + 1,
            integer(select.get("context"), -1) + 1,
            1 if integer(current.get("firstPlayer"), -1) == actor else 2,
            int(bool(current.get("supporterPlayed"))) + 1,
            int(bool(current.get("stadiumPlayed"))) + 1,
            int(bool(current.get("energyAttached"))) + 1,
            int(bool(current.get("retreated"))) + 1,
            _status_bits(own) + 1,
            _status_bits(other) + 1,
            int(snapshot.deck_membership_known) + 1,
            int(snapshot.deck_order_known) + 1,
        ],
        "global_num": [
            integer(current.get("turn"), 0) / 20.0,
            integer(current.get("turnActionCount"), 0) / 50.0,
            integer(own.get("deckCount"), 0) / 60.0,
            integer(other.get("deckCount"), 0) / 60.0,
            integer(own.get("handCount"), len(_items(own.get("hand")))) / 20.0,
            integer(other.get("handCount"), len(_items(other.get("hand")))) / 20.0,
            len(_items(own.get("prize"))) / 6.0,
            len(_items(other.get("prize"))) / 6.0,
            len(_items(own.get("bench"))) / 8.0,
            len(_items(other.get("bench"))) / 8.0,
            len(options) / 128.0,
            minimum / 16.0,
            maximum / 16.0,
            float(select.get("remainDamageCounter", 0) or 0) / 300.0,
            float(select.get("remainEnergyCost", 0) or 0) / 10.0,
            len(snapshot.known_opponent_hand) / 20.0,
            snapshot.unknown_opponent_hand / 20.0,
        ],
        "card_cat": card_cat,
        "card_num": card_num,
        "card_parent": card_parent,
        "resource_cat": resource_cat,
        "resource_num": resource_num,
        "event_cat": event_cat,
        "event_num": event_num,
        "option_cat": option_cat,
        "option_num": option_num,
        "option_state": option_state,
        "option_source": option_source,
        "option_target": option_target,
        "option_skill_id": skill_ids,
        "option_skill_role": skill_roles,
        "option_skill_parent": skill_parents,
        "option_effect_id": effect_ids,
        "option_effect_role": effect_roles,
        "option_effect_parent": effect_parents,
        "min_count": minimum,
        "max_count": maximum,
    }
    if set(actor_record) != ACTOR_KEYS:
        raise AssertionError("canonical actor contract drift")
    return {
        "schema_version": SCHEMA_VERSION,
        "actor": actor_record,
        "target": {"ordered_action": action, "termination": row.get("action_termination")},
        "audit": {"identity": row.get("identity"), "split": row.get("split")},
        "registered_deck": _deck(row),
    }


__all__ = ["compile_canonical_row"]
