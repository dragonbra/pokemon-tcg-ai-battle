"""Compile legacy-compatible state plus typed prototype and option references."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from ..knowledge.state import CausalSnapshot
from ..legacy.base_model import IDOnlyCodec, IDOnlyConfig
from .prototypes import FieldState, PrototypeIndex

SCHEMA_VERSION = "0025_semantic_decision_v1"
OPTION_CAT_WIDTH = 16
OPTION_NUM_WIDTH = 14
LEDGER_CAT_WIDTH = 4
LEDGER_NUM_WIDTH = 15
EVENT_CAT_WIDTH = 8
EVENT_NUM_WIDTH = 4


def _items(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _integer(value: Any, default: int = 0) -> int:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else default


def _card_id(value: Any) -> int:
    if isinstance(value, Mapping):
        return max(0, _integer(value.get("id", value.get("cardId", 0))))
    return max(0, _integer(value))


def _field(record: Mapping[str, Any] | None, name: str) -> tuple[float, int]:
    if record is None:
        return 0.0, int(FieldState.UNKNOWN)
    value = record.get(name)
    if not isinstance(value, Mapping):
        return 0.0, int(FieldState.UNKNOWN)
    return float(value.get("value", 0) or 0), int(value.get("state", FieldState.UNKNOWN))


def _attached_energy_types(active: Mapping[str, Any], prototypes: PrototypeIndex) -> list[int]:
    result: list[int] = []
    for item in _items(active.get("energyCards", active.get("energies", []))):
        card = prototypes.cards.get(_card_id(item))
        value, state = _field(card, "energy_type")
        if state == int(FieldState.PRESENT):
            result.append(int(value))
    return result


def _energy_gap(required: Sequence[int], attached: Sequence[int]) -> tuple[int, int, int]:
    """Return exact matches, typed deficit, and total deficit.

    Energy type 0 is colorless and 10 is rainbow in the official API.
    This intentionally does not guess bespoke Special Energy text semantics.
    """
    available = Counter(int(value) for value in attached)
    exact = 0
    typed_deficit = 0
    colored = [int(value) for value in required if int(value) != 0]
    colorless = sum(int(value) == 0 for value in required)
    for energy_type in colored:
        if available[energy_type] > 0:
            available[energy_type] -= 1
            exact += 1
        elif available[10] > 0:
            available[10] -= 1
            exact += 1
        else:
            typed_deficit += 1
    remaining = sum(available.values())
    colorless_deficit = max(0, colorless - remaining)
    return exact, typed_deficit, typed_deficit + colorless_deficit


def _deck_cards(row: Mapping[str, Any]) -> list[int]:
    values: list[int] = []
    for card_id, count in row["deck_manifest"]["counts"]:
        values.extend([int(card_id)] * int(count))
    if len(values) != 60:
        raise ValueError("registered deck must contain exactly 60 cards")
    return values


def _ledger(snapshot: CausalSnapshot) -> tuple[list[list[int]], list[list[float]]]:
    cats: list[list[int]] = []
    nums: list[list[float]] = []
    for card_id, item in sorted(snapshot.self_ledger.items()):
        deck, prize = item.deck, item.prize
        states = list(type(deck.state))
        cats.append([card_id, states.index(deck.state) + 1, states.index(prize.state) + 1, int(snapshot.deck_order_known)])
        visible = item.visible
        nums.append([
            float(item.initial), float(visible.get("active", 0)), float(visible.get("bench", 0)),
            float(visible.get("hand", 0)), float(visible.get("discard", 0)),
            float(visible.get("stadium", 0)), float(visible.get("playing", 0)),
            float(deck.value or 0), float(deck.lower), float(deck.upper),
            float(prize.value or 0), float(prize.lower), float(prize.upper),
            float(deck.age), float(prize.age),
        ])
    return cats, nums


def _events(snapshot: CausalSnapshot) -> tuple[list[list[int]], list[list[float]]]:
    cats: list[list[int]] = []
    nums: list[list[float]] = []
    newest = snapshot.recent_events[-1].source_event if snapshot.recent_events else -1
    for event in snapshot.recent_events:
        relative_actor = 0 if event.actor is None else (1 if event.actor == snapshot.perspective_actor else 2)
        cats.append([
            event.log_type + 1, relative_actor, int(event.card_id or 0),
            int(event.from_area if event.from_area is not None else -1) + 1,
            int(event.to_area if event.to_area is not None else -1) + 1,
            int(event.identity_visible), int("serial" in event.payload), int("cardIdTarget" in event.payload),
        ])
        nums.append([
            float(newest - event.source_event), float(event.payload.get("value", 0) or 0),
            float(event.payload.get("putDamageCounter", 0) or 0), float(event.payload.get("head", 0) or 0),
        ])
    return cats, nums


def compile_row(
    row: Mapping[str, Any], snapshot: CausalSnapshot, prototypes: PrototypeIndex
) -> dict[str, Any]:
    observation = row["actor_observation"]
    action = [int(value) for value in row["ordered_action"]]
    legacy = IDOnlyCodec(IDOnlyConfig(max_action_steps=64)).encode(observation, action)
    if legacy is None:
        raise ValueError("legacy codec rejected an accepted raw decision")
    current = observation["current"]
    select = observation["select"]
    actor = int(current["yourIndex"])
    own = current["players"][actor]
    active_items = [item for item in _items(own.get("active")) if isinstance(item, Mapping)]
    active = active_items[0] if active_items else {}
    attached_types = _attached_energy_types(active, prototypes)
    legacy_options = legacy["option_cat"]
    semantic_cat: list[list[int]] = []
    semantic_num: list[list[float]] = []
    semantic_state: list[list[int]] = []
    prototype_refs = {row[0] for row in legacy["entity_cat"] if row[0] > 0}
    attack_refs: set[int] = set()
    context_card = _card_id(select.get("contextCard"))
    effect_card = _card_id(select.get("effect"))
    for ordinal, raw in enumerate(_items(select.get("option"))):
        option = raw if isinstance(raw, Mapping) else {}
        old = legacy_options[ordinal]
        attack_id = max(0, _integer(option.get("attackId")))
        skill_card = max(0, _integer(option.get("cardId"))) if _integer(option.get("type"), -1) == 15 else 0
        attack = prototypes.attacks.get(attack_id)
        if attack_id:
            attack_refs.add(attack_id)
        source_card, target_card = old[4], old[5]
        prototype_refs.update(value for value in (source_card, target_card, skill_card, context_card, effect_card) if value)
        damage, damage_state = _field(attack, "base_damage")
        required = list(attack.get("energy_types", [])) if attack else []
        exact, typed_gap, total_gap = _energy_gap(required, attached_types)
        target_entity = old[10] - 1
        target_hp = target_max_hp = 0.0
        hp_state = int(FieldState.NOT_APPLICABLE)
        if target_entity >= 0:
            entity_num = legacy["entity_num"][target_entity]
            target_card_proto = prototypes.cards.get(target_card)
            target_max_hp, hp_state = _field(target_card_proto, "hp")
            target_hp = max(0.0, target_max_hp - entity_num[0] * 400.0)
        after = max(0.0, target_hp - damage)
        is_attack = _integer(option.get("type"), -1) == 13
        semantic_cat.append([
            _integer(option.get("type"), -1) + 1, attack_id, skill_card,
            max(0, _integer(option.get("serial"))) % 4096,
            _integer(option.get("specialConditionType"), -1) + 1,
            _integer(option.get("energyIndex"), -1) + 1,
            _integer(option.get("count"), -1) + 1,
            source_card, target_card, old[9], old[10], context_card, effect_card,
            _integer(select.get("context"), -1) + 1, ordinal + 1,
            int(FieldState.PRESENT if attack else FieldState.NOT_APPLICABLE),
        ])
        semantic_num.append([
            damage / 400.0, len(required) / 10.0, len(attached_types) / 10.0,
            exact / 10.0, typed_gap / 10.0, total_gap / 10.0,
            target_hp / 400.0, target_max_hp / 400.0, after / 400.0,
            float(bool(is_attack and target_hp > 0 and damage >= target_hp)),
            0.0, float(is_attack),
            float(select.get("remainEnergyCost", 0) or 0) / 10.0,
            float(select.get("remainDamageCounter", 0) or 0) / 300.0,
        ])
        semantic_state.append([
            damage_state,
            int(FieldState.PRESENT if attack else FieldState.NOT_APPLICABLE),
            int(FieldState.PRESENT), int(FieldState.PRESENT),
            int(FieldState.PRESENT if attack else FieldState.NOT_APPLICABLE),
            int(FieldState.PRESENT if attack else FieldState.NOT_APPLICABLE),
            hp_state, hp_state, hp_state,
            int(FieldState.PRESENT if is_attack and hp_state == int(FieldState.PRESENT) else FieldState.NOT_APPLICABLE),
            int(FieldState.UNKNOWN), int(FieldState.PRESENT), int(FieldState.PRESENT), int(FieldState.PRESENT),
        ])
    ledger_cat, ledger_num = _ledger(snapshot)
    event_cat, event_num = _events(snapshot)
    deck_counts = Counter(_deck_cards(row))
    prototype_skill_refs = sorted({
        int(skill_id)
        for card_id in prototype_refs
        for skill_id in (
            prototypes.engine_cards.get(card_id, {}).get("ability_skill_id", 0),
            prototypes.engine_cards.get(card_id, {}).get("play_skill_id", 0),
            prototypes.engine_cards.get(card_id, {}).get("delay_skill_id", 0),
        )
        if skill_id
    })
    prototype_effect_refs = sorted({
        effect_ref
        for skill_id in prototype_skill_refs
        for effect_ref in prototypes.skill_effect_refs.get(skill_id, ())
    } | {
        effect_ref
        for attack_id in attack_refs
        for effect_ref in prototypes.attack_effect_refs.get(attack_id, ())
    })
    return {
        "schema_version": SCHEMA_VERSION,
        "legacy": legacy,
        "prototype_card_refs": sorted(prototype_refs),
        "prototype_attack_refs": sorted(attack_refs),
        "prototype_skill_refs": prototype_skill_refs,
        "prototype_effect_refs": prototype_effect_refs,
        "semantic_option_cat": semantic_cat,
        "semantic_option_num": semantic_num,
        "semantic_option_state": semantic_state,
        "deck_card_ids": sorted(deck_counts),
        "deck_multiplicity": [deck_counts[value] for value in sorted(deck_counts)],
        "ledger_cat": ledger_cat,
        "ledger_num": ledger_num,
        "event_cat": event_cat,
        "event_num": event_num,
        "known_opponent_hand_card_ids": [item.card_id for item in snapshot.known_opponent_hand],
        "unknown_opponent_hand_count": snapshot.unknown_opponent_hand,
        "turn_budget": [
            int(bool(current.get("supporterPlayed"))), int(bool(current.get("stadiumPlayed"))),
            int(bool(current.get("energyAttached"))), int(bool(current.get("retreated"))),
            int(bool(current.get("turnEnd"))),
        ],
        "action_termination": row["action_termination"],
    }


def actor_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Explicit actor-visible boundary; provenance fields cannot enter forward."""
    allowed = {
        "legacy", "prototype_card_refs", "prototype_attack_refs", "prototype_skill_refs", "prototype_effect_refs", "semantic_option_cat",
        "semantic_option_num", "semantic_option_state", "deck_card_ids", "deck_multiplicity",
        "ledger_cat", "ledger_num", "event_cat", "event_num", "known_opponent_hand_card_ids",
        "unknown_opponent_hand_count", "turn_budget",
    }
    payload = {key: record[key] for key in allowed}
    legacy = dict(payload["legacy"])
    legacy.pop("action", None)
    payload["legacy"] = legacy
    return payload


__all__ = ["SCHEMA_VERSION", "actor_payload", "compile_row"]
