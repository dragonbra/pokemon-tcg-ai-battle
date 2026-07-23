from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any


ABRA = 741
KADABRA = 742
ALAKAZAM = 743
DUNSPARCE = 305
DUDUNSPARCE = 66
RARE_CANDY = 1079
POKE_PAD = 1152
HILDA = 1225
DAWN = 1231
PSYCHIC_ENERGIES = {5, 19}
POWERFUL_HAND_ATTACK_ID = 1072
RELAY_ROUTE_CARD_IDS = {ABRA, KADABRA, ALAKAZAM}


def _as_int(value: object, default: int | None = None) -> int | None:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _current(payload: dict[str, Any], step: int, player_index: int) -> dict[str, Any]:
    steps = payload.get("steps") or []
    if not 0 <= step < len(steps) or not isinstance(steps[step], list):
        return {}
    rows = steps[step]
    preferred = rows[player_index] if 0 <= player_index < len(rows) else {}
    candidates = [preferred, *rows]
    for row in candidates:
        if not isinstance(row, dict):
            continue
        observation = row.get("observation")
        current = observation.get("current") if isinstance(observation, dict) else None
        if isinstance(current, dict):
            return current
    return {}


def _observation(payload: dict[str, Any], step: int, player_index: int) -> dict[str, Any]:
    rows = (payload.get("steps") or [])[step]
    if not isinstance(rows, list):
        return {}
    preferred = rows[player_index] if 0 <= player_index < len(rows) else {}
    candidates = [preferred, *rows]
    for row in candidates:
        if isinstance(row, dict) and isinstance(row.get("observation"), dict):
            observation = row["observation"]
            if isinstance(observation.get("current"), dict):
                return observation
    return {}


def _player(current: dict[str, Any], player_index: int) -> dict[str, Any]:
    players = current.get("players") or []
    if 0 <= player_index < len(players) and isinstance(players[player_index], dict):
        return players[player_index]
    return {}


def _cards(player: dict[str, Any], area: str) -> list[dict[str, Any]]:
    values = player.get(area) or []
    return [value for value in values if isinstance(value, dict)]


def _active_id(current: dict[str, Any], player_index: int) -> int | None:
    active = _cards(_player(current, player_index), "active")
    return _as_int(active[0].get("id")) if active else None


def _active_has_psychic(current: dict[str, Any], player_index: int) -> bool:
    active = _cards(_player(current, player_index), "active")
    if not active:
        return False
    return bool(
        {
            energy_id
            for energy in active[0].get("energies") or active[0].get("energyCards") or []
            for energy_id in [
                _as_int(energy.get("id")) if isinstance(energy, dict) else _as_int(energy)
            ]
            if energy_id is not None
        }
        & PSYCHIC_ENERGIES
    )


def _field_ids(current: dict[str, Any], player_index: int) -> set[int]:
    player = _player(current, player_index)
    return {
        card_id
        for card in [*_cards(player, "active"), *_cards(player, "bench")]
        for card_id in [_as_int(card.get("id"))]
        if card_id is not None
    }


def _zone_ids(current: dict[str, Any], player_index: int, area: str) -> set[int]:
    return {
        card_id
        for card in _cards(_player(current, player_index), area)
        for card_id in [_as_int(card.get("id"))]
        if card_id is not None
    }


def _prize_count(current: dict[str, Any], player_index: int) -> int | None:
    player = _player(current, player_index)
    prize = player.get("prize")
    if isinstance(prize, list):
        return len(prize)
    return _as_int(player.get("prizeCount"))


def _selected_attack(
    payload: dict[str, Any], record: dict[str, Any]
) -> int | None:
    step = int(record["episode_step"])
    player_index = int(record["player_index"])
    observation = _observation(payload, step, player_index)
    select = observation.get("select") or {}
    options = select.get("option") or []
    for target in record.get("targets") or []:
        if isinstance(target, int) and 0 <= target < len(options):
            option = options[target]
            if isinstance(option, dict):
                attack_id = _as_int(option.get("attackId"))
                if attack_id is not None:
                    return attack_id
    return None


def _next_prize_count(
    payload: dict[str, Any], start_step: int, player_index: int, start_turn: int
) -> int | None:
    observed: list[int] = []
    for step in range(start_step + 1, len(payload.get("steps") or [])):
        current = _current(payload, step, player_index)
        value = _prize_count(current, player_index)
        if value is not None:
            observed.append(value)
        turn = _as_int(current.get("turn"))
        if observed and turn is not None and turn != start_turn:
            break
    return min(observed) if observed else None


def _opening_components(current: dict[str, Any], player_index: int) -> dict[str, bool]:
    hand = _zone_ids(current, player_index, "hand")
    components = {
        "active_abra": _active_id(current, player_index) == ABRA,
        "rare_candy": RARE_CANDY in hand,
        "alakazam_or_search": bool(hand & {ALAKAZAM, POKE_PAD, HILDA, DAWN}),
        "psychic_energy_or_hilda": bool(hand & (PSYCHIC_ENERGIES | {HILDA})),
    }
    components["all_four"] = all(components.values())
    return components


def _relay_knockout_steps(payload: dict[str, Any], player_index: int) -> list[int]:
    previous_field: dict[int, dict[str, Any]] = {}
    seen: set[int] = set()
    events: list[int] = []
    for step in range(len(payload.get("steps") or [])):
        observation = _observation(payload, step, player_index)
        current = observation.get("current") or {}
        player = _player(current, player_index)
        current_field = {
            serial: card
            for card in [*_cards(player, "active"), *_cards(player, "bench")]
            for serial in [_as_int(card.get("serial"))]
            if serial is not None
        }
        logs = observation.get("logs") or []
        damaged = {
            serial
            for log in logs
            if isinstance(log, dict)
            and _as_int(log.get("type")) == 16
            and _as_int(log.get("playerIndex")) == player_index
            for serial in [_as_int(log.get("serial"))]
            if serial is not None
        }
        for log in logs:
            if not isinstance(log, dict):
                continue
            serial = _as_int(log.get("serial"))
            if (
                _as_int(log.get("type")) != 6
                or _as_int(log.get("playerIndex")) != player_index
                or _as_int(log.get("cardId")) != ALAKAZAM
                or _as_int(log.get("fromArea")) not in {4, 5}
                or _as_int(log.get("toArea")) != 3
                or serial is None
                or serial in seen
            ):
                continue
            previous_hp = _as_int((previous_field.get(serial) or {}).get("hp"))
            current_hp = _as_int((current_field.get(serial) or {}).get("hp"))
            confirmed = (
                (previous_hp is not None and previous_hp <= 0)
                or (current_hp is not None and current_hp <= 0)
                or serial in damaged
            )
            if confirmed:
                seen.add(serial)
                events.append(step)
        previous_field = current_field
    return events


def annotate_episode_records(
    records: list[dict[str, Any]], payload: dict[str, Any]
) -> list[dict[str, Any]]:
    """Attach auditable raw metric signals without choosing their reward weights."""
    if not records:
        return []
    ordered = sorted(records, key=lambda record: int(record["episode_step"]))
    player_index = int(ordered[0]["player_index"])
    first_current = _current(payload, int(ordered[0]["episode_step"]), player_index)
    candidate_first = _as_int(first_current.get("firstPlayer")) == player_index
    first_turn = 1 if candidate_first else 2
    target_turn = 3 if candidate_first else 4
    opening = _opening_components(first_current, player_index)
    opening_hand = _zone_ids(first_current, player_index, "hand")
    opening_dunsparce_without_abra = (
        _active_id(first_current, player_index) == DUNSPARCE
        and ABRA not in opening_hand
        and ABRA not in _field_ids(first_current, player_index)
    )
    record_context: dict[int, dict[str, Any]] = {}
    max_turn = max(
        (
            _as_int(_current(payload, step, player_index).get("turn"), 0) or 0
            for step in range(len(payload.get("steps") or []))
        ),
        default=0,
    )
    reached_second_turn = False
    second_turn_powerful_hand = False
    target_active_alakazam = False
    first_turn_bench_abra = False
    target_states: list[dict[str, Any]] = []
    for record in ordered:
        step = int(record["episode_step"])
        current = _current(payload, step, player_index)
        turn = _as_int(current.get("turn"), 0) or 0
        attack_id = _selected_attack(payload, record)
        reached_second_turn |= turn == target_turn
        second_turn_powerful_hand |= turn == target_turn and attack_id == POWERFUL_HAND_ATTACK_ID
        target_active_alakazam |= (
            turn == target_turn and _active_id(current, player_index) == ALAKAZAM
        )
        first_turn_bench_abra |= turn == first_turn and any(
            _as_int(card.get("id")) == ABRA
            for card in _cards(_player(current, player_index), "bench")
        )
        if turn == target_turn:
            target_states.append(
                {
                    "current": current,
                    "active_id": _active_id(current, player_index),
                    "attack_id": attack_id,
                }
            )
        record_context[id(record)] = {
            "current": current,
            "turn": turn,
            "attack_id": attack_id,
        }
    target_first = target_states[0]["current"] if target_states else {}
    target_hand = _zone_ids(target_first, player_index, "hand")
    target_bench_abra = any(
        _as_int(card.get("id")) == ABRA
        for card in _cards(_player(target_first, player_index), "bench")
    )
    target_hand_ready = all(
        (
            DUDUNSPARCE in target_hand,
            RARE_CANDY in target_hand,
            ALAKAZAM in target_hand,
            bool(target_hand & PSYCHIC_ENERGIES),
        )
    )
    milestones = (DUDUNSPARCE, ABRA, ALAKAZAM, POWERFUL_HAND_ATTACK_ID)
    milestone_position = 0
    retreated_after_dudunsparce = False
    active_psychic_at_attack = False
    for state in target_states:
        current = state["current"]
        if milestone_position == 0 and state["active_id"] == milestones[0]:
            milestone_position = 1
        elif milestone_position == 1 and state["active_id"] == milestones[1]:
            if bool(current.get("retreated", False)):
                retreated_after_dudunsparce = True
                milestone_position = 2
        elif milestone_position == 2 and state["active_id"] == milestones[2]:
            milestone_position = 3
        if (
            milestone_position == 3
            and state["active_id"] == ALAKAZAM
            and state["attack_id"] == milestones[3]
        ):
            active_psychic_at_attack = _active_has_psychic(current, player_index)
            if active_psychic_at_attack:
                milestone_position = 4
                break
    bridge_opportunity = opening_dunsparce_without_abra
    bridge_components = {
        "opening_dunsparce_without_abra": opening_dunsparce_without_abra,
        "first_turn_bench_abra": first_turn_bench_abra,
        "target_turn_active_dunsparce": bool(
            target_states and target_states[0]["active_id"] == DUNSPARCE
        ),
        "target_turn_bench_abra": target_bench_abra,
        "target_hand_dudunsparce": DUDUNSPARCE in target_hand,
        "target_hand_rare_candy": RARE_CANDY in target_hand,
        "target_hand_alakazam": ALAKAZAM in target_hand,
        "target_hand_psychic_energy": bool(target_hand & PSYCHIC_ENERGIES),
        "target_hand_all_four": target_hand_ready,
        "evolved_active_dudunsparce": any(
            state["active_id"] == DUDUNSPARCE for state in target_states
        ),
        "retreated_into_abra": retreated_after_dudunsparce,
        "evolved_active_alakazam": target_active_alakazam,
        "active_psychic_at_attack": active_psychic_at_attack,
        "submitted_powerful_hand": second_turn_powerful_hand,
    }
    bridge_success = bool(
        bridge_opportunity
        and first_turn_bench_abra
        and target_bench_abra
        and target_hand_ready
        and milestone_position == 4
    )

    relay_assignments: dict[int, tuple[float, str]] = {}
    for knockout_step in _relay_knockout_steps(payload, player_index):
        response = [
            record for record in ordered if int(record["episode_step"]) > knockout_step
        ]
        if not response:
            continue
        response_turn = int(record_context[id(response[0])]["turn"])
        response = [
            record
            for record in response
            if int(record_context[id(record)]["turn"]) == response_turn
        ]
        success = any(
            record_context[id(record)]["attack_id"] is not None
            and _active_id(record_context[id(record)]["current"], player_index) == ALAKAZAM
            for record in response
        )
        first_response = record_context[id(response[0])]["current"]
        field = _field_ids(first_response, player_index)
        discard = _zone_ids(first_response, player_index, "discard")
        hand = _zone_ids(first_response, player_index, "hand")
        deck_count = _as_int(_player(first_response, player_index).get("deckCount"), 0) or 0
        if success:
            classification = "success"
        elif field & {ABRA, KADABRA}:
            classification = "field_route_miss"
        elif discard & RELAY_ROUTE_CARD_IDS:
            classification = "recoverable_discard_miss"
        elif hand & RELAY_ROUTE_CARD_IDS:
            classification = "recoverable_route_incomplete"
        elif deck_count > 0:
            classification = "nonterminal_no_field_route"
        else:
            classification = "terminal_no_resource"
        for record in response:
            relay_assignments[id(record)] = (1.0 if success else -1.0, classification)

    annotated: list[dict[str, Any]] = []
    for record in records:
        result = copy.deepcopy(record)
        context = record_context[id(record)]
        current = context["current"]
        turn = int(context["turn"])
        attack_id = context["attack_id"]
        before_prize = _prize_count(current, player_index)
        after_prize = (
            _next_prize_count(
                payload,
                int(record["episode_step"]),
                player_index,
                turn,
            )
            if attack_id is not None
            else None
        )
        prize_delta = (
            before_prize - after_prize
            if before_prize is not None and after_prize is not None
            else None
        )
        relay_credit, relay_class = relay_assignments.get(id(record), (0.0, "none"))
        setup_window = turn <= target_turn
        result["reward_schema_version"] = "alakazam_offline_reward_metrics_v1"
        result["reward_metrics"] = {
            "episode": {
                "candidate_first": candidate_first,
                "game_turn_count": max_turn,
                "target_second_turn": target_turn,
                "reached_second_turn": reached_second_turn,
                "second_turn_powerful_hand": second_turn_powerful_hand,
                "dunsparce_bridge_opportunity": bridge_opportunity,
                "dunsparce_bridge_success": bridge_success,
                "dunsparce_bridge_components": bridge_components,
                "opening_components": opening,
            },
            "decision": {
                "second_turn_setup_credit": (
                    1.0 if second_turn_powerful_hand else -1.0
                ) if setup_window else 0.0,
                "dunsparce_bridge_credit": (
                    (1.0 if bridge_success else -1.0)
                    if bridge_opportunity and setup_window
                    else 0.0
                ),
                "dunsparce_first_turn_setup_credit": (
                    (1.0 if first_turn_bench_abra else -1.0)
                    if bridge_opportunity and turn == first_turn
                    else 0.0
                ),
                "dunsparce_second_turn_execution_credit": (
                    (1.0 if bridge_success else -1.0)
                    if bridge_opportunity and turn == target_turn
                    else 0.0
                ),
                "attack_prize_delta": float(prize_delta or 0),
                "non_prize_attack": float(
                    attack_id is not None and prize_delta is not None and prize_delta <= 0
                ),
                "powerful_hand_non_prize_attack": float(
                    attack_id == POWERFUL_HAND_ATTACK_ID
                    and prize_delta is not None
                    and prize_delta <= 0
                ),
                "post_ko_relay_credit": relay_credit,
                "recoverable_discard_miss": float(
                    relay_class == "recoverable_discard_miss"
                ),
            },
            "audit": {
                "turn": turn,
                "selected_attack_id": attack_id,
                "attack_prize_delta_known": prize_delta is not None,
                "post_ko_relay_classification": relay_class,
            },
        }
        annotated.append(result)
    return annotated


def reward_metric_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "records": 0,
        "second_turn_powerful_hand_records": 0,
        "dunsparce_bridge_success_records": 0,
        "non_prize_attack_records": 0,
        "post_ko_relay_success_records": 0,
        "post_ko_relay_failure_records": 0,
        "recoverable_discard_miss_records": 0,
    }
    for record in records:
        counts["records"] += 1
        metrics = record.get("reward_metrics") or {}
        episode = metrics.get("episode") or {}
        decision = metrics.get("decision") or {}
        counts["second_turn_powerful_hand_records"] += int(
            bool(episode.get("second_turn_powerful_hand"))
        )
        counts["dunsparce_bridge_success_records"] += int(
            bool(episode.get("dunsparce_bridge_success"))
        )
        counts["non_prize_attack_records"] += int(decision.get("non_prize_attack", 0) > 0)
        relay = float(decision.get("post_ko_relay_credit", 0.0))
        counts["post_ko_relay_success_records"] += int(relay > 0)
        counts["post_ko_relay_failure_records"] += int(relay < 0)
        counts["recoverable_discard_miss_records"] += int(
            decision.get("recoverable_discard_miss", 0) > 0
        )
    return counts
