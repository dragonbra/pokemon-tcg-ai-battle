"""Build AutoIteration metric reports from full evaluator traces.

The evaluator owns game execution. This module only reads its JSON records,
classifies observable events, and renders lightweight JSON/HTML artifacts for
the iteration history.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


ALAKAZAM = 743
ABRA = 741
KADABRA = 742
DUNSPARCE = 305
DUDUNSPARCE = 66
RARE_CANDY = 1079
POKE_PAD = 1152
HILDA = 1225
DAWN = 1231
ENRICHING_ENERGY = 13
PSYCHIC_ENERGIES = {5, 19}
POWERFUL_HAND_ATTACK = 1072
DRAW_LOG = 4
TURN_START_LOG = 2
REPORT_SCHEMA_VERSION = 3

OPENING_COMPONENT_KEYS = (
    "active_abra",
    "rare_candy",
    "alakazam_or_search",
    "psychic_energy_or_hilda",
    "all_four",
)

RELAY_FAILURES = (
    "field_route_miss",
    "recoverable_discard_miss",
    "recoverable_route_incomplete",
    "nonterminal_no_field_route",
    "terminal_no_resource",
)


def _ids(cards: Any) -> set[int]:
    if not isinstance(cards, list):
        return set()
    return {
        int(card["id"])
        for card in cards
        if isinstance(card, Mapping) and isinstance(card.get("id"), int)
    }


def _player_from_step(step: Mapping[str, Any], player_index: int | None = None) -> dict[str, Any]:
    observation = step.get("observation")
    if isinstance(observation, Mapping):
        current = observation.get("current")
        if isinstance(current, Mapping):
            players = current.get("players")
            index = (
                player_index
                if isinstance(player_index, int)
                else current.get("yourIndex", step.get("yourIndex", 0))
            )
            if isinstance(players, list) and isinstance(index, int) and 0 <= index < len(players):
                player = players[index]
                if isinstance(player, dict):
                    return player

    players = step.get("players")
    index = player_index if isinstance(player_index, int) else step.get("yourIndex", 0)
    if isinstance(players, list) and isinstance(index, int) and 0 <= index < len(players):
        player = players[index]
        if isinstance(player, dict):
            return player
    return {}


def _turn(step: Mapping[str, Any]) -> int | None:
    value = step.get("turn")
    return value if isinstance(value, int) else None


def _agent_physical_index(record: Mapping[str, Any]) -> int:
    value = record.get("alakazamPhysicalIndex")
    return value if isinstance(value, int) and value in (0, 1) else 0


def _first_player_index(record: Mapping[str, Any]) -> int | None:
    trace = record.get("trace")
    if not isinstance(trace, list):
        return None
    for step in trace:
        if not isinstance(step, Mapping):
            continue
        observation = step.get("observation")
        current = observation.get("current") if isinstance(observation, Mapping) else None
        first_player = current.get("firstPlayer") if isinstance(current, Mapping) else None
        if isinstance(first_player, int) and first_player in (0, 1):
            return first_player
    return None


def _agent_goes_first(record: Mapping[str, Any]) -> bool:
    first_player = _first_player_index(record)
    if first_player is not None:
        return first_player == _agent_physical_index(record)
    # Older lightweight fixtures only carry swap; keep that fallback for them.
    return not bool(record.get("swap"))


def _active(player: Mapping[str, Any]) -> dict[str, Any] | None:
    active = player.get("active")
    if isinstance(active, list) and active and isinstance(active[0], dict):
        return active[0]
    return None


def _field_ids(player: Mapping[str, Any]) -> set[int]:
    cards: list[Any] = []
    active = player.get("active")
    bench = player.get("bench")
    if isinstance(active, list):
        cards.extend(active)
    if isinstance(bench, list):
        cards.extend(bench)
    return _ids(cards)


def _hand_ids(player: Mapping[str, Any]) -> set[int]:
    return _ids(player.get("hand"))


def _discard_ids(player: Mapping[str, Any]) -> set[int]:
    return _ids(player.get("discard"))


def _deck_count(player: Mapping[str, Any]) -> int:
    value = player.get("deckCount")
    return value if isinstance(value, int) else 0


def _prize_count(player: Mapping[str, Any]) -> int | None:
    prize = player.get("prize")
    if isinstance(prize, list):
        return len(prize)
    value = player.get("prizeCount")
    return value if isinstance(value, int) else None


def _step_options(step: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    select = step.get("select")
    if isinstance(select, Mapping):
        options = select.get("options")
        if isinstance(options, list):
            return [option for option in options if isinstance(option, Mapping)]
    observation = step.get("observation")
    if isinstance(observation, Mapping):
        select = observation.get("select")
        if isinstance(select, Mapping):
            options = select.get("option") or select.get("options")
            if isinstance(options, list):
                return [option for option in options if isinstance(option, Mapping)]
    return []


def _selected_attack_options(step: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    action = step.get("action")
    options = _step_options(step)
    if not isinstance(action, list):
        return []
    selected_options = []
    for selected in action:
        if isinstance(selected, int) and 0 <= selected < len(options):
            option = options[selected]
            if isinstance(option.get("attackId"), int):
                selected_options.append(option)
    return selected_options


def _selected_option(step: Mapping[str, Any]) -> Mapping[str, Any] | None:
    return next(
        (
            option
            for option in _selected_attack_options(step)
            if option.get("attackId") == POWERFUL_HAND_ATTACK
        ),
        None,
    )


def _is_agent_step(step: Mapping[str, Any], agent_label: str) -> bool:
    return step.get("role") == agent_label


def _first_agent_steps(record: Mapping[str, Any], agent_label: str) -> list[dict[str, Any]]:
    trace = record.get("trace")
    if not isinstance(trace, list):
        return []
    return [step for step in trace if isinstance(step, dict) and _is_agent_step(step, agent_label)]


def _turn_start_steps(record: Mapping[str, Any], agent_label: str) -> list[dict[str, Any]]:
    """Return the first observable agent state for each engine turn."""
    starts: list[dict[str, Any]] = []
    seen_turns: set[int] = set()
    for step in _first_agent_steps(record, agent_label):
        turn = _turn(step)
        if turn is None or turn in seen_turns:
            continue
        seen_turns.add(turn)
        starts.append(step)
    return starts


def _target_turn(record: Mapping[str, Any]) -> int:
    return 3 if _agent_goes_first(record) else 4


def _state_snapshot(step: Mapping[str, Any], player_index: int | None = None) -> dict[str, Any]:
    player = _player_from_step(step, player_index)
    return {
        "turn": _turn(step),
        "player": player,
        "active_id": (_active(player) or {}).get("id"),
        "active_serial": (_active(player) or {}).get("serial"),
        "field_ids": _field_ids(player),
        "hand_ids": _hand_ids(player),
        "discard_ids": _discard_ids(player),
        "deck_count": _deck_count(player),
        "prize_count": _prize_count(player),
    }


def _first_real_state(record: Mapping[str, Any], agent_label: str) -> dict[str, Any] | None:
    first_turn = 1 if _agent_goes_first(record) else 2
    for step in _turn_start_steps(record, agent_label):
        if _turn(step) != first_turn:
            continue
        snapshot = _state_snapshot(step)
        if snapshot["active_id"] is not None:
            return snapshot
    for step in _turn_start_steps(record, agent_label):
        snapshot = _state_snapshot(step)
        if snapshot["active_id"] is not None:
            return snapshot
    return None


def _target_turn_steps(record: Mapping[str, Any], agent_label: str) -> list[dict[str, Any]]:
    target = _target_turn(record)
    return [
        step
        for step in _first_agent_steps(record, agent_label)
        if _turn(step) == target
    ]


def _attack_success(record: Mapping[str, Any], agent_label: str, turn: int) -> bool:
    return any(
        _selected_option(step) is not None
        for step in _first_agent_steps(record, agent_label)
        if _turn(step) == turn
    )


def _step_logs(step: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    observation = step.get("observation")
    logs = observation.get("logs") if isinstance(observation, Mapping) else None
    if not isinstance(logs, list):
        return []
    return [log for log in logs if isinstance(log, Mapping)]


def _agent_attack_logs(step: Mapping[str, Any], player_index: int) -> list[Mapping[str, Any]]:
    return [
        log
        for log in _step_logs(step)
        if log.get("type") == 15 and log.get("playerIndex") == player_index
    ]


def _attack_quality_analysis(record: Mapping[str, Any], agent_label: str) -> dict[str, Any]:
    trace = record.get("trace")
    if not isinstance(trace, list):
        trace = []
    player_index = _agent_physical_index(record)
    submissions = 0
    resolved = 0
    unresolved = 0
    unknown_prize = 0
    non_prize = 0
    powerful_resolved = 0
    powerful_unknown_prize = 0
    powerful_non_prize = 0
    cases: list[dict[str, Any]] = []

    for index, step in enumerate(trace):
        if not isinstance(step, dict) or not _is_agent_step(step, agent_label):
            continue
        for option in _selected_attack_options(step):
            submissions += 1
            attack_id = option.get("attackId")
            is_powerful_hand = attack_id == POWERFUL_HAND_ATTACK
            before = _state_snapshot(step, player_index)
            resolution_index: int | None = None
            resolution_logs: list[Mapping[str, Any]] = []
            for candidate_index in range(index + 1, len(trace)):
                candidate = trace[candidate_index]
                if not isinstance(candidate, Mapping):
                    continue
                candidate_logs = _agent_attack_logs(candidate, player_index)
                matching_logs = [
                    log
                    for log in candidate_logs
                    if log.get("attackId") == attack_id
                ]
                if matching_logs:
                    resolution_index = candidate_index
                    resolution_logs = _step_logs(candidate)
                    break
                if _is_agent_step(candidate, agent_label) and _selected_attack_options(candidate):
                    break

            if resolution_index is None:
                unresolved += 1
                if is_powerful_hand:
                    cases.append(
                        {
                            "attack_id": attack_id,
                            "powerful_hand": True,
                            "resolved": False,
                            "non_prize": None,
                        }
                    )
                continue

            resolved += 1
            after_index = resolution_index + 1
            after = (
                _state_snapshot(trace[after_index], player_index)
                if after_index < len(trace)
                else None
            )
            before_prizes = before.get("prize_count")
            after_prizes = after.get("prize_count") if after is not None else None
            prize_delta = (
                before_prizes - after_prizes
                if isinstance(before_prizes, int) and isinstance(after_prizes, int)
                else None
            )
            no_prize = prize_delta is not None and prize_delta <= 0
            if prize_delta is None:
                unknown_prize += 1
            elif no_prize:
                non_prize += 1
            if is_powerful_hand:
                powerful_resolved += 1
                if prize_delta is None:
                    powerful_unknown_prize += 1
                elif no_prize:
                    powerful_non_prize += 1
            cases.append(
                {
                    "attack_id": attack_id,
                    "powerful_hand": is_powerful_hand,
                    "resolved": True,
                    "non_prize": no_prize if prize_delta is not None else None,
                    "prize_delta": prize_delta,
                    "damage_observed": any(
                        log.get("type") == 16 and log.get("playerIndex") != player_index
                        for log in resolution_logs
                    ),
                }
            )

    return {
        "attack_submissions": submissions,
        "resolved_attacks": resolved,
        "unresolved_attacks": unresolved,
        "unknown_prize_attacks": unknown_prize,
        "non_prize_attacks": non_prize,
        "powerful_hand": {
            "resolved_attacks": powerful_resolved,
            "unknown_prize_attacks": powerful_unknown_prize,
            "non_prize_attacks": powerful_non_prize,
        },
        "cases": cases,
    }


def _opening_four_components(snapshot: Mapping[str, Any] | None) -> dict[str, bool]:
    if snapshot is None:
        return {
            "active_abra": False,
            "rare_candy": False,
            "alakazam_or_search": False,
            "psychic_energy_or_hilda": False,
            "all_four": False,
        }
    hand_ids = snapshot["hand_ids"]
    components = {
        "active_abra": snapshot["active_id"] == ABRA,
        "rare_candy": RARE_CANDY in hand_ids,
        "alakazam_or_search": bool(hand_ids & {ALAKAZAM, POKE_PAD, HILDA, DAWN}),
        "psychic_energy_or_hilda": bool(hand_ids & (PSYCHIC_ENERGIES | {HILDA})),
    }
    components["all_four"] = all(components.values())
    return components


def _relay_failure(snapshot: Mapping[str, Any]) -> str:
    discard_ids = snapshot["discard_ids"]
    if discard_ids & {ABRA, KADABRA, ALAKAZAM}:
        return "recoverable_discard_miss"
    if ABRA in snapshot["field_ids"]:
        return "field_route_miss"
    if snapshot["deck_count"] > 0:
        return "nonterminal_no_field_route"
    return "terminal_no_resource"


def _relay_analysis(record: Mapping[str, Any], agent_label: str) -> dict[str, Any]:
    starts = _turn_start_steps(record, agent_label)
    opportunities = 0
    successes = 0
    failures: Counter[str] = Counter()
    cases: list[dict[str, Any]] = []
    for previous, current in zip(starts, starts[1:]):
        previous_snapshot = _state_snapshot(previous)
        current_snapshot = _state_snapshot(current)
        if previous_snapshot["active_id"] != ALAKAZAM:
            continue
        # A normal uninterrupted Alakazam turn is not a relay opportunity.
        # A changed serial means a new active instance; None/different IDs
        # mean the previous active left play after being KO'd.
        if (
            current_snapshot["active_id"] == ALAKAZAM
            and current_snapshot["active_serial"] == previous_snapshot["active_serial"]
        ):
            continue
        opportunities += 1
        current_turn = _turn(current)
        success = current_turn is not None and _attack_success(record, agent_label, current_turn)
        if success:
            successes += 1
            classification = "success"
        else:
            classification = _relay_failure(current_snapshot)
            failures[classification] += 1
        cases.append(
            {
                "previous_turn": previous_snapshot["turn"],
                "current_turn": current_snapshot["turn"],
                "success": success,
                "classification": classification,
            }
        )
    return {
        "opportunities": opportunities,
        "successes": successes,
        "success_rate": _rate(successes, opportunities),
        "failure_counts": {name: failures.get(name, 0) for name in RELAY_FAILURES},
        "cases": cases,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _draws_to_second_turn(record: Mapping[str, Any], agent_label: str) -> dict[str, Any]:
    """Count visible extra draws through the end of our second turn.

    LogType.Draw is exact about the number of cards entering the player's
    hand. A Draw accompanying that player's TurnStart is the mandatory draw
    for the turn and is reported separately; the remaining draws are the
    ability/effect draw signal used by this profile.
    """
    trace = record.get("trace")
    if not isinstance(trace, list):
        return {
            "reached_second_turn": False,
            "ability_draw_cards": 0,
            "normal_draw_cards": 0,
            "total_draw_cards": 0,
        }
    physical_index = _agent_physical_index(record)
    first_turn = 1 if _agent_goes_first(record) else 2
    target_turn = _target_turn(record)
    ability_draw_cards = 0
    normal_draw_cards = 0
    reached_second_turn = False
    for step in trace:
        if not isinstance(step, Mapping):
            continue
        if not _is_agent_step(step, agent_label):
            continue
        turn = _turn(step)
        if turn is None or turn < first_turn or turn > target_turn:
            continue
        if turn == target_turn:
            reached_second_turn = True
        observation = step.get("observation")
        logs = observation.get("logs") if isinstance(observation, Mapping) else None
        if not isinstance(logs, list):
            continue
        own_logs = [
            log
            for log in logs
            if isinstance(log, Mapping) and log.get("playerIndex") == physical_index
        ]
        draws = sum(log.get("type") == DRAW_LOG for log in own_logs)
        if any(log.get("type") == TURN_START_LOG for log in own_logs):
            normal_draw_cards += draws
        else:
            ability_draw_cards += draws
    return {
        "reached_second_turn": reached_second_turn,
        "ability_draw_cards": ability_draw_cards,
        "normal_draw_cards": normal_draw_cards,
        "total_draw_cards": ability_draw_cards + normal_draw_cards,
    }


def analyze_game_record(record: Mapping[str, Any], *, agent_label: str) -> dict[str, Any]:
    """Extract observable metrics for one evaluator game record."""
    agent_first = _agent_goes_first(record)
    first_state = _first_real_state(record, agent_label)
    target_steps = _target_turn_steps(record, agent_label)
    target_state = _state_snapshot(target_steps[0]) if target_steps else None
    target_turn = _target_turn(record)
    attack_success = _attack_success(record, agent_label, target_turn)
    components = _opening_four_components(first_state)
    draws = _draws_to_second_turn(record, agent_label)
    bridge_opportunity = components["active_abra"] is False and (
        first_state is not None and first_state["active_id"] == DUNSPARCE
    )
    saw_dudunsparce = any(
        DUDUNSPARCE in _state_snapshot(step)["field_ids"]
        for step in _first_agent_steps(record, agent_label)
        if (_turn(step) or 0) <= target_turn
    )
    bridge_success = bool(
        bridge_opportunity
        and saw_dudunsparce
        and target_state is not None
        and target_state["active_id"] == ALAKAZAM
        and attack_success
    )
    winner = record.get("winner")
    return {
        "opponent": record.get("opponent"),
        "game": record.get("game"),
        "turn_order": {"first": agent_first, "second": not agent_first},
        "outcome": {
            "winner": winner,
            "win": winner == 0,
            "loss": winner == 1,
            "draw": winner not in {0, 1},
            "error": bool(record.get("error")),
        },
        "t2": {
            "engine_turn": target_turn,
            "attack_success": attack_success,
            "opening_four_components": components,
            "draws_to_second_turn": draws,
            "dunsparce_bridge_opportunity": bridge_opportunity,
            "dunsparce_bridge_completed": bridge_success,
        },
        "attack_quality": _attack_quality_analysis(record, agent_label),
        "relay": _relay_analysis(record, agent_label),
    }


def _metric(numerator: int, denominator: int) -> dict[str, Any]:
    return {"numerator": numerator, "denominator": denominator, "rate": _rate(numerator, denominator)}


def _direction(results: list[Mapping[str, Any]], which: str) -> list[Mapping[str, Any]]:
    return [result for result in results if result.get("turn_order", {}).get(which)]


def _outcome_metrics(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    def one(group: list[Mapping[str, Any]]) -> dict[str, Any]:
        wins = sum(bool(result.get("outcome", {}).get("win")) for result in group)
        losses = sum(bool(result.get("outcome", {}).get("loss")) for result in group)
        draws = sum(bool(result.get("outcome", {}).get("draw")) for result in group)
        errors = sum(bool(result.get("outcome", {}).get("error")) for result in group)
        return {
            "games": len(group),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "errors": errors,
            "numerator": wins,
            "denominator": len(group),
            "rate": _rate(wins, len(group)),
        }

    return {
        "overall": one(results),
        "first": one(_direction(results, "first")),
        "second": one(_direction(results, "second")),
    }


def _t2_metrics(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    def one(group: list[Mapping[str, Any]]) -> dict[str, Any]:
        attack = sum(bool(result["t2"]["attack_success"]) for result in group)
        bridge_opportunity = sum(bool(result["t2"]["dunsparce_bridge_opportunity"]) for result in group)
        bridge_completed = sum(bool(result["t2"]["dunsparce_bridge_completed"]) for result in group)
        component_counts = {
            name: sum(bool(result["t2"]["opening_four_components"][name]) for result in group)
            for name in OPENING_COMPONENT_KEYS
        }
        return {
            "all_games": _metric(attack, len(group)),
            "opening_four_components": {
                "component_counts": component_counts,
                "sample_games": len(group),
            },
            "dunsparce_bridge": _metric(bridge_completed, bridge_opportunity),
        }

    overall = one(results)
    first = one(_direction(results, "first"))
    second = one(_direction(results, "second"))
    # Keep the explicit all_games name in the JSON contract. The nested
    # overall/first/second form is convenient for the HTML renderer, while
    # all_games makes the denominator unambiguous to downstream tooling.
    return {
        "overall": overall,
        "all_games": overall["all_games"],
        "first": first,
        "second": second,
    }


def _average_draw_metric(
    results: list[Mapping[str, Any]], field: str = "ability_draw_cards"
) -> dict[str, Any]:
    total = sum(int(result["t2"]["draws_to_second_turn"][field]) for result in results)
    return {
        "total": total,
        "games": len(results),
        "average": total / len(results) if results else None,
    }


def _second_turn_draw_metrics(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    reached = [
        result
        for result in results
        if result["t2"]["draws_to_second_turn"]["reached_second_turn"]
    ]
    metrics = {
        "all_games": _average_draw_metric(results),
        "reached_second_turn": _average_draw_metric(reached),
        "first": _average_draw_metric(_direction(results, "first")),
        "second": _average_draw_metric(_direction(results, "second")),
    }
    metrics["normal_draw_cards"] = {
        key: _average_draw_metric(group, "normal_draw_cards")
        for key, group in {
            "all_games": results,
            "reached_second_turn": reached,
            "first": _direction(results, "first"),
            "second": _direction(results, "second"),
        }.items()
    }
    return metrics


def _attack_quality_overall(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    submissions = sum(int(result["attack_quality"]["attack_submissions"]) for result in results)
    resolved = sum(int(result["attack_quality"]["resolved_attacks"]) for result in results)
    unresolved = sum(int(result["attack_quality"]["unresolved_attacks"]) for result in results)
    unknown_prize = sum(int(result["attack_quality"]["unknown_prize_attacks"]) for result in results)
    non_prize = sum(int(result["attack_quality"]["non_prize_attacks"]) for result in results)
    powerful_resolved = sum(
        int(result["attack_quality"]["powerful_hand"]["resolved_attacks"]) for result in results
    )
    powerful_unknown_prize = sum(
        int(result["attack_quality"]["powerful_hand"]["unknown_prize_attacks"])
        for result in results
    )
    powerful_non_prize = sum(
        int(result["attack_quality"]["powerful_hand"]["non_prize_attacks"])
        for result in results
    )
    return {
        "attack_submissions": submissions,
        "resolved_attacks": resolved,
        "unresolved_attacks": unresolved,
        "unknown_prize_attacks": unknown_prize,
        "non_prize_attacks": _metric(non_prize, resolved - unknown_prize),
        "powerful_hand": {
            "resolved_attacks": powerful_resolved,
            "unknown_prize_attacks": powerful_unknown_prize,
            "non_prize_attacks": _metric(
                powerful_non_prize, powerful_resolved - powerful_unknown_prize
            ),
        },
    }


def _attack_quality_metrics(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _attack_quality_overall(results),
        "first": _attack_quality_overall(_direction(results, "first")),
        "second": _attack_quality_overall(_direction(results, "second")),
    }


def _relay_overall(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    opportunities = sum(int(result["relay"]["opportunities"]) for result in results)
    successes = sum(int(result["relay"]["successes"]) for result in results)
    failures = {
        name: sum(int(result["relay"]["failure_counts"].get(name, 0)) for result in results)
        for name in RELAY_FAILURES
    }
    return {
        "opportunities": opportunities,
        "successes": successes,
        "success_rate": _rate(successes, opportunities),
        "failure_counts": failures,
    }


def _relay_metrics(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _relay_overall(results),
        "by_turn_order": {
            name: _relay_overall(_direction(results, name))
            for name in ("first", "second")
        },
    }


def _opponent_metrics(results: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result.get("opponent") or "unknown")].append(result)
    rows = []
    for opponent, group in sorted(grouped.items()):
        wins = sum(bool(result["outcome"]["win"]) for result in group)
        rows.append({"opponent": opponent, **_metric(wins, len(group))})
    return rows


def aggregate_iteration(
    results: list[Mapping[str, Any]], *, metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Aggregate game analyses without retaining full observations."""
    normalized = [dict(result) for result in results]
    metadata = dict(metadata or {})
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "metadata": metadata,
        "sample": {
            "games": len(normalized),
            "opponents": len({result.get("opponent") for result in normalized}),
            "errors": sum(bool(result.get("outcome", {}).get("error")) for result in normalized),
            "turn_order_games": {
                "first": len(_direction(normalized, "first")),
                "second": len(_direction(normalized, "second")),
            },
        },
        "metrics": {
            "win_rate": _outcome_metrics(normalized),
            "t2_alakazam": _t2_metrics(normalized),
            "second_turn_draws": _second_turn_draw_metrics(normalized),
            "non_prize_attacks": _attack_quality_metrics(normalized),
            "post_ko_relay": _relay_metrics(normalized),
        },
        "opponents": _opponent_metrics(normalized),
        "games": normalized,
    }


def _json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).replace("<", "\\u003c")


def _percent(value: Any) -> str:
    return "-" if value is None else f"{float(value):.1%}"


def _metric_text(metric: Mapping[str, Any]) -> str:
    return f"{_percent(metric.get('rate'))} ({metric.get('numerator', 0)}/{metric.get('denominator', 0)})"


def _draw_metric_text(metric: Mapping[str, Any]) -> str:
    average = metric.get("average")
    if average is None:
        return "-"
    return f"{float(average):.2f} 张 ({metric.get('total', 0)}/{metric.get('games', 0)})"


def _average_text(metric: Mapping[str, Any]) -> str:
    average = metric.get("average")
    return "-" if average is None else f"{float(average):.2f} 张"


def render_iteration_html(document: Mapping[str, Any], output_path: Path) -> None:
    """Render one iteration as a self-contained HTML document."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = document.get("metadata", {})
    metrics = document.get("metrics", {})
    win = metrics.get("win_rate", {})
    t2 = metrics.get("t2_alakazam", {})
    draws = metrics.get("second_turn_draws", {})
    attack_quality = metrics.get("non_prize_attacks", {})
    relay = metrics.get("post_ko_relay", {}).get("overall", {})
    title = html.escape(str(metadata.get("label") or metadata.get("iteration_id") or "AutoIteration"))
    cards = [
        ("胜率", _percent(win.get("overall", {}).get("rate")), "overall"),
        ("先手胜率", _percent(win.get("first", {}).get("rate")), "first"),
        ("后手胜率", _percent(win.get("second", {}).get("rate")), "second"),
        ("二回合胡地攻击", _percent(t2.get("overall", {}).get("all_games", {}).get("rate")), "t2"),
        ("第二回合平均过牌张数", _average_text(draws.get("all_games", {})), "draw"),
        (
            "攻击但未拿奖赏",
            _percent(attack_quality.get("overall", {}).get("non_prize_attacks", {}).get("rate")),
            "penalty",
        ),
        ("立即接力", _percent(relay.get("success_rate")), "relay"),
    ]
    card_html = "".join(
        f'<article class="card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong><small>{key}</small></article>'
        for label, value, key in cards
    )
    first = t2.get("first", {})
    second = t2.get("second", {})
    opening_profiles = {
        "overall": t2.get("overall", {}).get("opening_four_components", {}),
        "first": first.get("opening_four_components", {}),
        "second": second.get("opening_four_components", {}),
    }
    component_labels = {
        "active_abra": ("Active Abra", "第一回合开始时 Active 是 Abra"),
        "rare_candy": ("Rare Candy", "第一回合开始时手牌有 Rare Candy"),
        "alakazam_or_search": ("Alakazam 或检索路线", "Alakazam、Poke Pad、Hilda、Dawn 任一在手"),
        "psychic_energy_or_hilda": ("Psychic Energy 或 Hilda", "Psychic Energy 或 Hilda 任一在手"),
        "all_four": ("四项同时满足", "仅作状态观测，不作为机会或完成率"),
    }
    component_rows = "".join(
        "<tr>"
        f"<td>{html.escape(label)}</td><td>{html.escape(description)}</td>"
        + "".join(
            f"<td>{profile.get('component_counts', {}).get(key, 0)}/{profile.get('sample_games', 0)}</td>"
            for profile in opening_profiles.values()
        )
        + "</tr>"
        for key, (label, description) in component_labels.items()
    )
    bridge = t2.get("overall", {}).get("dunsparce_bridge", {})
    normal_draws = draws.get("normal_draw_cards", {}).get("all_games", {})
    draw_rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{_draw_metric_text(draws.get(key, {}))}</td></tr>"
        for key, label in (
            ("all_games", "全部对局"),
            ("reached_second_turn", "实际到达己方第二回合"),
            ("first", "先手实际样本"),
            ("second", "后手实际样本"),
        )
    )
    attack_quality_rows = "".join(
        "<tr>"
        f"<td>{html.escape(label)}</td>"
        f"<td>{_metric_text(profile.get('non_prize_attacks', {}))}</td>"
        f"<td>{_metric_text(profile.get('powerful_hand', {}).get('non_prize_attacks', {}))}</td>"
        f"<td>{profile.get('resolved_attacks', 0)}</td>"
        f"<td>{profile.get('powerful_hand', {}).get('resolved_attacks', 0)}</td>"
        "</tr>"
        for key, label in (
            ("overall", "总体"),
            ("first", "先手实际样本"),
            ("second", "后手实际样本"),
        )
        for profile in [attack_quality.get(key, {})]
    )
    failure_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{count}</td></tr>"
        for name, count in relay.get("failure_counts", {}).items()
    )
    opponent_rows = "".join(
        f"<tr><td>{html.escape(str(row['opponent']))}</td><td>{_metric_text(row)}</td></tr>"
        for row in document.get("opponents", [])
    )
    content = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
:root {{ color-scheme: light; --ink:#1d2935; --muted:#64717e; --line:#d9e0e6; --accent:#0d6b68; --wash:#f5f8f8; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:#eef2f3; color:var(--ink); font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:32px 20px 56px; }} header {{ border-bottom:1px solid var(--line); padding-bottom:20px; margin-bottom:22px; }}
h1 {{ margin:0 0 6px; font-size:28px; }} h2 {{ font-size:18px; margin:0 0 14px; }} p {{ color:var(--muted); margin:5px 0; }}
.cards {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-bottom:24px; }}
.card, section {{ background:white; border:1px solid var(--line); border-radius:6px; }} .card {{ padding:16px; min-height:110px; }}
.card span,.card small {{ display:block; color:var(--muted); }} .card strong {{ display:block; font-size:28px; margin:10px 0 4px; }}
.grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; margin-bottom:16px; }} section {{ padding:18px; overflow:auto; }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ padding:8px 6px; border-bottom:1px solid #edf0f2; text-align:left; }} th {{ color:var(--muted); font-weight:600; }}
.notice {{ background:var(--wash); border-left:3px solid var(--accent); padding:10px 12px; color:#42515d; margin-top:12px; }}
@media (max-width:760px) {{ .cards,.grid {{ grid-template-columns:1fr; }} main {{ padding:20px 12px 40px; }} }}
</style></head><body><main>
<header><h1>{title}</h1><p>Iteration: {html.escape(str(metadata.get('iteration_id', '-')))} · Profile: {html.escape(str(metadata.get('profile', '-')))}</p>
<p>{html.escape(str(metadata.get('change_summary', '')))}</p></header>
<div class="cards">{card_html}</div>
<div class="grid"><section><h2>胜率与先后手</h2><table><tr><th>分组</th><th>胜率</th><th>样本</th><th>胜/负/和</th></tr>
<tr><td>总体</td><td>{_metric_text(win.get('overall', {}))}</td><td>{win.get('overall', {}).get('games', 0)}</td><td>{win.get('overall', {}).get('wins', 0)}/{win.get('overall', {}).get('losses', 0)}/{win.get('overall', {}).get('draws', 0)}</td></tr>
<tr><td>先手</td><td>{_metric_text(win.get('first', {}))}</td><td>{win.get('first', {}).get('games', 0)}</td><td>{win.get('first', {}).get('wins', 0)}/{win.get('first', {}).get('losses', 0)}/{win.get('first', {}).get('draws', 0)}</td></tr>
<tr><td>后手</td><td>{_metric_text(win.get('second', {}))}</td><td>{win.get('second', {}).get('games', 0)}</td><td>{win.get('second', {}).get('wins', 0)}/{win.get('second', {}).get('losses', 0)}/{win.get('second', {}).get('draws', 0)}</td></tr></table></section>
<section><h2>二回合胡地实际攻击</h2><table><tr><th>分组</th><th>实际选择 attackId=1072</th><th>无 Abra 的 Dunsparce→Dudunsparce 换位</th></tr>
<tr><td>总体</td><td>{_metric_text(t2.get('overall', {}).get('all_games', {}))}</td><td>{_metric_text(bridge)}</td></tr>
<tr><td>先手</td><td>{_metric_text(first.get('all_games', {}))}</td><td>{_metric_text(first.get('dunsparce_bridge', {}))}</td></tr>
<tr><td>后手</td><td>{_metric_text(second.get('all_games', {}))}</td><td>{_metric_text(second.get('dunsparce_bridge', {}))}</td></tr></table>
<div class="notice">二回合指标合并 17 个对手展示；四组件只用于解释起始资源状态，不作为机会率或完成率评估。</div></section></div>
<div class="grid"><section><h2>四组件状态（仅状态观测）</h2><table><tr><th>组件</th><th>语义</th><th>总体</th><th>先手</th><th>后手</th></tr>{component_rows}</table></section>
<section><h2>第二回合平均过牌张数</h2><table><tr><th>分组</th><th>能力/效果抽牌平均值</th></tr>{draw_rows}</table>
<div class="notice">只统计从己方第一回合开始到己方第二回合结束的额外 Draw 事件；回合开始的正常抽牌单独记录，共 {_draw_metric_text(normal_draws)}，不计入主指标。格式为平均张数（总张数 / 样本数）。</div></section></div>
<section><h2>第三优先级：攻击但未拿奖赏</h2><table><tr><th>分组</th><th>全部攻击未拿奖赏</th><th>Powerful Hand 未拿奖赏</th><th>已结算攻击</th><th>Powerful Hand 攻击</th></tr>{attack_quality_rows}</table>
<div class="notice">这是惩罚项，主分母是已结算攻击次数。它只标记结果，不单独证明是手牌不足、伤害不足、Boss 漏用或特殊能量导致；后续应结合 action、手牌、对手 Active/能量和可用选项继续归因。</div></section>
<section><h2>Post-KO 接力</h2><table><tr><th>指标</th><th>结果</th></tr><tr><td>立即接力成功</td><td>{relay.get('successes', 0)} / {relay.get('opportunities', 0)} ({_percent(relay.get('success_rate'))})</td></tr>{failure_rows}</table>
<div class="notice">recoverable_discard_miss 只表示 trace 中看见攻击线弃牌资源的保守候选，不证明当时一定存在合法回收动作；需要 evaluator 提供选项级事实后再升级归因。</div></section></div>
<section><h2>按对手胜率</h2><table><tr><th>对手</th><th>胜率</th><th>胜</th><th>样本</th></tr>{''.join(f"<tr><td>{html.escape(str(row['opponent']))}</td><td>{_percent(row.get('rate'))}</td><td>{row.get('numerator', 0)}</td><td>{row.get('denominator', 0)}</td></tr>" for row in document.get('opponents', []))}</table></section>
<script id="iteration-data" type="application/json">{_json_for_script(document)}</script>
</main></body></html>"""
    output_path.write_text(content, encoding="utf-8")


def render_history_index(documents: list[Mapping[str, Any]], output_path: Path) -> None:
    """Render the cross-iteration summary page."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for document in sorted(documents, key=lambda item: str(item.get("metadata", {}).get("iteration_id", ""))):
        metadata = document.get("metadata", {})
        metrics = document.get("metrics", {})
        win = metrics.get("win_rate", {}).get("overall", {})
        win_first = metrics.get("win_rate", {}).get("first", {})
        win_second = metrics.get("win_rate", {}).get("second", {})
        t2 = metrics.get("t2_alakazam", {}).get("overall", {}).get("all_games", {})
        draws = metrics.get("second_turn_draws", {}).get("all_games", {})
        attack_quality = metrics.get("non_prize_attacks", {}).get("overall", {})
        relay = metrics.get("post_ko_relay", {}).get("overall", {})
        iteration_id = str(metadata.get("iteration_id", "unknown"))
        rows.append(
            f"<tr><td><a href=\"{html.escape(iteration_id)}/index.html\">{html.escape(iteration_id)}</a></td>"
            f"<td>{_metric_text(win)}</td><td>{_metric_text(win_first)}</td><td>{_metric_text(win_second)}</td>"
            f"<td>{_metric_text(t2)}</td><td>{_average_text(draws)}</td>"
            f"<td>{_percent(attack_quality.get('non_prize_attacks', {}).get('rate'))}</td>"
            f"<td>{_percent(attack_quality.get('powerful_hand', {}).get('non_prize_attacks', {}).get('rate'))}</td>"
            f"<td>{_percent(relay.get('success_rate'))}</td><td>{relay.get('failure_counts', {}).get('recoverable_discard_miss', 0)}</td>"
            f"<td>{html.escape(str(metadata.get('decision', '-')))}</td></tr>"
        )
    content = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AutoIteration History</title><style>
body {{ margin:0; background:#eef2f3; color:#1d2935; font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:32px 20px 56px; }} section {{ background:#fff; border:1px solid #d9e0e6; border-radius:6px; padding:20px; overflow:auto; }}
h1 {{ margin:0 0 6px; font-size:28px; }} p {{ color:#64717e; }} table {{ width:100%; border-collapse:collapse; margin-top:18px; }} th,td {{ padding:10px 8px; border-bottom:1px solid #edf0f2; text-align:left; white-space:nowrap; }} th {{ color:#64717e; }} a {{ color:#0d6b68; }}
</style></head><body><main><section><h1>AutoIteration History</h1><p>跨 iteration 的主要指标趋势。每个 iteration 页面保留完整分母与分类。</p>
<table><tr><th>Iteration</th><th>总体胜率</th><th>先手胜率</th><th>后手胜率</th><th>二回合胡地攻击</th><th>第二回合平均过牌张数</th><th>全部攻击未拿奖赏</th><th>Powerful Hand 未拿奖赏</th><th>立即接力</th><th>可回收弃牌漏做</th><th>Decision</th></tr>{''.join(rows) or '<tr><td colspan="11">暂无 iteration 结果</td></tr>'}</table></section>
<script id="history-data" type="application/json">{_json_for_script(documents)}</script></main></body></html>"""
    output_path.write_text(content, encoding="utf-8")


def _load_trace_records(trace_root: Path) -> list[dict[str, Any]]:
    paths = sorted(trace_root.glob("**/game_*.json"))
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        if isinstance(record, dict):
            records.append(record)
    return records


def _write_iteration_markdown(document: Mapping[str, Any], output_path: Path) -> None:
    metadata = document.get("metadata", {})
    metrics = document.get("metrics", {})
    win = metrics.get("win_rate", {}).get("overall", {})
    win_first = metrics.get("win_rate", {}).get("first", {})
    win_second = metrics.get("win_rate", {}).get("second", {})
    t2 = metrics.get("t2_alakazam", {}).get("overall", {}).get("all_games", {})
    t2_by_order = metrics.get("t2_alakazam", {})
    component_state = metrics.get("t2_alakazam", {}).get("overall", {}).get("opening_four_components", {})
    component_counts = component_state.get("component_counts", {})
    component_sample = component_state.get("sample_games", 0)
    component_labels = {
        "active_abra": "Active Abra",
        "rare_candy": "Rare Candy",
        "alakazam_or_search": "Alakazam 或检索路线",
        "psychic_energy_or_hilda": "Psychic Energy 或 Hilda",
        "all_four": "四项同时满足",
    }
    component_summary = "；".join(
        f"{label} {component_counts.get(key, 0)}/{component_sample}"
        for key, label in component_labels.items()
    )
    draws = metrics.get("second_turn_draws", {})
    attack_quality = metrics.get("non_prize_attacks", {})
    relay = metrics.get("post_ko_relay", {}).get("overall", {})
    text = f"""# {metadata.get('iteration_id', 'iteration')}

- 变更说明：{metadata.get('change_summary', '')}
- 指标 profile：`{metadata.get('profile', '-')}`
- Decision：`{metadata.get('decision', '-')}`
- 样本：{document.get('sample', {}).get('games', 0)} 局，{document.get('sample', {}).get('opponents', 0)} 个对手；先手 {document.get('sample', {}).get('turn_order_games', {}).get('first', 0)} 局，后手 {document.get('sample', {}).get('turn_order_games', {}).get('second', 0)} 局

## 结果

- 总体胜率：{_metric_text(win)}
- 先手胜率：{_metric_text(win_first)}
- 后手胜率：{_metric_text(win_second)}
- 二回合实际选择 `attackId=1072`：{_metric_text(t2)}
- 二回合实际攻击（先手 / 后手）：{_metric_text(t2_by_order.get('first', {}).get('all_games', {}))} / {_metric_text(t2_by_order.get('second', {}).get('all_games', {}))}
- 四组件状态（仅观测）：{component_summary}
- 第二回合平均过牌张数（能力/效果抽牌）：全部对局 {_draw_metric_text(draws.get('all_games', {}))}；实际到达二回合 {_draw_metric_text(draws.get('reached_second_turn', {}))}
- 第二回合平均过牌张数（先手 / 后手）：{_draw_metric_text(draws.get('first', {}))} / {_draw_metric_text(draws.get('second', {}))}
- 攻击但未拿奖赏（全部攻击）：{_metric_text(attack_quality.get('overall', {}).get('non_prize_attacks', {}))}
- 攻击但未拿奖赏（Powerful Hand）：{_metric_text(attack_quality.get('overall', {}).get('powerful_hand', {}).get('non_prize_attacks', {}))}
- 攻击结算审计：已结算 {attack_quality.get('overall', {}).get('resolved_attacks', 0)} 次，未完成 {attack_quality.get('overall', {}).get('unresolved_attacks', 0)} 次，奖赏状态未知 {attack_quality.get('overall', {}).get('unknown_prize_attacks', 0)} 次
- Post-KO 立即接力：{relay.get('successes', 0)} / {relay.get('opportunities', 0)}（{_percent(relay.get('success_rate'))}）
- `recoverable_discard_miss`：{relay.get('failure_counts', {}).get('recoverable_discard_miss', 0)}

## 口径限制

本报告只使用 trace 可见事实。四组件只记录第一回合起始资源状态，不参与主要评估门槛；Alakazam 组件包含 Alakazam、Poke Pad、Hilda、Dawn，Psychic Energy 组件包含 Psychic Energy 或 Hilda。过牌主指标排除回合开始的正常抽牌。攻击但未拿奖赏是结果惩罚项，不能单独证明是手牌不足、伤害不足、Boss 漏用或特殊能量造成；后续 evaluator 应补充选项级资源可达性。`recoverable_discard_miss` 是弃牌区出现攻击线资源的保守候选，不能单独证明当时一定有合法回收选项。
"""
    output_path.write_text(text, encoding="utf-8")


def build_iteration(
    trace_root: Path,
    history_root: Path,
    *,
    iteration_id: str,
    profile: str,
    profile_revision: int,
    label: str,
    change_summary: str,
    decision: str,
    agent_label: str,
) -> dict[str, Any]:
    records = _load_trace_records(trace_root)
    analyzed = [analyze_game_record(record, agent_label=agent_label) for record in records]
    metadata = {
        "iteration_id": iteration_id,
        "profile": profile,
        "profile_revision": profile_revision,
        "label": label,
        "change_summary": change_summary,
        "decision": decision,
        "agent_label": agent_label,
        "trace_root": str(trace_root),
    }
    document = aggregate_iteration(analyzed, metadata=metadata)
    iteration_root = history_root / iteration_id
    iteration_root.mkdir(parents=True, exist_ok=True)
    (iteration_root / "result.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render_iteration_html(document, iteration_root / "index.html")
    _write_iteration_markdown(document, iteration_root / "iteration.md")
    existing = []
    for result_path in sorted(history_root.glob("*/result.json")):
        with result_path.open(encoding="utf-8") as handle:
            existing.append(json.load(handle))
    render_history_index(existing, history_root / "index.html")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--iteration-id", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--profile-revision", type=int, default=2)
    parser.add_argument("--label", default="")
    parser.add_argument("--change-summary", default="")
    parser.add_argument("--decision", default="observe")
    parser.add_argument("--agent-label", required=True)
    args = parser.parse_args()
    document = build_iteration(
        args.traces,
        args.history_root,
        iteration_id=args.iteration_id,
        profile=args.profile,
        profile_revision=args.profile_revision,
        label=args.label,
        change_summary=args.change_summary,
        decision=args.decision,
        agent_label=args.agent_label,
    )
    print(json.dumps(document["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
