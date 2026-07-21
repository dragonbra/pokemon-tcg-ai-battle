from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import html

from .base import AggregateMetric, GameContext, GameMetric, MetricPresentation
from .powerful_hand import POWERFUL_HAND_ATTACK_ID, _is_candidate_step
from .trace_utils import (
    active,
    agent_turn_target,
    as_int,
    bench,
    current,
    logs,
    normalized_evidence,
    option_list,
    player_at,
    selected_attack_id,
    lifecycle_status,
    trace_steps,
)


ALAKAZAM = 743
ABRA = 741
DUNSPARCE = 305
DUDUNSPARCE = 66
RARE_CANDY = 1079
POKE_PAD = 1152
HILDA = 1225
DAWN = 1231
PSYCHIC_ENERGIES = {5, 19}
DRAW_LOG = 4
TURN_START_LOG = 2
OPENING_COMPONENT_KEYS = (
    "active_abra",
    "rare_candy",
    "alakazam_or_search",
    "psychic_energy_or_hilda",
    "all_four",
)


def _candidate_steps(trace: dict, physical_index: int) -> list[dict]:
    return [
        step
        for step in trace_steps(trace)
        if _is_candidate_step(step, physical_index)
    ]


def _card_ids(cards: object) -> set[int]:
    if not isinstance(cards, list):
        return set()
    return {
        card_id
        for card in cards
        if isinstance(card, dict)
        for card_id in [as_int(card.get("id"))]
        if card_id is not None
    }


def _snapshot(step: dict, physical_index: int) -> dict[str, object]:
    player = player_at(step, physical_index)
    active_card = active(player)
    field = [active_card, *bench(player)] if active_card else bench(player)
    return {
        "turn": as_int(current(step).get("turn")),
        "active_id": as_int(active_card.get("id")) if active_card else None,
        "active_serial": as_int(active_card.get("serial")) if active_card else None,
        "field_ids": _card_ids(field),
        "hand_ids": _card_ids(player.get("hand")),
        "discard_ids": _card_ids(player.get("discard")),
        "deck_count": as_int(player.get("deckCount"), 0) or 0,
    }


def _first_state(
    steps: list[dict], physical_index: int, first_turn: int
) -> tuple[int, dict] | None:
    candidates = [
        (index, step)
        for index, step in enumerate(steps)
        if as_int(current(step).get("turn")) == first_turn
    ]
    candidates.extend(
        (index, step)
        for index, step in enumerate(steps)
        if as_int(current(step).get("turn")) != first_turn
    )
    for index, step in candidates:
        snapshot = _snapshot(step, physical_index)
        if snapshot["active_id"] is not None:
            return index, snapshot
    return None


def _opening_components(snapshot: dict[str, object] | None) -> dict[str, bool]:
    if snapshot is None:
        return {key: False for key in OPENING_COMPONENT_KEYS}
    hand_ids = snapshot["hand_ids"]
    if not isinstance(hand_ids, set):
        hand_ids = set()
    components = {
        "active_abra": snapshot["active_id"] == ABRA,
        "rare_candy": RARE_CANDY in hand_ids,
        "alakazam_or_search": bool(hand_ids & {ALAKAZAM, POKE_PAD, HILDA, DAWN}),
        "psychic_energy_or_hilda": bool(hand_ids & (PSYCHIC_ENERGIES | {HILDA})),
    }
    components["all_four"] = all(components.values())
    return components


def _draws_to_second_turn(
    steps: list[dict], physical_index: int, first_turn: int, target_turn: int
) -> dict[str, object]:
    reached = False
    ability_draw_cards = 0
    normal_draw_cards = 0
    for step in steps:
        turn = as_int(current(step).get("turn"))
        if turn is None or turn < first_turn or turn > target_turn:
            continue
        if turn == target_turn:
            reached = True
        own_logs = [
            log for log in logs(step) if as_int(log.get("playerIndex")) == physical_index
        ]
        draws = sum(as_int(log.get("type")) == DRAW_LOG for log in own_logs)
        if any(as_int(log.get("type")) == TURN_START_LOG for log in own_logs):
            normal_draw_cards += draws
        else:
            ability_draw_cards += draws
    return {
        "reached_second_turn": reached,
        "ability_draw_cards": ability_draw_cards,
        "normal_draw_cards": normal_draw_cards,
        "total_draw_cards": ability_draw_cards + normal_draw_cards,
    }


class SetupRelayPlugin:
    metric_id = "setup_relay"

    def render(
        self, aggregate: AggregateMetric, results: tuple[GameMetric, ...]
    ) -> MetricPresentation:
        payload = aggregate.payload
        bridge = payload.get("dunsparce_bridge", {})
        draws = payload.get("second_turn_draws", {})
        bridge_values = bridge if isinstance(bridge, dict) else {}
        draw_values = draws if isinstance(draws, dict) else {}
        all_draws = draw_values.get("all_games", {})
        normal_draws = draw_values.get("normal_draw_cards", {})
        all_draw_values = all_draws if isinstance(all_draws, dict) else {}
        normal_draw_values = normal_draws if isinstance(normal_draws, dict) else {}
        numerator = bridge_values.get("numerator", 0)
        denominator = bridge_values.get("denominator", 0)
        ability_average = all_draw_values.get("average")
        normal_average = normal_draw_values.get("all_games", {})
        if not isinstance(normal_average, dict):
            normal_average = {}
        normal_average_value = normal_average.get("average")
        markdown = "\n".join(
            (
                "## Setup and relay",
                "",
                f"- bridge opportunities: {denominator}",
                f"- bridge completions: {numerator}",
                f"- ability draws / game: {ability_average}",
                f"- normal draws / game: {normal_average_value}",
                "",
            )
        )
        html_fragment = (
            '<section class="metric-plugin"><h2>Setup and relay</h2>'
            "<table><thead><tr><th>field</th><th>value</th></tr></thead><tbody>"
            f"<tr><td>bridge opportunities</td><td>{html.escape(str(denominator))}</td></tr>"
            f"<tr><td>bridge completions</td><td>{html.escape(str(numerator))}</td></tr>"
            f"<tr><td>second-turn draws</td><td>{html.escape(str(ability_average))}</td></tr>"
            f"<tr><td>normal draws</td><td>{html.escape(str(normal_average_value))}</td></tr>"
            "</tbody></table></section>"
        )
        return MetricPresentation(self.metric_id, "Setup and relay", markdown, html_fragment)

    def analyze_game(self, trace: dict, context: GameContext) -> GameMetric:
        lifecycle = lifecycle_status(trace)
        physical_index = context.candidate_physical_index
        first_turn = 1 if context.candidate_first else 2
        target_turn = agent_turn_target(trace, physical_index) or (
            3 if context.candidate_first else 4
        )
        steps = _candidate_steps(trace, physical_index)
        first_state_entry = _first_state(steps, physical_index, first_turn)
        first_state = first_state_entry[1] if first_state_entry else None
        target_steps = [
            step for step in steps if as_int(current(step).get("turn")) == target_turn
        ]
        reached = bool(target_steps)
        target_state = _snapshot(target_steps[0], physical_index) if target_steps else None
        attack_success = any(
            selected_attack_id(step) == POWERFUL_HAND_ATTACK_ID for step in target_steps
        )
        saw_dudunsparce = any(
            DUDUNSPARCE in _snapshot(step, physical_index)["field_ids"]
            for step in steps
            if (as_int(current(step).get("turn")) or 0) <= target_turn
        )
        bridge_opportunity = bool(
            first_state
            and first_state["active_id"] == DUNSPARCE
        )
        bridge_completed = bool(
            bridge_opportunity
            and saw_dudunsparce
            and target_state
            and target_state["active_id"] == ALAKAZAM
            and attack_success
        )
        components = _opening_components(first_state)
        draws = _draws_to_second_turn(steps, physical_index, first_turn, target_turn)
        payload = {
            "opening_components": components,
            "all_four": components["all_four"],
            "bridge_opportunity": bridge_opportunity,
            "bridge_completed": bridge_completed,
            "draws_to_second_turn": draws,
            "target_turn": target_turn,
            "reached_second_turn": reached,
            "attack_success": attack_success,
            "evidence_strength": "proxy" if bridge_completed else "observed",
            "games": 1,
            "error_games": int(lifecycle == "error"),
            "unfinished_games": int(lifecycle == "unfinished"),
            "lifecycle_status": lifecycle,
        }
        evidence_step = target_steps[0] if target_steps else (steps[-1] if steps else None)
        evidence_index = (
            trace_steps(trace).index(evidence_step) if evidence_step in trace_steps(trace) else 0
        )
        evidence = normalized_evidence(
            trace,
            evidence_step,
            fallback_step=evidence_index,
            candidate_physical_index=physical_index,
        )
        evidence.update({"bridge_opportunity": bridge_opportunity, "bridge_completed": bridge_completed})
        return GameMetric(
            metric_id=self.metric_id,
            status=(
                "error"
                if lifecycle == "error"
                else "unavailable"
                if lifecycle == "unfinished"
                else "success"
                if reached
                else "unavailable"
            ),
            numerator=int(attack_success),
            denominator=1,
            value="attack_selected" if attack_success else "not_selected",
            evidence=(evidence,),
            diagnostics=(
                {
                    "game_id": context.game_id,
                    "opponent": context.opponent_name,
                    "turn_order": "first" if context.candidate_first else "second",
                    "candidate_first": context.candidate_first,
                },
            ),
            payload=payload,
        )

    def aggregate(self, results: list[GameMetric]) -> AggregateMetric:
        values = list(results)
        numerator = sum(result.numerator for result in values)
        denominator = sum(result.denominator for result in values)
        by_turn_order = {
            turn_order: _aggregate_group(
                result
                for result in values
                if result.diagnostics
                and isinstance(result.diagnostics[0], dict)
                and result.diagnostics[0].get("turn_order") == turn_order
            )
            for turn_order in ("first", "second")
        }
        components = {
            key: sum(
                bool(result.payload.get("opening_components", {}).get(key))
                for result in values
                if isinstance(result.payload.get("opening_components"), dict)
            )
            for key in OPENING_COMPONENT_KEYS
        }
        error_games = sum(
            int(result.payload.get("error_games", result.status == "error"))
            for result in values
        )
        unfinished_games = sum(
            int(result.payload.get("unfinished_games", result.status == "unfinished"))
            for result in values
        )
        return AggregateMetric(
            metric_id=self.metric_id,
            numerator=numerator,
            denominator=denominator,
            value=numerator / denominator if denominator else None,
            by_opponent=_by_opponent(values),
            payload={
                "games": len(values),
                "error_games": error_games,
                "unfinished_games": unfinished_games,
                "all_games": _aggregate_group(values),
                "by_turn_order": by_turn_order,
                "opening_four_components": {
                    "component_counts": components,
                    "sample_games": len(values),
                },
                "dunsparce_bridge": {
                    "numerator": sum(bool(result.payload.get("bridge_completed")) for result in values),
                    "denominator": sum(bool(result.payload.get("bridge_opportunity")) for result in values),
                },
                "second_turn_draws": _draw_aggregate(values),
            },
        )


def _aggregate_group(results: Iterable[GameMetric]) -> dict[str, object]:
    values = list(results)
    attack = sum(result.numerator for result in values)
    opportunities = sum(bool(result.payload.get("bridge_opportunity")) for result in values)
    successes = sum(bool(result.payload.get("bridge_completed")) for result in values)
    error_games = sum(
        int(result.payload.get("error_games", result.status == "error"))
        for result in values
    )
    unfinished_games = sum(
        int(result.payload.get("unfinished_games", result.status == "unfinished"))
        for result in values
    )
    return {
        "games": len(values),
        "error_games": error_games,
        "unfinished_games": unfinished_games,
        "attack_numerator": attack,
        "attack_denominator": len(values),
        "attack_rate": attack / len(values) if values else None,
        "bridge_opportunities": opportunities,
        "bridge_successes": successes,
        "bridge_rate": successes / opportunities if opportunities else None,
    }


def _draw_average(results: list[GameMetric], field: str, *, reached_only: bool = False) -> dict[str, object]:
    values = [
        result
        for result in results
        if not reached_only or result.payload.get("draws_to_second_turn", {}).get("reached_second_turn")
    ]
    total = sum(
        int(result.payload.get("draws_to_second_turn", {}).get(field, 0))
        for result in values
    )
    return {"total": total, "games": len(values), "average": total / len(values) if values else None}


def _draw_aggregate(results: list[GameMetric]) -> dict[str, object]:
    return {
        "all_games": _draw_average(results, "ability_draw_cards"),
        "reached_second_turn": _draw_average(results, "ability_draw_cards", reached_only=True),
        "first": _draw_average(
            [result for result in results if result.diagnostics and result.diagnostics[0].get("turn_order") == "first"],
            "ability_draw_cards",
        ),
        "second": _draw_average(
            [result for result in results if result.diagnostics and result.diagnostics[0].get("turn_order") == "second"],
            "ability_draw_cards",
        ),
        "normal_draw_cards": {
            "all_games": _draw_average(results, "normal_draw_cards"),
            "reached_second_turn": _draw_average(results, "normal_draw_cards", reached_only=True),
            "first": _draw_average(
                [result for result in results if result.diagnostics and result.diagnostics[0].get("turn_order") == "first"],
                "normal_draw_cards",
            ),
            "second": _draw_average(
                [result for result in results if result.diagnostics and result.diagnostics[0].get("turn_order") == "second"],
                "normal_draw_cards",
            ),
        },
    }


def _by_opponent(results: Iterable[GameMetric]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[GameMetric]] = defaultdict(list)
    for result in results:
        diagnostic = result.diagnostics[0] if result.diagnostics else {}
        opponent = str(diagnostic.get("opponent", "unknown")) if isinstance(diagnostic, dict) else "unknown"
        grouped[opponent].append(result)
    return {opponent: _aggregate_group(values) for opponent, values in grouped.items()}
