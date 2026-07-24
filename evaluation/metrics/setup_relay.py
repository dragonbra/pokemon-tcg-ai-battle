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
    candidate_index,
    current,
    logs,
    normalized_evidence,
    option_list,
    player_at,
    selected_attack_id,
    selected_options,
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
        "active_card": active_card,
        "bench_cards": bench(player),
        "field_ids": _card_ids(field),
        "hand_ids": _card_ids(player.get("hand")),
        "discard_ids": _card_ids(player.get("discard")),
        "deck_count": as_int(player.get("deckCount"), 0) or 0,
    }


def _evolved_from(card: object, source_id: int, source_serial: int | None) -> bool:
    if not isinstance(card, dict):
        return False
    previous_cards = [
        previous
        for previous in card.get("preEvolution") or []
        if isinstance(previous, dict)
    ]
    if source_serial is not None:
        return any(
            as_int(previous.get("id")) == source_id
            and as_int(previous.get("serial")) == source_serial
            for previous in previous_cards
        )
    return any(as_int(previous.get("id")) == source_id for previous in previous_cards)


def _has_psychic_energy(card: object) -> bool:
    if not isinstance(card, dict):
        return False
    return any(
        (
            as_int(energy.get("id"))
            if isinstance(energy, dict)
            else as_int(energy)
        )
        in PSYCHIC_ENERGIES
        for energy in card.get("energies") or []
    )


def _selected_run_away_draw(step: dict, physical_index: int) -> bool:
    active_card = active(player_at(step, physical_index))
    for option in selected_options(step):
        if as_int(option.get("type")) != 10:
            continue
        card_id = next(
            (
                value
                for key in ("cardId", "card_id", "id")
                for value in [as_int(option.get(key))]
                if value is not None
            ),
            None,
        )
        if card_id is None and as_int(option.get("area", option.get("inPlayArea"))) == 4:
            card_id = as_int(active_card.get("id")) if active_card else None
        if card_id == DUDUNSPARCE:
            return True
    return False


def _bridge_route(
    target_steps: list[dict],
    physical_index: int,
    opening_dunsparce_serial: int | None,
    first_turn_abra_serials: set[int],
) -> dict[str, object]:
    milestone = 0
    handoff_abra_serial: int | None = None
    attacking_alakazam_serial: int | None = None
    evidence_step: dict | None = None
    components = {
        "evolved_opening_dunsparce": False,
        "used_run_away_draw": False,
        "abra_active_after_ability": False,
        "evolved_handoff_abra": False,
        "active_psychic_at_attack": False,
        "submitted_powerful_hand": False,
    }
    for step in target_steps:
        active_card = active(player_at(step, physical_index))
        active_id = as_int(active_card.get("id")) if active_card else None
        active_serial = as_int(active_card.get("serial")) if active_card else None

        if (
            milestone == 0
            and active_id == DUDUNSPARCE
            and _evolved_from(active_card, DUNSPARCE, opening_dunsparce_serial)
        ):
            components["evolved_opening_dunsparce"] = True
            milestone = 1
        if milestone == 1 and _selected_run_away_draw(step, physical_index):
            components["used_run_away_draw"] = True
            milestone = 2
        if (
            milestone == 2
            and active_id == ABRA
            and active_serial in first_turn_abra_serials
        ):
            components["abra_active_after_ability"] = True
            handoff_abra_serial = active_serial
            milestone = 3
        if (
            milestone == 3
            and active_id == ALAKAZAM
            and _evolved_from(active_card, ABRA, handoff_abra_serial)
        ):
            components["evolved_handoff_abra"] = True
            attacking_alakazam_serial = active_serial
            milestone = 4
        if (
            milestone == 4
            and active_id == ALAKAZAM
            and active_serial == attacking_alakazam_serial
            and selected_attack_id(step) == POWERFUL_HAND_ATTACK_ID
        ):
            components["submitted_powerful_hand"] = True
            components["active_psychic_at_attack"] = _has_psychic_energy(active_card)
            evidence_step = step
            if components["active_psychic_at_attack"]:
                milestone = 5
                break

    return {
        "completed": milestone == 5,
        "components": components,
        "evidence_step": evidence_step,
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
        physical_index = candidate_index(trace, context.candidate_physical_index)
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
        attack_success = any(
            selected_attack_id(step) == POWERFUL_HAND_ATTACK_ID for step in target_steps
        )
        opening_dunsparce_without_abra = bool(
            first_state
            and first_state["active_id"] == DUNSPARCE
            and ABRA not in first_state["hand_ids"]
            and ABRA not in first_state["field_ids"]
        )
        first_turn_abra_serials = {
            serial
            for step in steps
            if as_int(current(step).get("turn")) == first_turn
            for card in bench(player_at(step, physical_index))
            if as_int(card.get("id")) == ABRA
            for serial in [as_int(card.get("serial"))]
            if serial is not None
        }
        bridge_opportunity = bool(
            opening_dunsparce_without_abra and first_turn_abra_serials
        )
        route = _bridge_route(
            target_steps,
            physical_index,
            as_int(first_state.get("active_serial")) if first_state else None,
            first_turn_abra_serials,
        )
        bridge_completed = bool(bridge_opportunity and route["completed"])
        components = _opening_components(first_state)
        draws = _draws_to_second_turn(steps, physical_index, first_turn, target_turn)
        payload = {
            "opening_components": components,
            "all_four": components["all_four"],
            "bridge_opportunity": bridge_opportunity,
            "bridge_completed": bridge_completed,
            "bridge_components": {
                "opening_dunsparce_without_abra": opening_dunsparce_without_abra,
                "first_turn_bench_abra": bool(first_turn_abra_serials),
                **route["components"],
            },
            "draws_to_second_turn": draws,
            "target_turn": target_turn,
            "reached_second_turn": reached,
            "attack_success": attack_success,
            "evidence_strength": "observed",
            "games": 1,
            "error_games": int(lifecycle == "error"),
            "unfinished_games": int(lifecycle == "unfinished"),
            "lifecycle_status": lifecycle,
        }
        evidence_step = route["evidence_step"] or (
            target_steps[-1] if target_steps else (steps[-1] if steps else None)
        )
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
