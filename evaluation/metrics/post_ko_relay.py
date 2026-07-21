from __future__ import annotations

from collections import defaultdict

from .base import AggregateMetric, GameContext, GameMetric
from .trace_utils import (
    active,
    as_int,
    bench,
    candidate_index,
    field_state,
    knockout_is_confirmed,
    logs,
    lifecycle_status,
    normalized_evidence,
    player_at,
    trace_steps,
)


READY_ATTACKER_CARD_IDS = {742, 743}
PSYCHIC_ENERGY = 5
RELAY_ROUTE_CARD_IDS = {741, 742, 743}
RELAY_FAILURES = (
    "field_route_miss",
    "recoverable_discard_miss",
    "recoverable_route_incomplete",
    "nonterminal_no_field_route",
    "terminal_no_resource",
)


def _ready_attacker_count(step: dict, physical_index: int) -> int:
    player = player_at(step, physical_index)
    return sum(
        as_int(card.get("id")) in READY_ATTACKER_CARD_IDS
        and PSYCHIC_ENERGY in (card.get("energies") or [])
        for card in [active(player), *bench(player)]
        if card
    )


class PostKORelayPlugin:
    metric_id = "post_ko_relay"

    def analyze_game(self, trace: dict, context: GameContext) -> GameMetric:
        lifecycle = lifecycle_status(trace)
        physical_index = candidate_index(trace, context.candidate_physical_index)
        previous_field: dict[int, dict] = {}
        seen: set[tuple[int, int]] = set()
        events: list[dict[str, object]] = []
        evidence: list[dict[str, object]] = []

        for index, step in enumerate(trace_steps(trace)):
            for log in logs(step):
                serial = as_int(log.get("serial"))
                destination = as_int(log.get("toArea"))
                if (
                    as_int(log.get("type")) != 6
                    or as_int(log.get("playerIndex")) != physical_index
                    or as_int(log.get("fromArea")) not in {4, 5}
                    or destination != 3
                    or serial is None
                    or not knockout_is_confirmed(step, log, physical_index, previous_field)
                ):
                    continue
                key = (serial, destination)
                if key in seen:
                    continue
                seen.add(key)
                ready_count = _ready_attacker_count(step, physical_index)
                event = {
                    "knocked_out_card_id": as_int(log.get("cardId")),
                    "ready_attacker_count": ready_count,
                    "zero_ready": ready_count == 0,
                    "classification": _relay_failure_class(step, physical_index)
                    if ready_count == 0
                    else "success",
                    "evidence_strength": "proxy",
                }
                events.append(event)
                source = normalized_evidence(
                    trace,
                    step,
                    fallback_step=index,
                    candidate_physical_index=context.candidate_physical_index,
                )
                source.update(event)
                evidence.append(source)
            previous_field = field_state(step, physical_index)

        zero_ready_count = sum(bool(event["zero_ready"]) for event in events)
        if not evidence:
            steps = trace_steps(trace)
            evidence.append(
                normalized_evidence(
                    trace,
                    steps[-1] if steps else None,
                    fallback_step=len(steps) - 1 if steps else 0,
                    candidate_physical_index=context.candidate_physical_index,
                )
            )
        diagnostic = {
            "game_id": context.game_id,
            "opponent": context.opponent_name,
            "ko_events": len(events),
            "zero_ready_events": zero_ready_count,
            "games_denominator": 1,
            "games_with_zero_ready": int(zero_ready_count > 0),
            "candidate_first": context.candidate_first,
            "turn_order": "first" if context.candidate_first else "second",
            "lifecycle_status": lifecycle,
        }
        failure_counts = {
            name: sum(
                int(event.get("classification") == name)
                for event in events
            )
            for name in RELAY_FAILURES
        }
        return GameMetric(
            metric_id=self.metric_id,
            status=(
                "error"
                if lifecycle == "error"
                else "unavailable"
                if lifecycle == "unfinished"
                else "success"
                if events
                else "unavailable"
            ),
            numerator=zero_ready_count,
            denominator=len(events),
            value=zero_ready_count / len(events) if events else None,
            evidence=tuple(evidence),
            diagnostics=(diagnostic,),
            payload={
                "opportunities": len(events),
                "successes": len(events) - zero_ready_count,
                "failure_counts": failure_counts,
                "events": events,
                "evidence_strength": "proxy",
                "games": 1,
                "error_games": int(lifecycle == "error"),
                "unfinished_games": int(lifecycle == "unfinished"),
                "lifecycle_status": lifecycle,
            },
        )

    def aggregate(self, results: list[GameMetric]) -> AggregateMetric:
        numerator = sum(result.numerator for result in results)
        denominator = sum(result.denominator for result in results)
        grouped: dict[str, dict[str, object]] = defaultdict(
            lambda: {
                "zero_ready_events": 0,
                "ko_events": 0,
                "games": 0,
                "games_with_zero_ready": 0,
            }
        )
        for result in results:
            diagnostic = result.diagnostics[0] if result.diagnostics else {}
            opponent = str(diagnostic.get("opponent", "unknown"))
            group = grouped[opponent]
            group["zero_ready_events"] = int(group["zero_ready_events"]) + result.numerator
            group["ko_events"] = int(group["ko_events"]) + result.denominator
            group["games"] = int(group["games"]) + 1
            group["games_with_zero_ready"] = (
                int(group["games_with_zero_ready"])
                + int(diagnostic.get("games_with_zero_ready", 0))
            )
        for group in grouped.values():
            ko_events = int(group["ko_events"])
            games = int(group["games"])
            group["value"] = (
                int(group["zero_ready_events"]) / ko_events if ko_events else None
            )
            group["games_with_zero_ready_rate"] = (
                int(group["games_with_zero_ready"]) / games if games else None
            )
        payload = _aggregate_payload(results)
        return AggregateMetric(
            metric_id=self.metric_id,
            numerator=numerator,
            denominator=denominator,
            value=numerator / denominator if denominator else None,
            by_opponent=dict(grouped),
            payload=payload,
        )


def _relay_failure_class(step: dict, physical_index: int) -> str:
    player = player_at(step, physical_index)
    field_ids = {
        as_int(card.get("id"))
        for card in [active(player), *bench(player)]
        if isinstance(card, dict) and as_int(card.get("id")) is not None
    }
    discard_ids = {
        as_int(card.get("id"))
        for card in (player.get("discard") or [])
        if isinstance(card, dict) and as_int(card.get("id")) is not None
    }
    hand_ids = {
        as_int(card.get("id"))
        for card in (player.get("hand") or [])
        if isinstance(card, dict) and as_int(card.get("id")) is not None
    }
    if field_ids & {741, 742}:
        return "field_route_miss"
    if discard_ids & RELAY_ROUTE_CARD_IDS:
        return "recoverable_discard_miss"
    if hand_ids & RELAY_ROUTE_CARD_IDS:
        return "recoverable_route_incomplete"
    deck_count = as_int(player.get("deckCount"), 0) or 0
    return "nonterminal_no_field_route" if deck_count > 0 else "terminal_no_resource"


def _aggregate_payload(
    results: list[GameMetric], *, include_turn_order: bool = True
) -> dict[str, object]:
    values = list(results)
    failure_counts = {
        name: sum(
            int(result.payload.get("failure_counts", {}).get(name, 0))
            for result in values
        )
        for name in RELAY_FAILURES
    }
    opportunities = sum(int(result.payload.get("opportunities", 0)) for result in values)
    successes = sum(int(result.payload.get("successes", 0)) for result in values)
    by_turn_order = {
        turn_order: _aggregate_payload(
            [
                result
                for result in values
                if result.diagnostics
                and isinstance(result.diagnostics[0], dict)
                and result.diagnostics[0].get("turn_order") == turn_order
            ],
            include_turn_order=False,
        )
        for turn_order in ("first", "second")
    } if include_turn_order else {}
    return {
        "games": len(values),
        "error_games": sum(
            int(result.payload.get("error_games", result.status == "error"))
            for result in values
        ),
        "unfinished_games": sum(
            int(result.payload.get("unfinished_games", result.status == "unfinished"))
            for result in values
        ),
        "opportunities": opportunities,
        "successes": successes,
        "success_rate": successes / opportunities if opportunities else None,
        "failure_counts": failure_counts,
        "by_turn_order": by_turn_order,
        "evidence_strength": "proxy",
    }
