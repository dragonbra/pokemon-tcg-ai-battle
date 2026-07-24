from __future__ import annotations

from collections import defaultdict

from .base import AggregateMetric, GameContext, GameMetric
from .trace_utils import (
    as_int,
    current,
    lifecycle_status,
    normalized_evidence,
    result_field,
    result_steps,
    trace_steps,
)


class LengthPlugin:
    metric_id = "length"

    def analyze_game(self, trace: dict, context: GameContext) -> GameMetric:
        steps = trace_steps(trace)
        engine_turn = _turn_count(steps)
        round_count = _round_count(engine_turn)
        selection_count = result_steps(trace)
        lifecycle = lifecycle_status(trace)
        if lifecycle != "finished":
            return GameMetric(
                self.metric_id,
                "error" if lifecycle == "error" else "unavailable",
                0,
                0,
                None,
                _evidence(trace, steps, selection_count, context.candidate_physical_index),
                (
                    {
                        "game_id": context.game_id,
                        "opponent": context.opponent_name,
                        "lifecycle_status": lifecycle,
                    },
                ),
                {
                    "observed_engine_turn": engine_turn,
                    "observed_round": round_count,
                    "action_selections": selection_count,
                    "lifecycle_status": lifecycle,
                },
            )
        if engine_turn is None or round_count is None:
            return GameMetric(
                self.metric_id,
                "unavailable",
                0,
                0,
                None,
                _unavailable_evidence(),
                ({"game_id": context.game_id, "opponent": context.opponent_name},),
            )
        return GameMetric(
            self.metric_id,
            "success",
            round_count,
            1,
            round_count,
            _evidence(trace, steps, selection_count, context.candidate_physical_index),
            (
                {
                    "game_id": context.game_id,
                    "opponent": context.opponent_name,
                    "lifecycle_status": lifecycle,
                },
            ),
            {
                "round": round_count,
                "engine_turn": engine_turn,
                "ending_phase": _ending_phase(engine_turn),
                "candidate_turn_order": "first" if context.candidate_first else "second",
                "outcome": _outcome(trace, context),
                "action_selections": selection_count,
                "lifecycle_status": lifecycle,
            },
        )

    def aggregate(self, results: list[GameMetric]) -> AggregateMetric:
        numerator = sum(result.numerator for result in results)
        denominator = sum(result.denominator for result in results)
        grouped: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        round_grouped: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        engine_turn_grouped: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        selection_grouped: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        round_numerator = 0
        round_denominator = 0
        engine_turn_numerator = 0
        engine_turn_denominator = 0
        for result in results:
            opponent = str(result.diagnostics[0].get("opponent", "unknown"))
            grouped[opponent][0] += result.numerator
            grouped[opponent][1] += result.denominator
            rounds = as_int(result.payload.get("round"))
            if rounds is not None:
                round_numerator += rounds
                round_denominator += 1
                round_grouped[opponent][0] += rounds
                round_grouped[opponent][1] += 1
            engine_turn = as_int(result.payload.get("engine_turn"))
            if engine_turn is not None:
                engine_turn_numerator += engine_turn
                engine_turn_denominator += 1
                engine_turn_grouped[opponent][0] += engine_turn
                engine_turn_grouped[opponent][1] += 1
            selections = as_int(result.payload.get("action_selections"))
            if selections is not None:
                selection_grouped[opponent][0] += selections
                selection_grouped[opponent][1] += 1
        by_opponent = {
            opponent: {
                "numerator": values[0],
                "denominator": values[1],
                "value": values[0] / values[1] if values[1] else None,
            }
            for opponent, values in grouped.items()
        }
        round_summary = _metric_summary(
            round_numerator,
            round_denominator,
            round_grouped,
        )
        return AggregateMetric(
            self.metric_id,
            numerator,
            denominator,
            numerator / denominator if denominator else None,
            by_opponent,
            {
                "rounds": round_summary,
                "turns": round_summary,
                "engine_turns": _metric_summary(
                    engine_turn_numerator,
                    engine_turn_denominator,
                    engine_turn_grouped,
                ),
                "by_outcome": {
                    outcome: _outcome_summary(results, outcome)
                    for outcome in ("win", "loss")
                },
                "action_selections": _group_summary(selection_grouped),
            },
        )


def _turn_count(steps: list[dict]) -> int | None:
    turns = [as_int(current(step).get("turn")) for step in steps]
    valid_turns = [turn for turn in turns if turn is not None and turn >= 0]
    return max(valid_turns) if valid_turns else None


def _round_count(engine_turn: int | None) -> int | None:
    if engine_turn is None:
        return None
    return (engine_turn + 1) // 2


def _ending_phase(engine_turn: int) -> str:
    return "first_player" if engine_turn % 2 == 1 else "second_player"


def _outcome(trace: dict, context: GameContext) -> str:
    winner = result_field(trace, "winner")
    if isinstance(winner, str):
        normalized = winner.lower()
        if normalized in {"candidate", context.candidate_name.lower(), "win", "won"}:
            return "win"
        if normalized in {"opponent", context.opponent_name.lower(), "loss", "lost"}:
            return "loss"
        return "draw"
    if winner == 0:
        return "win"
    if winner == 1:
        return "loss"
    return "draw"


def _evidence(
    trace: dict,
    steps: list[dict],
    selection_count: int | None,
    candidate_physical_index: int,
) -> tuple:
    evidence = normalized_evidence(
        trace,
        steps[-1] if steps else None,
        fallback_step=len(steps) - 1 if steps else 0,
        candidate_physical_index=candidate_physical_index,
    )
    evidence["action_selections"] = selection_count
    return (evidence,)


def _group_summary(grouped: dict[str, list[int]]) -> dict[str, object]:
    numerator = sum(values[0] for values in grouped.values())
    denominator = sum(values[1] for values in grouped.values())
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
        "by_opponent": {
            opponent: {
                "numerator": values[0],
                "denominator": values[1],
                "value": values[0] / values[1] if values[1] else None,
            }
            for opponent, values in grouped.items()
        },
    }


def _metric_summary(
    numerator: int,
    denominator: int,
    grouped: dict[str, list[int]],
) -> dict[str, object]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
        "by_opponent": {
            opponent: {
                "numerator": values[0],
                "denominator": values[1],
                "value": values[0] / values[1] if values[1] else None,
            }
            for opponent, values in grouped.items()
        },
    }


def _outcome_summary(results: list[GameMetric], outcome: str) -> dict[str, object]:
    values = [
        result
        for result in results
        if result.payload.get("outcome") == outcome
        and as_int(result.payload.get("round")) is not None
    ]
    rounds = [as_int(result.payload.get("round"), 0) or 0 for result in values]
    distribution: dict[str, dict[str, int]] = {}
    for result, round_count in zip(values, rounds):
        key = str(round_count)
        bucket = distribution.setdefault(
            key,
            {"count": 0, "candidate_first": 0, "candidate_second": 0},
        )
        bucket["count"] += 1
        turn_order = str(result.payload.get("candidate_turn_order", ""))
        if turn_order in {"first", "second"}:
            bucket[f"candidate_{turn_order}"] += 1
    numerator = sum(rounds)
    denominator = len(rounds)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "average": numerator / denominator if denominator else None,
        "distribution": distribution,
    }


def _unavailable_evidence() -> tuple:
    return ({"step": 0, "turn": 0, "role": "trace", "action": []},)
