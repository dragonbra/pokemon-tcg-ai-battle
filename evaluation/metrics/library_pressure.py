from __future__ import annotations

from collections import defaultdict

from .base import AggregateMetric, GameContext, GameMetric
from .powerful_hand import _is_candidate_step
from .trace_utils import (
    as_int,
    candidate_index,
    current,
    logs,
    normalized_evidence,
    player_at,
    result_field,
    trace_steps,
)


WARNING_DECK_COUNT = 15
CRITICAL_DECK_COUNT = 10
RESULT_LOG_TYPE = 23
DECK_OUT_REASON = 2


def _deck_count(step: dict, physical_index: int) -> int | None:
    return as_int(player_at(step, physical_index).get("deckCount"))


def _next_deck_count(steps: list[dict], index: int, physical_index: int) -> int | None:
    for future in steps[index + 1 :]:
        value = _deck_count(future, physical_index)
        if value is not None:
            return value
    return None


def _candidate_deck_out(steps: list[dict], physical_index: int) -> bool:
    for step in steps:
        for log in logs(step):
            if (
                as_int(log.get("type")) == RESULT_LOG_TYPE
                and as_int(log.get("reason")) == DECK_OUT_REASON
            ):
                winner = as_int(log.get("result"))
                if winner in (0, 1) and winner != physical_index:
                    return True
    return False


def _end_turn_deck_counts(steps: list[dict], physical_index: int) -> list[dict[str, int]]:
    candidate_turns: list[int] = []
    for step in steps:
        turn = as_int(current(step).get("turn"))
        if (
            turn is not None
            and _is_candidate_step(step, physical_index)
            and turn not in candidate_turns
        ):
            candidate_turns.append(turn)

    result: list[dict[str, int]] = []
    for turn in candidate_turns:
        turn_indices = [
            index for index, step in enumerate(steps) if as_int(current(step).get("turn")) == turn
        ]
        last_index = turn_indices[-1]
        source_index = next(
            (
                index
                for index in range(last_index + 1, len(steps))
                if as_int(current(steps[index]).get("turn")) != turn
            ),
            last_index,
        )
        deck_count = _deck_count(steps[source_index], physical_index)
        if deck_count is not None:
            result.append({"turn": turn, "deck_count": deck_count})
    return result


class LibraryPressurePlugin:
    metric_id = "library_pressure"

    def analyze_game(self, trace: dict, context: GameContext) -> GameMetric:
        steps = trace_steps(trace)
        physical_index = candidate_index(trace, context.candidate_physical_index)
        warning_consumption = 0
        critical_consumption = 0
        evidence: list[dict[str, object]] = []

        for index, step in enumerate(steps):
            if not _is_candidate_step(step, physical_index) or not step.get("action"):
                continue
            before = _deck_count(step, physical_index)
            after = _next_deck_count(steps, index, physical_index)
            if before is None or after is None or before > WARNING_DECK_COUNT or after >= before:
                continue
            consumed = before - after
            warning_consumption += consumed
            if before <= CRITICAL_DECK_COUNT:
                critical_consumption += consumed
            source = normalized_evidence(
                trace,
                step,
                fallback_step=index,
                candidate_physical_index=context.candidate_physical_index,
            )
            source.update(
                {
                    "deck_count_before": before,
                    "deck_count_after": after,
                    "cards_consumed": consumed,
                    "threshold": (
                        CRITICAL_DECK_COUNT if before <= CRITICAL_DECK_COUNT else WARNING_DECK_COUNT
                    ),
                }
            )
            evidence.append(source)

        result_reason = as_int(result_field(trace, "reason"))
        normalized_winner = as_int(result_field(trace, "winner"))
        deck_out = _candidate_deck_out(steps, physical_index) or (
            result_reason == DECK_OUT_REASON and normalized_winner == 1
        )
        status = str(result_field(trace, "status", "")).lower()
        unfinished = result_field(trace, "finished") is False or status in {
            "unfinished",
            "step_limit",
        }
        if not evidence:
            source_step = next(
                (
                    (index, step)
                    for index, step in enumerate(steps)
                    if any(as_int(log.get("reason")) == DECK_OUT_REASON for log in logs(step))
                ),
                (len(steps) - 1, steps[-1]) if steps else (0, None),
            )
            evidence.append(
                normalized_evidence(
                    trace,
                    source_step[1],
                    fallback_step=source_step[0],
                    candidate_physical_index=context.candidate_physical_index,
                )
            )
        diagnostic = {
            "game_id": context.game_id,
            "opponent": context.opponent_name,
            "at_or_below_15_consumption": warning_consumption,
            "at_or_below_10_consumption": critical_consumption,
            "end_turn_deck_counts": _end_turn_deck_counts(steps, physical_index),
            "deck_out": deck_out,
            "unfinished": unfinished,
        }
        return GameMetric(
            metric_id=self.metric_id,
            status="success",
            numerator=warning_consumption,
            denominator=1,
            value=warning_consumption,
            evidence=tuple(evidence),
            diagnostics=(diagnostic,),
        )

    def aggregate(self, results: list[GameMetric]) -> AggregateMetric:
        numerator = sum(result.numerator for result in results)
        denominator = sum(result.denominator for result in results)
        grouped: dict[str, dict[str, object]] = defaultdict(
            lambda: {
                "at_or_below_15_consumption": 0,
                "at_or_below_10_consumption": 0,
                "games": 0,
                "deck_out_games": 0,
                "unfinished_games": 0,
            }
        )
        for result in results:
            diagnostic = result.diagnostics[0] if result.diagnostics else {}
            opponent = str(diagnostic.get("opponent", "unknown"))
            group = grouped[opponent]
            group["at_or_below_15_consumption"] = (
                int(group["at_or_below_15_consumption"]) + result.numerator
            )
            group["at_or_below_10_consumption"] = (
                int(group["at_or_below_10_consumption"])
                + int(diagnostic.get("at_or_below_10_consumption", 0))
            )
            group["games"] = int(group["games"]) + result.denominator
            group["deck_out_games"] = int(group["deck_out_games"]) + int(
                bool(diagnostic.get("deck_out"))
            )
            group["unfinished_games"] = int(group["unfinished_games"]) + int(
                bool(diagnostic.get("unfinished"))
            )
        for group in grouped.values():
            games = int(group["games"])
            group["value"] = (
                int(group["at_or_below_15_consumption"]) / games if games else None
            )
        return AggregateMetric(
            metric_id=self.metric_id,
            numerator=numerator,
            denominator=denominator,
            value=numerator / denominator if denominator else None,
            by_opponent=dict(grouped),
        )
