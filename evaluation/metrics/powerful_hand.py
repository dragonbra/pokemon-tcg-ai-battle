from __future__ import annotations

from collections import defaultdict

from .base import AggregateMetric, GameContext, GameMetric
from .trace_utils import (
    agent_turn_target,
    as_int,
    candidate_index,
    current,
    normalized_evidence,
    observation,
    option_attack_id,
    option_list,
    selected_attack_id,
    trace_steps,
)


POWERFUL_HAND_ATTACK_ID = 1072


def _is_candidate_step(step: dict, physical_index: int) -> bool:
    observed_current = observation(step).get("current")
    state = step.get("state")
    if isinstance(state, dict) and isinstance(state.get("current"), dict):
        state = state["current"]
    for source in (observed_current, state, step):
        if not isinstance(source, dict) or "yourIndex" not in source:
            continue
        your_index = as_int(source.get("yourIndex"))
        if your_index in (0, 1):
            return your_index == physical_index
    return step.get("role") not in {"opponent", "finished"}


def second_own_turn_entries(
    trace: dict,
    context: GameContext,
) -> list[tuple[int, dict]]:
    """返回 candidate 第二个己方回合中的决策 step。"""
    physical_index = candidate_index(trace, context.candidate_physical_index)
    target_turn = agent_turn_target(trace, physical_index)
    if target_turn is None:
        target_turn = 3 if context.candidate_first else 4
    return [
        (index, step)
        for index, step in enumerate(trace_steps(trace))
        if _is_candidate_step(step, physical_index)
        and as_int(current(step).get("turn")) == target_turn
    ]


def _evidence(
    trace: dict,
    context: GameContext,
    index: int,
    step: dict | None,
    **facts: object,
) -> tuple:
    value = normalized_evidence(
        trace,
        step,
        fallback_step=index,
        candidate_physical_index=context.candidate_physical_index,
    )
    value.update(facts)
    return (value,)


class PowerfulHandPlugin:
    metric_id = "powerful_hand"

    def analyze_game(self, trace: dict, context: GameContext) -> GameMetric:
        entries = second_own_turn_entries(trace, context)
        selected = next(
            (
                (index, step)
                for index, step in entries
                if selected_attack_id(step) == POWERFUL_HAND_ATTACK_ID
            ),
            None,
        )
        available = next(
            (
                (index, step)
                for index, step in entries
                if any(
                    option_attack_id(option) == POWERFUL_HAND_ATTACK_ID
                    for option in option_list(step)
                )
            ),
            None,
        )

        if selected is not None:
            value = "selected"
            status = "success"
            reason = "selected"
            source = selected
        elif available is not None:
            value = "not_selected"
            status = "success"
            reason = "powerful_hand_not_selected"
            source = available
        elif entries:
            value = "unavailable"
            status = "unavailable"
            reason = "powerful_hand_unavailable"
            source = entries[-1]
        else:
            value = "unavailable"
            status = "unavailable"
            reason = "second_turn_not_reached"
            steps = trace_steps(trace)
            source = (len(steps) - 1, steps[-1]) if steps else (0, None)

        diagnostic = {
            "game_id": context.game_id,
            "opponent": context.opponent_name,
            "all_games_denominator": 1,
            "reached_second_turn_denominator": int(bool(entries)),
            "reason": reason,
            "powerful_hand_available": available is not None,
        }
        return GameMetric(
            metric_id=self.metric_id,
            status=status,
            numerator=int(selected is not None),
            denominator=1,
            value=value,
            evidence=_evidence(
                trace,
                context,
                source[0],
                source[1],
                reason=reason,
            ),
            diagnostics=(diagnostic,),
        )

    def aggregate(self, results: list[GameMetric]) -> AggregateMetric:
        numerator = sum(result.numerator for result in results)
        denominator = sum(result.denominator for result in results)
        grouped: dict[str, dict[str, object]] = defaultdict(
            lambda: {
                "numerator": 0,
                "all_games_denominator": 0,
                "reached_second_turn_denominator": 0,
                "reason_counts": {},
            }
        )
        for result in results:
            diagnostic = result.diagnostics[0] if result.diagnostics else {}
            opponent = str(diagnostic.get("opponent", "unknown"))
            group = grouped[opponent]
            group["numerator"] = int(group["numerator"]) + result.numerator
            group["all_games_denominator"] = (
                int(group["all_games_denominator"]) + result.denominator
            )
            group["reached_second_turn_denominator"] = (
                int(group["reached_second_turn_denominator"])
                + int(diagnostic.get("reached_second_turn_denominator", 0))
            )
            reasons = group["reason_counts"]
            if isinstance(reasons, dict):
                reason = str(diagnostic.get("reason", "unknown"))
                reasons[reason] = int(reasons.get(reason, 0)) + 1
        for group in grouped.values():
            all_games = int(group["all_games_denominator"])
            group["value"] = int(group["numerator"]) / all_games if all_games else None
        return AggregateMetric(
            metric_id=self.metric_id,
            numerator=numerator,
            denominator=denominator,
            value=numerator / denominator if denominator else None,
            by_opponent=dict(grouped),
        )
