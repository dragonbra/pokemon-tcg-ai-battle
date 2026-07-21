from __future__ import annotations

from collections import defaultdict

from .base import AggregateMetric, GameContext, GameMetric
from .trace_utils import normalized_evidence, result_steps, trace_steps


class LengthPlugin:
    metric_id = "length"

    def analyze_game(self, trace: dict, context: GameContext) -> GameMetric:
        steps = trace_steps(trace)
        count = result_steps(trace)
        if count is None and not steps:
            return GameMetric(
                self.metric_id,
                "unavailable",
                0,
                0,
                None,
                _unavailable_evidence(),
                ({"game_id": context.game_id, "opponent": context.opponent_name},),
            )
        if count is None:
            count = len(steps)
        return GameMetric(
            self.metric_id,
            "success",
            count,
            1,
            count,
            _evidence(trace, steps, count, context.candidate_physical_index),
            ({"game_id": context.game_id, "opponent": context.opponent_name},),
        )

    def aggregate(self, results: list[GameMetric]) -> AggregateMetric:
        numerator = sum(result.numerator for result in results)
        denominator = sum(result.denominator for result in results)
        grouped: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for result in results:
            opponent = str(result.diagnostics[0].get("opponent", "unknown"))
            grouped[opponent][0] += result.numerator
            grouped[opponent][1] += result.denominator
        by_opponent = {
            opponent: {
                "numerator": values[0],
                "denominator": values[1],
                "value": values[0] / values[1] if values[1] else None,
            }
            for opponent, values in grouped.items()
        }
        return AggregateMetric(
            self.metric_id,
            numerator,
            denominator,
            numerator / denominator if denominator else None,
            by_opponent,
        )


def _evidence(
    trace: dict,
    steps: list[dict],
    count: int,
    candidate_physical_index: int,
) -> tuple:
    evidence = normalized_evidence(
        trace,
        steps[-1] if steps else None,
        fallback_step=len(steps) - 1 if steps else 0,
        candidate_physical_index=candidate_physical_index,
    )
    evidence["steps"] = count
    return (evidence,)


def _unavailable_evidence() -> tuple:
    return ({"step": 0, "turn": 0, "role": "trace", "action": []},)
