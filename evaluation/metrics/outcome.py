from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .base import AggregateMetric, GameContext, GameMetric
from .trace_utils import normalized_evidence, result_field, trace_steps


def _error_text(trace: dict) -> str:
    values = [
        result_field(trace, "error"),
        result_field(trace, "reason"),
        result_field(trace, "exception"),
    ]
    return " ".join(str(value) for value in values if value).lower()


def _category(trace: dict, context: GameContext) -> str:
    for value in (result_field(trace, "error_kind"), result_field(trace, "status")):
        normalized = str(value).lower() if value else ""
        if normalized in {"step_limit", "step limit", "unfinished"}:
            return "unfinished"
        if normalized and normalized not in {"finished", "success"}:
            return "error"

    error = _error_text(trace)
    if any(token in error for token in ("step_limit", "step limit", "unfinished")):
        return "unfinished"
    if error:
        return "error"
    if result_field(trace, "finished") is False or (
        result_field(trace, "winner") is None and result_field(trace, "finished") is not True
    ):
        return "unfinished"
    winner = result_field(trace, "winner")
    if isinstance(winner, str):
        normalized = winner.lower()
        if normalized in {"candidate", context.candidate_name.lower(), "win", "won"}:
            return "win"
        if normalized in {"opponent", context.opponent_name.lower(), "loss", "lost"}:
            return "loss"
        if normalized in {"draw", "tie"}:
            return "draw"
    # The worker normalizes numeric winners to candidate-relative values.
    if winner == 0:
        return "win"
    if winner == 1:
        return "loss"
    return "draw"


class OutcomePlugin:
    metric_id = "outcome"

    def analyze_game(self, trace: dict, context: GameContext) -> GameMetric:
        category = _category(trace, context)
        evidence = _evidence(trace, category, context.candidate_physical_index)
        return GameMetric(
            metric_id=self.metric_id,
            status="error" if category == "error" else "success",
            numerator=int(category == "win"),
            denominator=1,
            value=category,
            evidence=evidence,
            diagnostics=(
                {
                    "game_id": context.game_id,
                    "opponent": context.opponent_name,
                    "category": category,
                },
            ),
        )

    def aggregate(self, results: list[GameMetric]) -> AggregateMetric:
        denominator = sum(result.denominator for result in results)
        numerator = sum(result.numerator for result in results)
        by_opponent = _by_opponent(results)
        return AggregateMetric(
            metric_id=self.metric_id,
            numerator=numerator,
            denominator=denominator,
            value=(numerator / denominator if denominator else None),
            by_opponent=by_opponent,
        )


def _evidence(trace: dict, category: str, candidate_physical_index: int) -> tuple:
    steps = trace_steps(trace)
    evidence = normalized_evidence(
        trace,
        steps[-1] if steps else None,
        fallback_step=len(steps) - 1 if steps else 0,
        candidate_physical_index=candidate_physical_index,
    )
    evidence["category"] = category
    return (evidence,)


def _by_opponent(results: Iterable[GameMetric]) -> dict[str, dict[str, object]]:
    grouped: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "errors": 0,
            "unfinished": 0,
            "attempts": 0,
        }
    )
    for result in results:
        opponent = "unknown"
        if result.diagnostics and isinstance(result.diagnostics[0], dict):
            opponent = str(result.diagnostics[0].get("opponent", "unknown"))
        category = str(result.value)
        group = grouped[opponent]
        group["attempts"] = int(group["attempts"]) + result.denominator
        category_key = "errors" if category == "error" else category
        if category_key in group:
            group[category_key] = int(group[category_key]) + 1
    return dict(grouped)
