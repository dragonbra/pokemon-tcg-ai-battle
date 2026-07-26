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
    terminal_forfeit = str(result_field(trace, "error_kind") or result_field(trace, "status") or "").lower()
    if result_field(trace, "finished") is True and terminal_forfeit in {
        "candidate_error",
        "opponent_error",
    }:
        winner = result_field(trace, "winner")
        if winner == 0:
            return "win"
        if winner == 1:
            return "loss"
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
                    "candidate_first": context.candidate_first,
                    "turn_order": "first" if context.candidate_first else "second",
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
            payload={
                "all_games": _turn_order_group(results),
                "by_turn_order": {
                    turn_order: _turn_order_group(
                        result
                        for result in results
                        if result.diagnostics
                        and isinstance(result.diagnostics[0], dict)
                        and result.diagnostics[0].get("turn_order") == turn_order
                    )
                    for turn_order in ("first", "second")
                },
            },
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


def _turn_order_group(results: Iterable[GameMetric]) -> dict[str, object]:
    values = list(results)
    games = len(values)
    wins = sum(result.value == "win" for result in values)
    losses = sum(result.value == "loss" for result in values)
    draws = sum(result.value == "draw" for result in values)
    errors = sum(result.value == "error" for result in values)
    unfinished = sum(result.value == "unfinished" for result in values)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "errors": errors,
        "unfinished": unfinished,
        "numerator": wins,
        "denominator": games,
        "value": wins / games if games else None,
    }
