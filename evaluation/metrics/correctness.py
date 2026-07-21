from __future__ import annotations

from collections import defaultdict

from .base import AggregateMetric, GameContext, GameMetric
from .trace_utils import normalized_evidence, result_field, trace_steps


ERROR_CLASSES = (
    "candidate_error",
    "illegal_action",
    "engine_error",
    "worker_crash",
    "unfinished",
    "visualization_error",
)


def _error_text(trace: dict) -> str:
    values = [
        result_field(trace, "error"),
        result_field(trace, "reason"),
        result_field(trace, "exception"),
    ]
    return " ".join(str(value) for value in values if value).lower()


def _explicit_classification(value: object) -> str | None:
    normalized = str(value).lower() if value else ""
    if normalized in {"candidate_error", "opponent_error", "illegal_action", "unfinished"}:
        return normalized
    if normalized in {"step_limit", "step limit"}:
        return "unfinished"
    if normalized in {"worker_error", "worker_crash"}:
        return "worker_crash"
    if normalized in {"engine_error", "game_error", "start_error", "finish_error", "cg_mismatch"}:
        return "engine_error"
    if normalized == "load_error":
        return "worker_crash"
    if normalized == "visualization_error":
        return "visualization_error"
    return None


def _classify(trace: dict) -> str | None:
    for value in (result_field(trace, "error_kind"), result_field(trace, "status")):
        category = _explicit_classification(value)
        if category is not None:
            return category
    if trace.get("visualization_error"):
        return "visualization_error"

    text = _error_text(trace)
    if "visual" in text:
        return "visualization_error"
    if "illegal" in text:
        return "illegal_action"
    if "worker" in text or "crash" in text:
        return "worker_crash"
    if "engine" in text or "native" in text:
        return "engine_error"
    if "step" in text or "unfinished" in text or result_field(trace, "finished") is False:
        return "unfinished"
    if "candidate" in text or "agent" in text:
        return "candidate_error"
    if "opponent" in text:
        return "opponent_error"
    if text:
        return "engine_error"
    if result_field(trace, "finished") is False:
        return "unfinished"
    return None


def _evidence(trace: dict, category: str | None, candidate_physical_index: int = 0) -> tuple:
    steps = trace_steps(trace)
    evidence = normalized_evidence(
        trace,
        steps[-1] if steps else None,
        fallback_step=len(steps) - 1 if steps else 0,
        candidate_physical_index=candidate_physical_index,
    )
    evidence["category"] = category or "ok"
    return (evidence,)


class CorrectnessPlugin:
    metric_id = "correctness"

    def analyze_game(self, trace: dict, context: GameContext) -> GameMetric:
        category = _classify(trace)
        if not trace and category is None:
            return GameMetric(
                self.metric_id,
                "unavailable",
                0,
                0,
                None,
                _evidence(trace, category, context.candidate_physical_index),
                ({"game_id": context.game_id, "opponent": context.opponent_name},),
            )
        status = "error" if category in ERROR_CLASSES else "success"
        if category == "opponent_error":
            status = "success"
        return GameMetric(
            self.metric_id,
            status,
            int(category is not None and category != "opponent_error"),
            1,
            category or "ok",
            _evidence(trace, category, context.candidate_physical_index),
            ({"game_id": context.game_id, "opponent": context.opponent_name},),
        )

    def aggregate(self, results: list[GameMetric]) -> AggregateMetric:
        numerator = sum(result.numerator for result in results)
        denominator = sum(result.denominator for result in results)
        grouped: dict[str, dict[str, object]] = defaultdict(
            lambda: {"errors": 0, "attempts": 0, "by_class": {}}
        )
        for result in results:
            opponent = str(result.diagnostics[0].get("opponent", "unknown"))
            group = grouped[opponent]
            group["errors"] = int(group["errors"]) + result.numerator
            group["attempts"] = int(group["attempts"]) + result.denominator
            category = str(result.value)
            classes = group["by_class"]
            if isinstance(classes, dict):
                classes[category] = int(classes.get(category, 0)) + 1
        return AggregateMetric(
            self.metric_id,
            numerator,
            denominator,
            numerator / denominator if denominator else None,
            dict(grouped),
        )
