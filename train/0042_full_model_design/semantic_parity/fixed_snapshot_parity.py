"""Ordered fixed-snapshot feature/model parity comparison."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence
from typing import Any


DISCRETE_STAGES = (
    "observation_schema", "option_skill_relations", "option_effect_relations",
    "event_history", "resource_ledger", "deck_membership", "option_ordering",
    "option_mask", "decision_gate",
)
FLOAT_STAGES = ("observation_tensor", "state_representation", "option_representation")
MODEL_STAGES = ("root_logits", "allocation_logits", "value")


@dataclass(frozen=True, slots=True)
class SnapshotDifference:
    stage: str
    path: str
    left: Any
    right: Any
    absolute_error: float | None = None


@dataclass(frozen=True, slots=True)
class SnapshotParityResult:
    status: str
    first_difference: SnapshotDifference | None
    maximum_float_error: float
    mean_float_error: float
    compared_float_values: int
    greedy_top1_equal: bool | None
    top1_margin_left: float | None
    top1_margin_right: float | None


def _flatten(value: Any, path: str = "") -> list[tuple[str, Any]]:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if isinstance(value, Mapping):
        rows: list[tuple[str, Any]] = []
        for key in sorted(value):
            rows.extend(_flatten(value[key], f"{path}.{key}" if path else str(key)))
        return rows
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = []
        for index, item in enumerate(value):
            rows.extend(_flatten(item, f"{path}[{index}]"))
        return rows
    return [(path, value)]


def _exact_difference(stage: str, left: Any, right: Any) -> SnapshotDifference | None:
    left_rows = _flatten(left)
    right_rows = _flatten(right)
    for index in range(max(len(left_rows), len(right_rows))):
        if index >= len(left_rows):
            return SnapshotDifference(stage, right_rows[index][0], "<missing>", right_rows[index][1])
        if index >= len(right_rows):
            return SnapshotDifference(stage, left_rows[index][0], left_rows[index][1], "<missing>")
        left_path, left_value = left_rows[index]
        right_path, right_value = right_rows[index]
        if left_path != right_path or left_value != right_value:
            return SnapshotDifference(stage, left_path, left_value, right_value)
    return None


def _float_errors(stage: str, left: Any, right: Any, *, atol: float, rtol: float) -> tuple[SnapshotDifference | None, list[float]]:
    left_rows = _flatten(left)
    right_rows = _flatten(right)
    errors: list[float] = []
    if len(left_rows) != len(right_rows):
        return SnapshotDifference(stage, "length", len(left_rows), len(right_rows)), errors
    for (left_path, left_value), (right_path, right_value) in zip(left_rows, right_rows):
        if left_path != right_path:
            return SnapshotDifference(stage, left_path, left_path, right_path), errors
        try:
            error = abs(float(left_value) - float(right_value))
        except (TypeError, ValueError):
            return SnapshotDifference(stage, left_path, left_value, right_value), errors
        errors.append(error)
        if not math.isclose(float(left_value), float(right_value), abs_tol=atol, rel_tol=rtol):
            return SnapshotDifference(stage, left_path, left_value, right_value, error), errors
    return None, errors


def _top1(logits: Any) -> tuple[int, float]:
    rows = [float(value) for _, value in _flatten(logits)]
    if not rows:
        raise ValueError("root_logits must not be empty")
    order = sorted(range(len(rows)), key=lambda index: (-rows[index], index))
    margin = math.inf if len(order) == 1 else rows[order[0]] - rows[order[1]]
    return order[0], margin


def compare_fixed_snapshot(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    feature_atol: float = 1e-6,
    feature_rtol: float = 1e-6,
    model_atol: float = 1e-5,
    model_rtol: float = 1e-5,
) -> SnapshotParityResult:
    float_errors: list[float] = []
    for stage in DISCRETE_STAGES:
        if stage not in left or stage not in right:
            difference = SnapshotDifference(stage, stage, left.get(stage, "<missing>"), right.get(stage, "<missing>"))
            return SnapshotParityResult("FAIL", difference, 0.0, 0.0, 0, None, None, None)
        difference = _exact_difference(stage, left[stage], right[stage])
        if difference:
            return SnapshotParityResult("FAIL", difference, 0.0, 0.0, 0, None, None, None)
    for stage in FLOAT_STAGES + MODEL_STAGES:
        if stage not in left or stage not in right:
            difference = SnapshotDifference(stage, stage, left.get(stage, "<missing>"), right.get(stage, "<missing>"))
            return SnapshotParityResult("FAIL", difference, max(float_errors, default=0.0), sum(float_errors) / len(float_errors) if float_errors else 0.0, len(float_errors), None, None, None)
        tolerance = (feature_atol, feature_rtol) if stage in FLOAT_STAGES else (model_atol, model_rtol)
        difference, errors = _float_errors(stage, left[stage], right[stage], atol=tolerance[0], rtol=tolerance[1])
        float_errors.extend(errors)
        if difference:
            return SnapshotParityResult("FAIL", difference, max(float_errors, default=0.0), sum(float_errors) / len(float_errors) if float_errors else 0.0, len(float_errors), None, None, None)
    left_top1, left_margin = _top1(left["root_logits"])
    right_top1, right_margin = _top1(right["root_logits"])
    return SnapshotParityResult(
        "PASS" if left_top1 == right_top1 else "FAIL",
        None if left_top1 == right_top1 else SnapshotDifference("greedy_top1", "root_logits", left_top1, right_top1),
        max(float_errors, default=0.0),
        sum(float_errors) / len(float_errors) if float_errors else 0.0,
        len(float_errors), left_top1 == right_top1, left_margin, right_margin,
    )


__all__ = ["SnapshotDifference", "SnapshotParityResult", "compare_fixed_snapshot"]
