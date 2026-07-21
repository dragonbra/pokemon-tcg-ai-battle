from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class ReportData:
    """同一轮评测已经完成聚合后交给报告层的只读数据。"""

    manifest: dict[str, object]
    summary: dict[str, object]
    games: tuple
    metrics: dict[str, object]
    cases: tuple


def as_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def ordered_mapping(value: object) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(as_mapping(value).items()))


def display_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if not isfinite(value):
            return "-"
        return f"{value:.6g}"
    return str(value)


def percentage(value: object) -> str:
    if isinstance(value, bool):
        return "-"
    if isinstance(value, int | float) and isfinite(value):
        return f"{value * 100:.2f}%"
    return display_value(value)


def failure_class_distribution(metrics: Mapping[str, object]) -> Mapping[str, object]:
    correctness = as_mapping(metrics.get("correctness"))
    diagnostics = as_mapping(correctness.get("diagnostics"))
    return as_mapping(
        diagnostics.get("failure_classes", diagnostics.get("failure_class_distribution"))
    )


def control_differences(summary: Mapping[str, object]) -> Mapping[str, object]:
    control = as_mapping(summary.get("control"))
    return as_mapping(control.get("differences"))


def case_evidence_steps(case: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    evidence = case.get("evidence", case.get("evidence_steps", ()))
    if not isinstance(evidence, tuple | list):
        return ()
    return tuple(as_mapping(item) for item in evidence)


def json_ready(value: Any) -> object:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    return value
