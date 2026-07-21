from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class GameContext:
    game_id: str
    candidate_name: str
    opponent_name: str
    candidate_physical_index: int
    candidate_first: bool


@dataclass(frozen=True)
class GameMetric:
    metric_id: str
    status: str
    numerator: int
    denominator: int
    value: float | int | str | None
    evidence: tuple
    diagnostics: tuple
    payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AggregateMetric:
    metric_id: str
    numerator: int
    denominator: int
    value: float | int | str | None
    by_opponent: dict[str, dict[str, object]]
    payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricPresentation:
    metric_id: str
    title: str
    markdown: str
    html: str


@runtime_checkable
class MetricPlugin(Protocol):
    metric_id: str

    def analyze_game(self, trace: dict, context: GameContext) -> GameMetric:
        raise NotImplementedError

    def aggregate(self, results: list[GameMetric]) -> AggregateMetric:
        raise NotImplementedError
