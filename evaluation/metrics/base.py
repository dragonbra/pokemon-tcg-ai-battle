from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class AggregateMetric:
    metric_id: str
    numerator: int
    denominator: int
    value: float | int | str | None
    by_opponent: dict[str, dict[str, object]]


@runtime_checkable
class MetricPlugin(Protocol):
    metric_id: str

    def analyze_game(self, trace: dict, context: GameContext) -> GameMetric:
        raise NotImplementedError

    def aggregate(self, results: list[GameMetric]) -> AggregateMetric:
        raise NotImplementedError
