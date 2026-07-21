from __future__ import annotations

from .base import AggregateMetric, GameContext, GameMetric, MetricPlugin
from .correctness import CorrectnessPlugin
from .length import LengthPlugin
from .outcome import OutcomePlugin


def default_metric_plugins() -> tuple[MetricPlugin, ...]:
    return (
        OutcomePlugin(),
        LengthPlugin(),
        CorrectnessPlugin(),
    )


__all__ = [
    "AggregateMetric",
    "CorrectnessPlugin",
    "GameContext",
    "GameMetric",
    "LengthPlugin",
    "MetricPlugin",
    "OutcomePlugin",
    "default_metric_plugins",
]
