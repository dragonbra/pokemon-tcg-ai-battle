from __future__ import annotations

from .base import AggregateMetric, GameContext, GameMetric, MetricPlugin, MetricPresentation
from .correctness import CorrectnessPlugin
from .length import LengthPlugin
from .outcome import OutcomePlugin
from .profiles import (
    AUTO_ITERATION_PROFILE_ID,
    CORE_PROFILE_ID,
    MetricPriority,
    MetricProfile,
    available_metric_profiles,
    get_metric_profile,
)


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
    "MetricPresentation",
    "MetricPriority",
    "MetricProfile",
    "OutcomePlugin",
    "AUTO_ITERATION_PROFILE_ID",
    "CORE_PROFILE_ID",
    "available_metric_profiles",
    "default_metric_plugins",
    "get_metric_profile",
]
