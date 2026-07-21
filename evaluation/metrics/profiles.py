from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib

from .base import MetricPlugin
from .correctness import CorrectnessPlugin
from .length import LengthPlugin
from .library_pressure import LibraryPressurePlugin
from .outcome import OutcomePlugin
from .powerful_hand import PowerfulHandPlugin
from .post_ko_relay import PostKORelayPlugin
from .rare_candy import RareCandyPlugin
from .run_away_draw import RunAwayDrawPlugin


CORE_PROFILE_ID = "core"
AUTO_ITERATION_PROFILE_ID = "auto_iteration_v8_setup_relay"

CORE_METRIC_IDS = (
    "outcome",
    "length",
    "correctness",
    "powerful_hand",
    "rare_candy",
    "post_ko_relay",
    "run_away_draw",
    "library_pressure",
)
AUTO_ITERATION_METRIC_IDS = (*CORE_METRIC_IDS, "setup_relay", "attack_quality")


@dataclass(frozen=True)
class MetricPriority:
    metric_id: str
    priority: str
    direction: str


@dataclass(frozen=True)
class MetricProfile:
    profile_id: str
    revision: int
    description: str
    priorities: tuple[MetricPriority, ...]
    metric_ids: tuple[str, ...]
    plugin_factories: tuple[Callable[[], MetricPlugin], ...]

    def manifest(self) -> dict[str, object]:
        return {
            "id": self.profile_id,
            "revision": self.revision,
            "description": self.description,
            "metric_ids": list(self.metric_ids),
            "priorities": [
                {
                    "metric_id": priority.metric_id,
                    "priority": priority.priority,
                    "direction": priority.direction,
                }
                for priority in self.priorities
            ],
        }


def _lazy_factory(module_name: str, class_name: str) -> Callable[[], MetricPlugin]:
    def factory() -> MetricPlugin:
        module = importlib.import_module(module_name)
        plugin_class = getattr(module, class_name)
        return plugin_class()

    return factory


def _core_factories() -> tuple[Callable[[], MetricPlugin], ...]:
    return (
        OutcomePlugin,
        LengthPlugin,
        CorrectnessPlugin,
        PowerfulHandPlugin,
        RareCandyPlugin,
        PostKORelayPlugin,
        RunAwayDrawPlugin,
        LibraryPressurePlugin,
    )


CORE_PROFILE = MetricProfile(
    profile_id=CORE_PROFILE_ID,
    revision=1,
    description="Evaluation framework core metrics",
    priorities=(),
    metric_ids=CORE_METRIC_IDS,
    plugin_factories=_core_factories(),
)

AUTO_ITERATION_PROFILE = MetricProfile(
    profile_id=AUTO_ITERATION_PROFILE_ID,
    revision=2,
    description="V8 Setup and Relay AutoIteration metric profile",
    priorities=(
        MetricPriority("outcome", "result_guardrail", "higher"),
        MetricPriority("powerful_hand", "target", "higher"),
        MetricPriority("post_ko_relay", "target", "higher"),
        MetricPriority("attack_quality", "penalty", "lower"),
    ),
    metric_ids=AUTO_ITERATION_METRIC_IDS,
    plugin_factories=(
        *_core_factories(),
        _lazy_factory("evaluation.metrics.setup_relay", "SetupRelayPlugin"),
        _lazy_factory("evaluation.metrics.attack_quality", "AttackQualityPlugin"),
    ),
)


_PROFILES = {
    CORE_PROFILE_ID: CORE_PROFILE,
    AUTO_ITERATION_PROFILE_ID: AUTO_ITERATION_PROFILE,
}


def available_metric_profiles() -> tuple[str, ...]:
    return tuple(_PROFILES)


def get_metric_profile(profile_id: str) -> MetricProfile:
    try:
        return _PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown metric profile: {profile_id}") from exc
