from typing import Any

from .protocol import Episode, RolloutJob, TrajectoryDecision

__all__ = ["Episode", "RolloutCollector", "RolloutJob", "TrajectoryDecision"]


def __getattr__(name: str) -> Any:
    if name == "RolloutCollector":
        from .collector import RolloutCollector

        return RolloutCollector
    raise AttributeError(name)
