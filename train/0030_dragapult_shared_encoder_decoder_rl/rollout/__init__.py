"""Official-engine rollout protocol with a Torch-light worker import path."""

from .protocol import EpisodeTrajectory, RolloutJob, TrajectoryDecision


def __getattr__(name: str):
    if name == "HeterogeneousRolloutCollector":
        from .collector import HeterogeneousRolloutCollector

        return HeterogeneousRolloutCollector
    raise AttributeError(name)

__all__ = [
    "EpisodeTrajectory",
    "HeterogeneousRolloutCollector",
    "RolloutJob",
    "TrajectoryDecision",
]
