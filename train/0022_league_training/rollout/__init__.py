"""Official-engine rollout protocol with a Torch-light worker import path."""

from .protocol import EpisodeTrajectory, LeaguePolicyView, RolloutJob, TrajectoryDecision


def __getattr__(name: str):
    if name == "LeagueRolloutCollector":
        from .collector import LeagueRolloutCollector
        return LeagueRolloutCollector
    raise AttributeError(name)

__all__ = ["EpisodeTrajectory", "LeaguePolicyView", "LeagueRolloutCollector", "RolloutJob", "TrajectoryDecision"]
