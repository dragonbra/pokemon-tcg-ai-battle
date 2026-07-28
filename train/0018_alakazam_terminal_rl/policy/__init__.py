"""Self-contained frozen R15 implementation and RL policy helpers."""

PROJECT_ID = "0018_alakazam_terminal_rl"

from .actor_critic import AlakazamActorCritic, load_source_actor_critic

__all__ = ["AlakazamActorCritic", "load_source_actor_critic"]
