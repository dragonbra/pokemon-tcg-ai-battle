"""Self-contained frozen R15 implementation and RL policy helpers."""

PROJECT_ID = "0020_pluggable_deck_rl"

from .actor_critic import DragapultActorCritic, load_source_actor_critic

__all__ = ["DragapultActorCritic", "load_source_actor_critic"]
