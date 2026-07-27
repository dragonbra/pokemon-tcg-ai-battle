"""Self-contained frozen R15 implementation and RL policy helpers."""

PROJECT_ID = "0017_dragapult_terminal_rl"

from .actor_critic import DragapultActorCritic, load_source_actor_critic

__all__ = ["DragapultActorCritic", "load_source_actor_critic"]
