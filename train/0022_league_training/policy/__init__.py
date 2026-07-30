"""Self-contained actor-critic policy surface for 0022 League training."""

from .actor_critic import LeagueActorCritic, load_league_actor_critic

__all__ = ["LeagueActorCritic", "load_league_actor_critic"]
