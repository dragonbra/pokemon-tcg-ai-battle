"""Independently switchable 0038 integrated-RL modules."""

from .config import IntegratedFlags
from .loss_registry import LossRegistry, LossTerm
from .presets import PRESETS, preset
from .prize import PrizeAuxHead, combine_actor_advantages, prize_gae
from .opponent_meta import OpponentMetaConditioner, OpponentMetaHead
from .tempo import TempoEvent, TempoTracker

__all__ = [
    "IntegratedFlags", "LossRegistry", "LossTerm", "PRESETS", "PrizeAuxHead",
    "OpponentMetaConditioner", "OpponentMetaHead", "combine_actor_advantages",
    "TempoEvent", "TempoTracker", "preset", "prize_gae",
]
