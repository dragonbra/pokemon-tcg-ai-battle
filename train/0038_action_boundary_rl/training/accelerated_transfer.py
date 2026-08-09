"""Learning-rate and health contract for the accelerated 0038 acceptance run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


ACTOR_GROUPS = frozenset({
    "action_decoder", "allocation_head", "last_option_qv_lora",
    "local_output_layernorm", "opponent_conditioner",
})
VALUE_GROUPS = frozenset({"value_win", "value_prize"})


@dataclass(frozen=True, slots=True)
class AcceleratedTransferConfig:
    value_multiplier: float = 2.0
    actor_max_multiplier: float = 10.0
    behavior_kl_warning: float = 0.005
    behavior_kl_reduce: float = 0.01
    behavior_kl_rollback: float = 0.02
    clip_warning: float = 0.10
    clip_rollback: float = 0.30

    def validate(self) -> None:
        if not 2.0 <= self.value_multiplier <= 3.0:
            raise ValueError("Value multiplier must remain in the accepted 2x-3x range")
        if self.actor_max_multiplier not in {3.0, 5.0, 10.0}:
            raise ValueError("Actor maximum multiplier must be 3x, 5x, or 10x")
        if not (
            0 < self.behavior_kl_warning < self.behavior_kl_reduce
            < self.behavior_kl_rollback
        ):
            raise ValueError("behavior KL thresholds are not ordered")
        if not 0 < self.clip_warning < self.clip_rollback < 1:
            raise ValueError("clip thresholds are not ordered")


@dataclass(frozen=True, slots=True)
class UpdateHealth:
    behavior_kl: float
    clip_fraction: float
    warning: bool
    reduce_actor_lr: bool
    rollback: bool
    reason: str | None


class AcceleratedTransferController:
    def __init__(self, config: AcceleratedTransferConfig = AcceleratedTransferConfig()) -> None:
        config.validate()
        self.config = config
        self.actor_cap = config.actor_max_multiplier

    @staticmethod
    def scheduled_actor_multiplier(update: int) -> float:
        if update < 1:
            raise ValueError("PPO update must be positive")
        if update <= 2:
            return 3.0
        if update <= 5:
            return 5.0
        return 10.0

    def actor_multiplier(self, update: int) -> float:
        return min(self.scheduled_actor_multiplier(update), self.actor_cap)

    def learning_rates(
        self, base_learning_rates: Mapping[str, float], *, update: int
    ) -> dict[str, float]:
        actor = self.actor_multiplier(update)
        rates: dict[str, float] = {}
        for name, value in base_learning_rates.items():
            if name in ACTOR_GROUPS:
                rates[name] = float(value) * actor
            elif name in VALUE_GROUPS:
                rates[name] = float(value) * self.config.value_multiplier
            else:
                # Disabled modules must not silently enter this acceptance run.
                raise ValueError(f"unclassified optimizer group: {name}")
        return rates

    def health(self, metrics: Mapping[str, float]) -> UpdateHealth:
        behavior = max(
            float(metrics.get("ppo/behavior_kl", 0.0)),
            float(metrics.get("ppo/rejected_behavior_kl", 0.0)),
        )
        clip = float(metrics.get("ppo/clip_fraction", 0.0))
        early_stop = bool(metrics.get("ppo/target_kl_early_stop", 0.0))
        rollback = (
            early_stop
            or behavior >= self.config.behavior_kl_rollback
            or clip > self.config.clip_rollback
        )
        reduce = behavior > self.config.behavior_kl_reduce or rollback
        warning = behavior > self.config.behavior_kl_warning or clip > self.config.clip_warning
        reason = None
        if early_stop or behavior >= self.config.behavior_kl_rollback:
            reason = "behavior_kl"
        elif clip > self.config.clip_rollback:
            reason = "clip_fraction"
        elif reduce:
            reason = "behavior_kl_reduce"
        elif warning:
            reason = "warning"
        return UpdateHealth(behavior, clip, warning, reduce, rollback, reason)

    def reduce_actor_cap(self, *, current_multiplier: float | None = None) -> float:
        current = self.actor_cap if current_multiplier is None else current_multiplier
        if current > 5.0:
            self.actor_cap = 5.0
        elif current > 3.0:
            self.actor_cap = 3.0
        else:
            raise RuntimeError("accelerated Actor LR is unstable at the minimum 3x cap")
        return self.actor_cap

    def metadata(self) -> dict[str, object]:
        return {
            "schema": "0038_accelerated_transfer_lr_v1",
            "actor_schedule": {"1-2": 3.0, "3-5": 5.0, "6+": 10.0},
            "actor_cap": self.actor_cap,
            "value_multiplier": self.config.value_multiplier,
            "behavior_kl_warning": self.config.behavior_kl_warning,
            "behavior_kl_reduce": self.config.behavior_kl_reduce,
            "behavior_kl_rollback": self.config.behavior_kl_rollback,
            "clip_warning": self.config.clip_warning,
            "clip_rollback": self.config.clip_rollback,
        }


__all__ = [
    "ACTOR_GROUPS", "VALUE_GROUPS", "AcceleratedTransferConfig",
    "AcceleratedTransferController", "UpdateHealth",
]
