from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


PrizeMode = Literal["off", "directional", "terminal_neutral"]


@dataclass(frozen=True, slots=True)
class IntegratedFlags:
    meta_anchor_coef: float = 0.10
    enable_action_boundary: bool = True
    enable_forced_shortcut: bool = True
    enable_dragapult_macro: bool = True
    enable_prize_aux: bool = False
    prize_aux_mode: PrizeMode = "off"
    prize_aux_scale: float = 1.0 / 24.0
    prize_aux_actor_weight: float = 0.0
    prize_value_loss_weight: float = 0.5
    enable_tempo_metrics: bool = True
    enable_tempo_curriculum: bool = False
    enable_tempo_aux_loss: bool = False

    def validate(self) -> None:
        if self.meta_anchor_coef <= 0:
            raise ValueError("0044 meta_anchor_coef must be explicitly positive")
        if not self.enable_action_boundary and (
            self.enable_forced_shortcut or self.enable_dragapult_macro
        ):
            raise ValueError("forced shortcut and Dragapult macro require action boundary")
        if self.enable_dragapult_macro and not self.enable_forced_shortcut:
            raise ValueError("Dragapult macro requires forced callback shortcut")
        if self.prize_aux_mode not in {"off", "directional", "terminal_neutral"}:
            raise ValueError("invalid prize_aux_mode")
        if self.enable_prize_aux != (self.prize_aux_mode != "off"):
            raise ValueError("enable_prize_aux and prize_aux_mode disagree")
        if (self.prize_aux_scale <= 0 or self.prize_aux_actor_weight < 0
                or self.prize_value_loss_weight < 0):
            raise ValueError("Prize scale/actor weight must be nonnegative")
        if self.enable_tempo_aux_loss and not self.enable_tempo_metrics:
            raise ValueError("Tempo auxiliary loss requires tempo metrics")

    def metadata(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


__all__ = ["IntegratedFlags", "PrizeMode"]
