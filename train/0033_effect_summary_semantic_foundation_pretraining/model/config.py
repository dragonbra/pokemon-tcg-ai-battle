"""Model capacity and finite identity limits."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    d_model: int = 320
    heads: int = 8
    state_layers: int = 4
    option_layers: int = 3
    ffn_multiplier: int = 3
    dropout: float = 0.10
    max_card_id: int = 2048
    max_attack_id: int = 2048
    max_skill_id: int = 512
    max_options: int = 128
    max_action_steps: int = 64

    def validate(self) -> None:
        if self.d_model <= 0 or self.d_model % self.heads:
            raise ValueError("d_model must be positive and divisible by heads")
        if min(self.state_layers, self.option_layers, self.ffn_multiplier) < 1:
            raise ValueError("layer counts and ffn_multiplier must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if min(
            self.max_card_id,
            self.max_attack_id,
            self.max_skill_id,
            self.max_options,
            self.max_action_steps,
        ) < 1:
            raise ValueError("identity and action limits must be positive")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


__all__ = ["ModelConfig"]
