"""Finite model and codec capacities for the POD-native actor."""

from __future__ import annotations

from dataclasses import asdict, dataclass


GLOBAL_CAT_VOCABS = (129, 257, 4, 4, 65, 16, 16, 64)
ENTITY_CAT_VOCABS = (4097, 4, 32, 65, 8, 64)
OPTION_CAT_VOCABS = (65, 33, 33, 4, 4097, 4097, 4097, 129, 129, 129, 129, 129)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    d_model: int = 320
    heads: int = 8
    state_layers: int = 4
    option_layers: int = 2
    ffn_multiplier: int = 3
    dropout: float = 0.10
    max_action_steps: int = 64

    def validate(self) -> None:
        if self.d_model < 1 or self.d_model % self.heads:
            raise ValueError("d_model must be positive and divisible by heads")
        if min(self.state_layers, self.option_layers, self.ffn_multiplier) < 1:
            raise ValueError("layer counts and ffn_multiplier must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 1 <= self.max_action_steps <= 64:
            raise ValueError("max_action_steps must be in [1, 64]")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


__all__ = [
    "ENTITY_CAT_VOCABS",
    "GLOBAL_CAT_VOCABS",
    "ModelConfig",
    "OPTION_CAT_VOCABS",
]
