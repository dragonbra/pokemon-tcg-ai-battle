"""Auditable expert-persona residual over the frozen V12 R2 backbone."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from .r2_model import R2ModelConfig, R2StrongScenarioPolicy


@dataclass(frozen=True)
class SourceModelConfig:
    vocabulary_size: int = 25
    initial_scale: float = 0.05

    def validate(self) -> None:
        if self.vocabulary_size < 1:
            raise ValueError("source vocabulary must be nonempty")
        if not 0.0 < self.initial_scale < 1.0:
            raise ValueError("source initial scale must be in (0, 1)")


class SourceConditionedR2Policy(R2StrongScenarioPolicy):
    """Keep V12 R2 intact and add a zero-neutral source-persona residual."""

    def __init__(
        self,
        config: R2ModelConfig,
        source_config: SourceModelConfig,
        *,
        ontology_path: Path | str,
    ) -> None:
        source_config.validate()
        super().__init__(config, ontology_path=ontology_path)
        self.source_config = source_config
        width = self.config.d_model
        self.source_persona = nn.Embedding(
            source_config.vocabulary_size + 1,
            width,
            padding_idx=0,
        )
        self.source_state = nn.Linear(width, width, bias=False)
        self.source_option = nn.Linear(width, width, bias=False)
        inverse_scale = torch.atanh(torch.tensor(source_config.initial_scale))
        self.source_gate = nn.Parameter(torch.full((width,), float(inverse_scale)))
        nn.init.normal_(self.source_persona.weight[1:], std=0.02)
        with torch.no_grad():
            self.source_persona.weight[0].zero_()

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        state, options = super().encode(batch)
        source_id = batch.get("source_id")
        if source_id is None:
            raise ValueError("source-conditioned R2 requires source_id")
        if source_id.ndim != 1 or source_id.size(0) != state.size(0):
            raise ValueError("source_id batch dimension mismatch")
        if source_id.min() < 0 or source_id.max() > self.source_config.vocabulary_size:
            raise ValueError("source_id is outside the frozen vocabulary")
        persona = self.source_persona(source_id)
        gate = torch.tanh(self.source_gate)
        state = state + gate * self.source_state(persona)
        options = options + gate * self.source_option(persona).unsqueeze(1)
        return state, options


__all__ = ["SourceConditionedR2Policy", "SourceModelConfig"]
