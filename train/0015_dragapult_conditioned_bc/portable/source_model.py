"""Portable source-conditioned R15 model copied into candidate packages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from .r15_model import R15DeterministicGradualOptionPolicy, R15ModelConfig


@dataclass(frozen=True)
class SourceConditionedR15Config:
    r15: R15ModelConfig
    source_vocabulary_size: int
    source_initial_scale: float = 0.10


class SourceConditionedR15Policy(R15DeterministicGradualOptionPolicy):
    def __init__(self, config: SourceConditionedR15Config, *, ontology_path: Path | str) -> None:
        super().__init__(config.r15, ontology_path=ontology_path)
        self.source_config = config
        width = self.config.d_model
        self.source_persona = nn.Embedding(config.source_vocabulary_size + 1, width)
        self.source_state_norm = nn.LayerNorm(width)
        self.source_option_norm = nn.LayerNorm(width)
        initial = torch.tensor(config.source_initial_scale)
        self.source_gate = nn.Parameter(torch.full((width,), float(torch.atanh(initial))))
        nn.init.normal_(self.source_persona.weight, std=0.02)
        with torch.no_grad():
            self.source_persona.weight[0].zero_()

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        state, options = super().encode(batch)
        source_id = batch["source_id"]
        persona = self.source_persona(source_id)
        delta = torch.tanh(self.source_gate) * persona
        return (
            self.source_state_norm(state + delta),
            self.source_option_norm(options + delta.unsqueeze(1)),
        )


__all__ = ["SourceConditionedR15Config", "SourceConditionedR15Policy"]
