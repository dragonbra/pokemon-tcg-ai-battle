"""R15 policy with the auditable source-persona condition required by multi-team BC."""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

_r15 = importlib.import_module("train.0014_faithful_board_causal_features.r15_model")
R15ModelConfig = _r15.R15ModelConfig
R15DeterministicGradualOptionPolicy = _r15.R15DeterministicGradualOptionPolicy


@dataclass(frozen=True)
class SourceConditionedR15Config:
    r15: object = field(default_factory=R15ModelConfig)
    source_vocabulary_size: int = 128
    source_initial_scale: float = 0.10

    def validate(self) -> None:
        self.r15.validate()
        if self.source_vocabulary_size < 1:
            raise ValueError("source vocabulary must be nonempty")
        if not 0.0 < self.source_initial_scale < 1.0:
            raise ValueError("source initial scale must be in (0, 1)")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SourceConditionedR15Policy(R15DeterministicGradualOptionPolicy):
    """Keep R15 intact and add a small declared expert-persona residual."""

    def __init__(self, config: SourceConditionedR15Config, *, ontology_path: Path | str) -> None:
        config.validate()
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
        source_id = batch.get("source_id")
        if source_id is None:
            raise ValueError("source-conditioned R15 requires source_id")
        if source_id.ndim != 1 or source_id.size(0) != state.size(0):
            raise ValueError("source_id batch dimension mismatch")
        if source_id.min() < 0 or source_id.max() > self.source_config.source_vocabulary_size:
            raise ValueError("source_id is outside the frozen vocabulary")
        persona = self.source_persona(source_id)
        delta = torch.tanh(self.source_gate) * persona
        return (
            self.source_state_norm(state + delta),
            self.source_option_norm(options + delta.unsqueeze(1)),
        )


def parameter_count(model: SourceConditionedR15Policy) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = [
    "R15ModelConfig",
    "SourceConditionedR15Config",
    "SourceConditionedR15Policy",
    "parameter_count",
]
