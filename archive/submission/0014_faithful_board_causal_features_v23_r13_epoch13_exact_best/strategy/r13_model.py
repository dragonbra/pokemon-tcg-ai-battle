"""R13 full-state policy with gradual option evidence scale."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from .ac_model import ACModelConfig
from .r12_model import R12ModelConfig, R12StatePreservingDropoutPolicy


@dataclass(frozen=True)
class R13ModelConfig:
    """Keep R12 state strength while lowering only the option-path prior."""

    ac: ACModelConfig = field(default_factory=ACModelConfig)
    scenario_layers: int = 2
    scenario_ffn_multiplier: int = 3
    scale_gate_ffn_multiplier: int = 2
    option_evidence_dropout: float = 0.2
    option_initial_scale: float = 0.35

    def validate(self) -> None:
        self.ac.validate()
        if self.scenario_layers < 1 or self.scenario_ffn_multiplier < 2:
            raise ValueError("invalid R13 scenario encoder depth or width")
        if self.scale_gate_ffn_multiplier < 1:
            raise ValueError("invalid R13 ScaleGate width")
        if not 0.0 < self.option_evidence_dropout < 1.0:
            raise ValueError("R13 requires positive option evidence dropout")
        if not 0.0 < self.option_initial_scale < 2.0:
            raise ValueError("invalid R13 option initial scale")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class R13GradualOptionScalePolicy(R12StatePreservingDropoutPolicy):
    """Use a strong state reader and a gradual regularized option reader."""

    def __init__(self, config: R13ModelConfig, *, ontology_path: Path | str) -> None:
        config.validate()
        super().__init__(
            R12ModelConfig(
                ac=config.ac,
                scenario_layers=config.scenario_layers,
                scenario_ffn_multiplier=config.scenario_ffn_multiplier,
                scale_gate_ffn_multiplier=config.scale_gate_ffn_multiplier,
                option_evidence_dropout=config.option_evidence_dropout,
            ),
            ontology_path=ontology_path,
        )
        self.r13_config = config
        output = self.option_scale_gate.network[-1]
        assert isinstance(output, nn.Linear)
        probability = torch.tensor(config.option_initial_scale / 2.0)
        nn.init.zeros_(output.weight)
        nn.init.constant_(output.bias, float(torch.logit(probability)))


def parameter_count(model: R13GradualOptionScalePolicy) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = ["R13GradualOptionScalePolicy", "R13ModelConfig", "parameter_count"]
