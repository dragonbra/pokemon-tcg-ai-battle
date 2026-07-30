"""R15 deterministic full-R2 policy with gradual option scale."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from .ac_model import ACModelConfig
from .r2_model import R2ModelConfig, R2StrongScenarioPolicy


@dataclass(frozen=True)
class R15ModelConfig:
    """Preserve the R2 evidence paths and lower only the option residual prior."""

    ac: ACModelConfig = field(default_factory=ACModelConfig)
    scenario_layers: int = 2
    scenario_ffn_multiplier: int = 3
    scale_gate_ffn_multiplier: int = 2
    option_initial_scale: float = 0.35

    def validate(self) -> None:
        self.ac.validate()
        if self.scenario_layers < 1 or self.scenario_ffn_multiplier < 2:
            raise ValueError("invalid R15 scenario encoder depth or width")
        if self.scale_gate_ffn_multiplier < 1:
            raise ValueError("invalid R15 ScaleGate width")
        if not 0.0 < self.option_initial_scale < 2.0:
            raise ValueError("invalid R15 option initial scale")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class R15DeterministicGradualOptionPolicy(R2StrongScenarioPolicy):
    """Full deterministic R2 with a gradual but unconstrained option gate."""

    def __init__(self, config: R15ModelConfig, *, ontology_path: Path | str) -> None:
        config.validate()
        super().__init__(
            R2ModelConfig(
                ac=config.ac,
                scenario_layers=config.scenario_layers,
                scenario_ffn_multiplier=config.scenario_ffn_multiplier,
                scale_gate_ffn_multiplier=config.scale_gate_ffn_multiplier,
            ),
            ontology_path=ontology_path,
        )
        self.r15_config = config
        output = self.option_scale_gate.network[-1]
        assert isinstance(output, nn.Linear)
        probability = torch.tensor(config.option_initial_scale / 2.0)
        nn.init.zeros_(output.weight)
        nn.init.constant_(output.bias, float(torch.logit(probability)))


def parameter_count(model: R15DeterministicGradualOptionPolicy) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = ["R15DeterministicGradualOptionPolicy", "R15ModelConfig", "parameter_count"]
