"""R10 bounded state-calibration policy derived from the R8 deletion failure."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from .ac_model import ACModelConfig
from .r2_model import R2ModelConfig, R2StrongScenarioPolicy


class BoundedStateScaleGate(nn.Module):
    """A small positive state-only scale; the option gate remains unchanged."""

    def __init__(
        self,
        input_width: int,
        d_model: int,
        multiplier: int,
        *,
        maximum: float,
        initial: float,
    ) -> None:
        super().__init__()
        if not 0.0 < initial < maximum:
            raise ValueError("state scale initial value must be inside its open range")
        self.maximum = maximum
        self.network = nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, d_model * multiplier),
            nn.GELU(),
            nn.Linear(d_model * multiplier, d_model),
        )
        output = self.network[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        probability = torch.tensor(initial / maximum)
        nn.init.constant_(output.bias, float(torch.logit(probability)))

    def forward(self, scenario: Tensor) -> Tensor:
        return self.maximum * torch.sigmoid(self.network(scenario))


@dataclass(frozen=True)
class R10ModelConfig:
    """Full R2 with a bounded low-amplitude global-state calibration path."""

    ac: ACModelConfig = field(default_factory=ACModelConfig)
    scenario_layers: int = 2
    scenario_ffn_multiplier: int = 3
    scale_gate_ffn_multiplier: int = 2
    state_maximum_scale: float = 0.25
    state_initial_scale: float = 0.05

    def validate(self) -> None:
        self.ac.validate()
        if self.scenario_layers < 1 or self.scenario_ffn_multiplier < 2:
            raise ValueError("invalid R10 scenario encoder depth or width")
        if self.scale_gate_ffn_multiplier < 1:
            raise ValueError("invalid R10 ScaleGate width")
        if not 0.0 < self.state_initial_scale < self.state_maximum_scale:
            raise ValueError("invalid R10 state scale contract")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class R10BoundedStateCalibrationPolicy(R2StrongScenarioPolicy):
    """Restore R2's state reader but constrain it to contextual calibration."""

    def __init__(self, config: R10ModelConfig, *, ontology_path: Path | str) -> None:
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
        self.r10_config = config
        d = self.config.d_model
        self.state_scale_gate = BoundedStateScaleGate(
            4 * d,
            d,
            config.scale_gate_ffn_multiplier,
            maximum=config.state_maximum_scale,
            initial=config.state_initial_scale,
        )


def parameter_count(model: R10BoundedStateCalibrationPolicy) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = [
    "BoundedStateScaleGate",
    "R10BoundedStateCalibrationPolicy",
    "R10ModelConfig",
    "parameter_count",
]
