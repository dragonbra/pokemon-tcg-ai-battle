"""R11 strong-scale evidence-dropout policy derived from the R9 result."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from .ac_model import ACModelConfig
from .r7_model import R7ModelConfig, R7RegularizedOptionQueryPolicy


@dataclass(frozen=True)
class R11ModelConfig:
    """Combine R9's strong prior with the robustness mechanism retained by R7."""

    ac: ACModelConfig = field(default_factory=ACModelConfig)
    scenario_layers: int = 1
    scenario_ffn_multiplier: int = 2
    route_ffn_multiplier: int = 1
    evidence_dropout: float = 0.2
    initial_scale: float = 1.0

    def validate(self) -> None:
        self.ac.validate()
        if self.scenario_layers < 1 or self.scenario_ffn_multiplier < 2:
            raise ValueError("invalid R11 scenario encoder depth or width")
        if self.route_ffn_multiplier < 1:
            raise ValueError("invalid R11 route width")
        if not 0.0 < self.evidence_dropout < 1.0:
            raise ValueError("R11 requires positive evidence dropout")
        if self.initial_scale != 1.0:
            raise ValueError("R11 requires a unit initial option scale")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class R11StrongDropoutOptionQueryPolicy(R7RegularizedOptionQueryPolicy):
    """A strong option-query reader trained against missing-evidence shortcuts."""

    def __init__(self, config: R11ModelConfig, *, ontology_path: Path | str) -> None:
        config.validate()
        super().__init__(
            R7ModelConfig(
                ac=config.ac,
                scenario_layers=config.scenario_layers,
                scenario_ffn_multiplier=config.scenario_ffn_multiplier,
                route_ffn_multiplier=config.route_ffn_multiplier,
                evidence_dropout=config.evidence_dropout,
                initial_scale=config.initial_scale,
            ),
            ontology_path=ontology_path,
        )
        self.r11_config = config


def parameter_count(model: R11StrongDropoutOptionQueryPolicy) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = ["R11ModelConfig", "R11StrongDropoutOptionQueryPolicy", "parameter_count"]
