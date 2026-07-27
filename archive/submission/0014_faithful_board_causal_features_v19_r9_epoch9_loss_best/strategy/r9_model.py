"""R9 narrow strong option-query policy derived from the R7 evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from .ac_model import ACModelConfig
from .r7_model import R7ModelConfig, R7RegularizedOptionQueryPolicy


@dataclass(frozen=True)
class R9ModelConfig:
    """Keep R7's narrow reader but remove under-conditioning regularizers."""

    ac: ACModelConfig = field(default_factory=ACModelConfig)
    scenario_layers: int = 1
    scenario_ffn_multiplier: int = 2
    route_ffn_multiplier: int = 1
    evidence_dropout: float = 0.0
    initial_scale: float = 1.0

    def validate(self) -> None:
        self.ac.validate()
        if self.scenario_layers < 1 or self.scenario_ffn_multiplier < 2:
            raise ValueError("invalid R9 scenario encoder depth or width")
        if self.route_ffn_multiplier < 1:
            raise ValueError("invalid R9 route width")
        if self.evidence_dropout != 0.0:
            raise ValueError("R9 requires deterministic full evidence")
        if self.initial_scale != 1.0:
            raise ValueError("R9 requires a unit initial option scale")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class R9NarrowStrongOptionQueryPolicy(R7RegularizedOptionQueryPolicy):
    """A capacity-controlled query-only reader with full evidence from step zero."""

    def __init__(self, config: R9ModelConfig, *, ontology_path: Path | str) -> None:
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
        self.r9_config = config


def parameter_count(model: R9NarrowStrongOptionQueryPolicy) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = ["R9ModelConfig", "R9NarrowStrongOptionQueryPolicy", "parameter_count"]
