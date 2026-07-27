"""R7 regularized option-query memory derived from R5 overfitting."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from .ac_model import ACModelConfig, AllFeatureCausalPolicy
from .r5_model import R5ModelConfig, R5OptionQueryScenarioPolicy


@dataclass(frozen=True)
class R7ModelConfig:
    """Lower-prior, narrower R5 path with evidence dropout."""

    ac: ACModelConfig = field(default_factory=ACModelConfig)
    scenario_layers: int = 1
    scenario_ffn_multiplier: int = 2
    route_ffn_multiplier: int = 1
    evidence_dropout: float = 0.2
    initial_scale: float = 0.35

    def validate(self) -> None:
        self.ac.validate()
        if self.scenario_layers < 1 or self.scenario_ffn_multiplier < 2:
            raise ValueError("invalid R7 scenario encoder depth or width")
        if self.route_ffn_multiplier < 1:
            raise ValueError("invalid R7 route width")
        if not 0.0 <= self.evidence_dropout < 1.0:
            raise ValueError("invalid R7 evidence dropout")
        if not 0.0 < self.initial_scale < 2.0:
            raise ValueError("invalid R7 initial scale")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class R7RegularizedOptionQueryPolicy(R5OptionQueryScenarioPolicy):
    """Retain R5's boundary while regularizing its scenario residual."""

    def __init__(self, config: R7ModelConfig, *, ontology_path: Path | str) -> None:
        config.validate()
        super().__init__(
            R5ModelConfig(
                ac=config.ac,
                scenario_layers=config.scenario_layers,
                scenario_ffn_multiplier=config.scenario_ffn_multiplier,
                route_ffn_multiplier=config.route_ffn_multiplier,
            ),
            ontology_path=ontology_path,
        )
        self.r7_config = config
        self.evidence_dropout = nn.Dropout(config.evidence_dropout)
        scale_output = self.option_scale[-1]
        assert isinstance(scale_output, nn.Linear)
        initial_probability = torch.tensor(config.initial_scale / 2.0)
        nn.init.zeros_(scale_output.weight)
        nn.init.constant_(scale_output.bias, float(torch.logit(initial_probability)))

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        board_state, options = AllFeatureCausalPolicy.encode(self, batch)
        memory, memory_mask, resources = self._scenario_memory(batch)
        scenario_read, _ = self.option_scenario_attention(
            options,
            memory,
            memory,
            key_padding_mask=~memory_mask,
            need_weights=False,
        )
        goals = self.goal_qkv(board_state, resources, batch["registered_mask"])
        role_weight = torch.softmax(self.role_router(options), dim=-1)
        goal_read = torch.einsum("bor,brd->bod", role_weight, self.role_value(goals))
        scenario_read = self.evidence_dropout(scenario_read)
        goal_read = self.evidence_dropout(goal_read)
        delta = self.option_delta(torch.cat((options, scenario_read, goal_read), dim=-1))
        scale = 2.0 * torch.sigmoid(
            self.option_scale(torch.cat((options, scenario_read), dim=-1))
        )
        return board_state, self.option_output_norm(options + scale * delta)


def parameter_count(model: R7RegularizedOptionQueryPolicy) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = ["R7ModelConfig", "R7RegularizedOptionQueryPolicy", "parameter_count"]
