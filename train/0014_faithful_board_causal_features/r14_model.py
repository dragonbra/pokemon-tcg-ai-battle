"""R14 full R2 policy with structured scenario-only modality dropout."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from .ac_model import ACModelConfig, AllFeatureCausalPolicy
from .r2_model import R2ModelConfig, R2StrongScenarioPolicy


class ScenarioModalityDropout(nn.Module):
    """Drop a complete scenario route per sample without corrupting its channels."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("invalid scenario modality dropout probability")
        self.probability = probability

    def forward(self, values: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return values
        keep = torch.rand(
            (values.size(0), 1, 1), device=values.device, dtype=values.dtype
        ) >= self.probability
        return values * keep / (1.0 - self.probability)


@dataclass(frozen=True)
class R14ModelConfig:
    """Keep exact state/Goal evidence and regularize only scenario availability."""

    ac: ACModelConfig = field(default_factory=ACModelConfig)
    scenario_layers: int = 2
    scenario_ffn_multiplier: int = 3
    scale_gate_ffn_multiplier: int = 2
    scenario_modality_dropout: float = 0.2

    def validate(self) -> None:
        self.ac.validate()
        if self.scenario_layers < 1 or self.scenario_ffn_multiplier < 2:
            raise ValueError("invalid R14 scenario encoder depth or width")
        if self.scale_gate_ffn_multiplier < 1:
            raise ValueError("invalid R14 ScaleGate width")
        if not 0.0 < self.scenario_modality_dropout < 1.0:
            raise ValueError("R14 requires positive scenario modality dropout")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class R14ScenarioModalityDropoutPolicy(R2StrongScenarioPolicy):
    """Regularize scenario shortcuts while preserving precise Goal-QKV geometry."""

    def __init__(self, config: R14ModelConfig, *, ontology_path: Path | str) -> None:
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
        self.r14_config = config
        self.scenario_modality_dropout = ScenarioModalityDropout(
            config.scenario_modality_dropout
        )

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        board_state, options = AllFeatureCausalPolicy.encode(self, batch)
        memory, memory_mask, family_summary, resources = self._scenario_bank(batch)

        state_read, _ = self.state_scenario_attention(
            board_state.unsqueeze(1),
            memory,
            memory,
            key_padding_mask=~memory_mask,
            need_weights=False,
        )
        state_read = state_read.squeeze(1)
        summary = self.scenario_summary(family_summary)
        state_parameters = self.state_film(torch.cat((state_read, summary), dim=-1))
        state_delta = self._film_delta(state_read, state_parameters)
        state_scale = self.state_scale_gate(family_summary)
        conditioned_state = self.state_output_norm(board_state + state_scale * state_delta)

        option_read, _ = self.option_scenario_attention(
            options,
            memory,
            memory,
            key_padding_mask=~memory_mask,
            need_weights=False,
        )
        goals = self.goal_qkv(board_state, resources, batch["registered_mask"])
        role_weight = torch.softmax(self.role_router(options), dim=-1)
        role_read = torch.einsum("bor,brd->bod", role_weight, self.role_value(goals))

        # One shared mask per sample preserves the scenario vector geometry and
        # option consistency. Goal-QKV and the entire state path are untouched.
        option_read = self.scenario_modality_dropout(option_read)
        summary_for_options = summary.unsqueeze(1).expand(-1, options.size(1), -1)
        option_parameters = self.option_film(
            torch.cat((option_read, role_read, summary_for_options), dim=-1)
        )
        option_evidence = option_read + role_read
        option_delta = self._film_delta(option_evidence, option_parameters)
        option_scale = self.option_scale_gate(
            torch.cat((option_read, summary_for_options), dim=-1)
        )
        conditioned_options = self.option_output_norm(
            options + option_scale * option_delta
        )
        return conditioned_state, conditioned_options


def parameter_count(model: R14ScenarioModalityDropoutPolicy) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = [
    "R14ModelConfig",
    "R14ScenarioModalityDropoutPolicy",
    "ScenarioModalityDropout",
    "parameter_count",
]
