"""R12 full R2 state calibration with option-only evidence dropout."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from .ac_model import ACModelConfig, AllFeatureCausalPolicy
from .r2_model import R2ModelConfig, R2StrongScenarioPolicy


@dataclass(frozen=True)
class R12ModelConfig:
    """Keep the complete R2 reader while regularizing only option evidence."""

    ac: ACModelConfig = field(default_factory=ACModelConfig)
    scenario_layers: int = 2
    scenario_ffn_multiplier: int = 3
    scale_gate_ffn_multiplier: int = 2
    option_evidence_dropout: float = 0.2

    def validate(self) -> None:
        self.ac.validate()
        if self.scenario_layers < 1 or self.scenario_ffn_multiplier < 2:
            raise ValueError("invalid R12 scenario encoder depth or width")
        if self.scale_gate_ffn_multiplier < 1:
            raise ValueError("invalid R12 ScaleGate width")
        if not 0.0 < self.option_evidence_dropout < 1.0:
            raise ValueError("R12 requires positive option evidence dropout")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class R12StatePreservingDropoutPolicy(R2StrongScenarioPolicy):
    """Preserve R2 state calibration and regularize option-only shortcuts."""

    def __init__(self, config: R12ModelConfig, *, ontology_path: Path | str) -> None:
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
        self.r12_config = config
        self.option_evidence_dropout = nn.Dropout(config.option_evidence_dropout)

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

        # Drop only option-specific evidence during training. The global state
        # reader and summary remain lossless, and eval() observes full evidence.
        option_read = self.option_evidence_dropout(option_read)
        role_read = self.option_evidence_dropout(role_read)
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


def parameter_count(model: R12StatePreservingDropoutPolicy) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = ["R12ModelConfig", "R12StatePreservingDropoutPolicy", "parameter_count"]
