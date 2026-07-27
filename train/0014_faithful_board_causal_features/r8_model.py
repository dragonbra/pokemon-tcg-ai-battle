"""R8 faithful R2 option-only conditioning derived from the R6 router failure."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from .ac_model import ACModelConfig, AllFeatureCausalPolicy
from .r2_model import R2ModelConfig, R2StrongScenarioPolicy


@dataclass(frozen=True)
class R8ModelConfig:
    """Retain the successful R2 option path and remove its collapsed state path."""

    ac: ACModelConfig = field(default_factory=ACModelConfig)
    scenario_layers: int = 2
    scenario_ffn_multiplier: int = 3
    scale_gate_ffn_multiplier: int = 2

    def validate(self) -> None:
        self.ac.validate()
        if self.scenario_layers < 1 or self.scenario_ffn_multiplier < 2:
            raise ValueError("invalid R8 scenario encoder depth or width")
        if self.scale_gate_ffn_multiplier < 1:
            raise ValueError("invalid R8 ScaleGate width")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class R8R2OptionOnlyPolicy(R2StrongScenarioPolicy):
    """Use R2's measured active path without allocating a global-state reader."""

    def __init__(self, config: R8ModelConfig, *, ontology_path: Path | str) -> None:
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
        self.r8_config = config
        # These modules are not merely gated off: R2 diagnostics showed that
        # the state scale collapsed, so R8 removes their capacity entirely.
        del self.state_scenario_attention
        del self.state_film
        del self.state_scale_gate
        del self.state_output_norm

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        board_state, options = AllFeatureCausalPolicy.encode(self, batch)
        memory, memory_mask, family_summary, resources = self._scenario_bank(batch)
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
        summary = self.scenario_summary(family_summary)
        summary_for_options = summary.unsqueeze(1).expand(-1, options.size(1), -1)
        option_parameters = self.option_film(
            torch.cat((option_read, role_read, summary_for_options), dim=-1)
        )
        option_evidence = option_read + role_read
        option_delta = self._film_delta(option_evidence, option_parameters)
        option_scale = self.option_scale_gate(
            torch.cat((option_read, summary_for_options), dim=-1)
        )
        return board_state, self.option_output_norm(options + option_scale * option_delta)


def parameter_count(model: R8R2OptionOnlyPolicy) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = ["R8ModelConfig", "R8R2OptionOnlyPolicy", "parameter_count"]
