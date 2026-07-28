"""R2 policy with strong scenario-conditioned ScaleGate residuals."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from .ac_model import ACModelConfig, AllFeatureCausalPolicy


@dataclass(frozen=True)
class R2ModelConfig:
    """AC trunk plus independent Goal-QKV and dynamic scenario ScaleGates."""

    ac: ACModelConfig = field(default_factory=ACModelConfig)
    scenario_layers: int = 2
    scenario_ffn_multiplier: int = 3
    scale_gate_ffn_multiplier: int = 2

    def validate(self) -> None:
        self.ac.validate()
        if self.scenario_layers < 1 or self.scenario_ffn_multiplier < 2:
            raise ValueError("invalid R2 scenario encoder depth or width")
        if self.scale_gate_ffn_multiplier < 1:
            raise ValueError("invalid R2 ScaleGate width")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ScenarioScaleGate(nn.Module):
    """Map actor-visible scenario evidence to a positive per-channel 0-2 scale."""

    def __init__(self, input_width: int, d_model: int, multiplier: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, d_model * multiplier),
            nn.GELU(),
            nn.Linear(d_model * multiplier, d_model),
        )
        output = self.network[-1]
        assert isinstance(output, nn.Linear)
        # A neutral 1.0 scale is already ten times R1's initial 0.10 path,
        # while gradients can immediately specialize it by scenario.
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, scenario: Tensor) -> Tensor:
        return 2.0 * torch.sigmoid(self.network(scenario))


class R2StrongScenarioPolicy(AllFeatureCausalPolicy):
    """Strong scenario reader whose dynamic ScaleGates are independent of Goal-QKV."""

    def __init__(self, config: R2ModelConfig, *, ontology_path: Path | str) -> None:
        config.validate()
        super().__init__(config.ac, ontology_path=ontology_path)
        self.r2_config = config
        d = self.config.d_model
        heads = self.config.heads
        scenario_width = d * config.scenario_ffn_multiplier

        self.zone_scenario_kind = nn.Parameter(torch.empty(d))
        self.event_scenario_kind = nn.Parameter(torch.empty(d))
        self.hand_scenario_kind = nn.Parameter(torch.empty(d))
        self.scenario_input_norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=heads,
            dim_feedforward=scenario_width,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.scenario_encoder = nn.TransformerEncoder(
            layer, num_layers=config.scenario_layers, norm=nn.LayerNorm(d)
        )
        self.state_scenario_attention = nn.MultiheadAttention(
            d, heads, dropout=self.config.dropout, batch_first=True
        )
        self.option_scenario_attention = nn.MultiheadAttention(
            d, heads, dropout=self.config.dropout, batch_first=True
        )
        self.role_router = nn.Linear(d, len(self.goal_role_names), bias=False)
        self.role_value = nn.Linear(d, d, bias=False)

        family_width = 4 * d
        self.scenario_summary = nn.Sequential(
            nn.LayerNorm(family_width),
            nn.Linear(family_width, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
        )
        self.state_film = nn.Sequential(
            nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 2 * d)
        )
        self.option_film = nn.Sequential(
            nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 2 * d)
        )
        self.state_scale_gate = ScenarioScaleGate(
            family_width, d, config.scale_gate_ffn_multiplier
        )
        self.option_scale_gate = ScenarioScaleGate(
            2 * d, d, config.scale_gate_ffn_multiplier
        )
        self.state_output_norm = nn.LayerNorm(d)
        self.option_output_norm = nn.LayerNorm(d)

        nn.init.normal_(self.zone_scenario_kind, std=0.02)
        nn.init.normal_(self.event_scenario_kind, std=0.02)
        nn.init.normal_(self.hand_scenario_kind, std=0.02)

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        weights = mask.unsqueeze(-1)
        return (values * weights).sum(1) / weights.sum(1).clamp_min(1)

    @staticmethod
    def _film_delta(evidence: Tensor, parameters: Tensor) -> Tensor:
        scale, shift = parameters.chunk(2, dim=-1)
        return evidence * (1.0 + torch.tanh(scale)) + shift

    def _scenario_bank(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        values = batch["zone_inventory_num"] / self.zone_scale
        owners = self.zone_owner(
            torch.arange(2, device=values.device).view(1, 2).expand(values.size(0), -1)
        )
        zones = self.zone_reader(values) + owners + self.zone_scenario_kind
        resources = self._resource_tokens(batch)
        event = self._event_context(batch).unsqueeze(1) + self.event_scenario_kind
        hand = self._hand_context(batch).unsqueeze(1) + self.hand_scenario_kind
        memory = torch.cat((zones, resources, event, hand), dim=1)
        mask = torch.cat(
            (
                torch.ones((memory.size(0), 2), dtype=torch.bool, device=memory.device),
                batch["registered_mask"],
                torch.ones((memory.size(0), 2), dtype=torch.bool, device=memory.device),
            ),
            dim=1,
        )
        memory = self.scenario_encoder(
            self.scenario_input_norm(memory), src_key_padding_mask=~mask
        )
        resource_end = 2 + resources.size(1)
        family_summary = torch.cat(
            (
                memory[:, :2].mean(1),
                self._masked_mean(memory[:, 2:resource_end], batch["registered_mask"]),
                memory[:, resource_end],
                memory[:, resource_end + 1],
            ),
            dim=-1,
        )
        return memory, mask, family_summary, resources

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        board_state, options = super().encode(batch)
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
        # Goal-QKV is a separate planning path: it reads the raw registered-card
        # resource bank and never contributes to the Scenario ScaleGate input.
        goals = self.goal_qkv(board_state, resources, batch["registered_mask"])
        role_weight = torch.softmax(self.role_router(options), dim=-1)
        role_read = torch.einsum("bor,brd->bod", role_weight, self.role_value(goals))
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


def parameter_count(model: R2StrongScenarioPolicy) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = [
    "R2ModelConfig",
    "R2StrongScenarioPolicy",
    "ScenarioScaleGate",
    "parameter_count",
]
