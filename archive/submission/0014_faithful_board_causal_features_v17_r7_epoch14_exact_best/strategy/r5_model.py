"""R5 option-query scenario memory derived from the R3 early-fusion result."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from .ac_model import ACModelConfig, AllFeatureCausalPolicy


@dataclass(frozen=True)
class R5ModelConfig:
    """Keep the board trunk clean and expose typed scenario memory to options only."""

    ac: ACModelConfig = field(default_factory=ACModelConfig)
    scenario_layers: int = 1
    scenario_ffn_multiplier: int = 2
    route_ffn_multiplier: int = 2

    def validate(self) -> None:
        self.ac.validate()
        if self.scenario_layers < 1 or self.scenario_ffn_multiplier < 2:
            raise ValueError("invalid R5 scenario encoder depth or width")
        if self.route_ffn_multiplier < 1:
            raise ValueError("invalid R5 route width")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class R5OptionQueryScenarioPolicy(AllFeatureCausalPolicy):
    """Let legal-option queries read typed memory without altering board entities."""

    def __init__(self, config: R5ModelConfig, *, ontology_path: Path | str) -> None:
        config.validate()
        super().__init__(config.ac, ontology_path=ontology_path)
        self.r5_config = config
        d = self.config.d_model

        # R3 overfit after scenario tokens were inserted into every board layer.
        # Close only the duplicated aggregate state/option routes. Card semantics
        # remain trainable in entity and option identities.
        for parameter in (
            self.global_zone_gate,
            self.state_aux_gate,
            self.option_aux_gate,
        ):
            parameter.requires_grad_(False)

        self.zone_scenario_kind = nn.Parameter(torch.empty(d))
        self.event_scenario_kind = nn.Parameter(torch.empty(d))
        self.hand_scenario_kind = nn.Parameter(torch.empty(d))
        scenario_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=self.config.heads,
            dim_feedforward=d * config.scenario_ffn_multiplier,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.scenario_encoder = nn.TransformerEncoder(
            scenario_layer,
            num_layers=config.scenario_layers,
            norm=nn.LayerNorm(d),
        )
        self.scenario_input_norm = nn.LayerNorm(d)
        self.option_scenario_attention = nn.MultiheadAttention(
            d, self.config.heads, dropout=self.config.dropout, batch_first=True
        )
        self.role_router = nn.Linear(d, len(self.goal_role_names), bias=False)
        self.role_value = nn.Linear(d, d, bias=False)
        route_width = d * config.route_ffn_multiplier
        self.option_delta = nn.Sequential(
            nn.LayerNorm(3 * d),
            nn.Linear(3 * d, route_width),
            nn.GELU(),
            nn.Linear(route_width, d),
        )
        self.option_scale = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, route_width),
            nn.GELU(),
            nn.Linear(route_width, d),
        )
        scale_output = self.option_scale[-1]
        assert isinstance(scale_output, nn.Linear)
        nn.init.zeros_(scale_output.weight)
        nn.init.zeros_(scale_output.bias)
        self.option_output_norm = nn.LayerNorm(d)
        nn.init.normal_(self.zone_scenario_kind, std=0.02)
        nn.init.normal_(self.event_scenario_kind, std=0.02)
        nn.init.normal_(self.hand_scenario_kind, std=0.02)

    def _scenario_memory(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
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
        encoded = self.scenario_encoder(
            self.scenario_input_norm(memory), src_key_padding_mask=~mask
        )
        return encoded, mask, resources

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        board_state, options = super().encode(batch)
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
        delta = self.option_delta(torch.cat((options, scenario_read, goal_read), dim=-1))
        scale = 2.0 * torch.sigmoid(
            self.option_scale(torch.cat((options, scenario_read), dim=-1))
        )
        return board_state, self.option_output_norm(options + scale * delta)


def parameter_count(model: R5OptionQueryScenarioPolicy) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = ["R5ModelConfig", "R5OptionQueryScenarioPolicy", "parameter_count"]
