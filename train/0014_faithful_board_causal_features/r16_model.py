"""R16 deterministic R2 policy with an explicit rule-contract option reader."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from .ac_model import ACModelConfig
from .r2_model import R2ModelConfig, R2StrongScenarioPolicy, ScenarioScaleGate


@dataclass(frozen=True)
class R16ModelConfig:
    """Preserve R2 and add actor-visible typed rule-contract tokens."""

    ac: ACModelConfig = field(default_factory=ACModelConfig)
    scenario_layers: int = 2
    scenario_ffn_multiplier: int = 3
    scale_gate_ffn_multiplier: int = 2
    rule_layers: int = 1
    rule_ffn_multiplier: int = 2

    def validate(self) -> None:
        self.ac.validate()
        if self.scenario_layers < 1 or self.scenario_ffn_multiplier < 2:
            raise ValueError("invalid R16 scenario encoder depth or width")
        if self.scale_gate_ffn_multiplier < 1:
            raise ValueError("invalid R16 ScaleGate width")
        if self.rule_layers < 1 or self.rule_ffn_multiplier < 1:
            raise ValueError("invalid R16 rule-contract depth or width")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class R16RuleContractPolicy(R2StrongScenarioPolicy):
    """Let each option read typed turn, race, library, and relay contracts."""

    rule_role_names = (
        "turn_budget",
        "prize_race",
        "library_pressure",
        "board_relay",
    )

    def __init__(self, config: R16ModelConfig, *, ontology_path: Path | str) -> None:
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
        self.r16_config = config
        d = self.config.d_model

        self.rule_role = nn.Parameter(torch.empty(len(self.rule_role_names), d))
        self.turn_budget_reader = nn.Sequential(
            nn.Linear(10, d), nn.GELU(), nn.Linear(d, d)
        )
        self.prize_race_reader = nn.Sequential(
            nn.Linear(16, d), nn.GELU(), nn.Linear(d, d)
        )
        self.library_pressure_reader = nn.Sequential(
            nn.Linear(12, d), nn.GELU(), nn.Linear(d, d)
        )
        self.board_relay_reader = nn.Sequential(
            nn.Linear(19, d), nn.GELU(), nn.Linear(d, d)
        )
        self.rule_input_norm = nn.LayerNorm(d)
        rule_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=self.config.heads,
            dim_feedforward=d * config.rule_ffn_multiplier,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.rule_encoder = nn.TransformerEncoder(
            rule_layer, num_layers=config.rule_layers, norm=nn.LayerNorm(d)
        )
        self.option_rule_attention = nn.MultiheadAttention(
            d, self.config.heads, dropout=self.config.dropout, batch_first=True
        )
        self.rule_delta = nn.Sequential(
            nn.LayerNorm(3 * d),
            nn.Linear(3 * d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, d),
        )
        self.rule_scale_gate = ScenarioScaleGate(
            3 * d, d, config.scale_gate_ffn_multiplier
        )
        self.rule_output_norm = nn.LayerNorm(d)
        nn.init.normal_(self.rule_role, std=0.02)

    @staticmethod
    def _flag_bits(flags: Tensor) -> Tensor:
        shifts = torch.arange(5, device=flags.device)
        return ((flags.unsqueeze(1) >> shifts) & 1).to(torch.float32)

    def _rule_tokens(self, batch: dict[str, Tensor]) -> Tensor:
        global_num = batch["global_num"].float()
        global_cat = batch["global_cat"]
        zone = batch["zone_inventory_num"].float() / self.zone_scale
        flags = self._flag_bits(global_cat[:, 3])

        categorical = (
            self.select_type(global_cat[:, 0])
            + self.select_context(global_cat[:, 1])
            + self.first_player(global_cat[:, 2])
            + self.flags(global_cat[:, 3])
        )
        turn_budget = torch.cat(
            (global_num[:, [0, 1, 8, 9, 10]], flags), dim=-1
        )
        prize_race = torch.cat(
            (
                global_num[:, 2:8],
                zone[:, :, [3, 4, 6, 9, 11]].flatten(1),
            ),
            dim=-1,
        )
        library_pressure = zone[:, :, [0, 1, 2, 5, 12, 13]].flatten(1)
        board_relay = torch.cat(
            (
                zone[:, :, [3, 4, 6, 7, 8, 9, 10, 11, 15]].flatten(1),
                global_num[:, [11]],
            ),
            dim=-1,
        )
        numeric = torch.stack(
            (
                self.turn_budget_reader(turn_budget),
                self.prize_race_reader(prize_race),
                self.library_pressure_reader(library_pressure),
                self.board_relay_reader(board_relay),
            ),
            dim=1,
        )
        tokens = numeric + categorical.unsqueeze(1) + self.rule_role.unsqueeze(0)
        return self.rule_encoder(self.rule_input_norm(tokens))

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        board_state, options = super().encode(batch)
        rules = self._rule_tokens(batch)
        rule_read, _ = self.option_rule_attention(
            options, rules, rules, need_weights=False
        )
        state = board_state.unsqueeze(1).expand(-1, options.size(1), -1)
        gate_input = torch.cat((options, rule_read, state), dim=-1)
        delta = self.rule_delta(gate_input)
        scale = self.rule_scale_gate(gate_input)
        return board_state, self.rule_output_norm(options + scale * delta)


def parameter_count(model: R16RuleContractPolicy) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = ["R16ModelConfig", "R16RuleContractPolicy", "parameter_count"]
