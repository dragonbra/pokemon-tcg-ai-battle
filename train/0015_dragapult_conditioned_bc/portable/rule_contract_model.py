"""Portable source-conditioned R15 rule-contract model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from .r15_model import R15ModelConfig
from .source_model import SourceConditionedR15Config, SourceConditionedR15Policy


@dataclass(frozen=True)
class SourceConditionedR15RuleConfig:
    r15: R15ModelConfig
    source_vocabulary_size: int
    source_initial_scale: float = 0.10
    rule_layers: int = 1
    rule_ffn_multiplier: int = 2
    rule_initial_scale: float = 0.10
    rule_model_width: int = 320
    rule_heads: int = 8

    def base_config(self) -> SourceConditionedR15Config:
        return SourceConditionedR15Config(
            r15=self.r15,
            source_vocabulary_size=self.source_vocabulary_size,
            source_initial_scale=self.source_initial_scale,
        )


class _DoubleSigmoid(nn.Module):
    def forward(self, values: Tensor) -> Tensor:
        return 2.0 * torch.sigmoid(values)


class SourceConditionedR15RulePolicy(SourceConditionedR15Policy):
    rule_role_names = (
        "turn_budget",
        "prize_race",
        "library_pressure",
        "board_relay",
    )
    rule_token_widths = (10, 16, 12, 19)

    def __init__(
        self,
        config: SourceConditionedR15RuleConfig,
        *,
        ontology_path: Path | str,
    ) -> None:
        super().__init__(config.base_config(), ontology_path=ontology_path)
        self.rule_config = config
        base_width = self.config.d_model
        width = config.rule_model_width
        self.rule_role = nn.Parameter(torch.empty(len(self.rule_role_names), width))
        self.rule_option_projection = self._projection(base_width, width)
        self.rule_state_projection = self._projection(base_width, width)
        self.rule_categorical_projection = self._projection(base_width, width)
        self.turn_budget_reader = self._reader(10, width)
        self.prize_race_reader = self._reader(16, width)
        self.library_pressure_reader = self._reader(12, width)
        self.board_relay_reader = self._reader(19, width)
        self.rule_input_norm = nn.LayerNorm(width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.rule_heads,
            dim_feedforward=width * config.rule_ffn_multiplier,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.rule_encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.rule_layers,
            norm=nn.LayerNorm(width),
        )
        self.option_rule_attention = nn.MultiheadAttention(
            width,
            config.rule_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.rule_delta = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, base_width),
        )
        self.rule_scale_gate = self._scale_gate(3 * width, base_width)
        self.rule_output_norm = nn.LayerNorm(base_width)
        nn.init.normal_(self.rule_role, std=0.02)

    @staticmethod
    def _reader(input_width: int, model_width: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_width, model_width),
            nn.GELU(),
            nn.Linear(model_width, model_width),
        )

    @staticmethod
    def _projection(input_width: int, output_width: int) -> nn.Module:
        if input_width == output_width:
            return nn.Identity()
        return nn.Linear(input_width, output_width)

    def _scale_gate(self, input_width: int, model_width: int) -> nn.Sequential:
        gate = nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, model_width * self.rule_config.rule_ffn_multiplier),
            nn.GELU(),
            nn.Linear(model_width * self.rule_config.rule_ffn_multiplier, model_width),
        )
        output = gate[-1]
        assert isinstance(output, nn.Linear)
        probability = torch.tensor(self.rule_config.rule_initial_scale / 2.0)
        nn.init.zeros_(output.weight)
        nn.init.constant_(output.bias, float(torch.logit(probability)))
        return nn.Sequential(gate, _DoubleSigmoid())

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
        turn_budget = torch.cat((global_num[:, [0, 1, 8, 9, 10]], flags), dim=-1)
        prize_race = torch.cat(
            (global_num[:, 2:8], zone[:, :, [3, 4, 6, 9, 11]].flatten(1)),
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
        categorical = self.rule_categorical_projection(categorical)
        tokens = numeric + categorical.unsqueeze(1) + self.rule_role.unsqueeze(0)
        return self.rule_encoder(self.rule_input_norm(tokens))

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        state, options = super().encode(batch)
        rules = self._rule_tokens(batch)
        rule_options = self.rule_option_projection(options)
        rule_read, _ = self.option_rule_attention(
            rule_options, rules, rules, need_weights=False
        )
        rule_state = self.rule_state_projection(state)
        expanded_state = rule_state.unsqueeze(1).expand(-1, options.size(1), -1)
        gate_input = torch.cat((rule_options, rule_read, expanded_state), dim=-1)
        delta = self.rule_delta(gate_input)
        scale = self.rule_scale_gate(gate_input)
        return state, self.rule_output_norm(options + scale * delta)


__all__ = ["SourceConditionedR15RuleConfig", "SourceConditionedR15RulePolicy"]
