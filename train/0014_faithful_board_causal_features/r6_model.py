"""R6 sparse family mixture derived from the R4 additive-router failure."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from .ac_model import ACModelConfig, AllFeatureCausalPolicy


@dataclass(frozen=True)
class R6ModelConfig:
    """Option experts compete with an explicit null/base route."""

    ac: ACModelConfig = field(default_factory=ACModelConfig)
    expert_ffn_multiplier: int = 2

    def validate(self) -> None:
        self.ac.validate()
        if self.expert_ffn_multiplier < 1:
            raise ValueError("invalid R6 expert width")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class R6SparseFamilyMixturePolicy(AllFeatureCausalPolicy):
    """Mix option evidence families competitively instead of summing every route."""

    family_names = ("ledger", "goal", "zone", "event", "hand")

    def __init__(self, config: R6ModelConfig, *, ontology_path: Path | str) -> None:
        config.validate()
        super().__init__(config.ac, ontology_path=ontology_path)
        self.r6_config = config
        d = self.config.d_model
        width = d * config.expert_ffn_multiplier

        # Preserve local card-capability adapters, but close aggregate scenario
        # routes that would duplicate the explicit mixture below.
        for parameter in (
            self.global_zone_gate,
            self.state_aux_gate,
            self.option_aux_gate,
        ):
            parameter.requires_grad_(False)

        self.zone_option_kind = nn.Parameter(torch.empty(d))
        self.ledger_attention = nn.MultiheadAttention(
            d, self.config.heads, dropout=self.config.dropout, batch_first=True
        )
        self.zone_attention = nn.MultiheadAttention(
            d, self.config.heads, dropout=self.config.dropout, batch_first=True
        )
        self.role_router = nn.Linear(d, len(self.goal_role_names), bias=False)
        self.role_value = nn.Linear(d, d, bias=False)
        self.event_relevance = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
        self.hand_relevance = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())

        self.experts = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(d),
                    nn.Linear(d, width),
                    nn.GELU(),
                    nn.Linear(width, d),
                )
                for name in self.family_names
            }
        )
        self.family_scorers = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(2 * d),
                    nn.Linear(2 * d, d),
                    nn.GELU(),
                    nn.Linear(d, 1),
                )
                for name in self.family_names
            }
        )
        self.null_scorer = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
        )
        priors = {
            "null": 0.50,
            "ledger": 0.20,
            "goal": 0.15,
            "zone": 0.075,
            "event": 0.0375,
            "hand": 0.0375,
        }
        for name, scorer in self.family_scorers.items():
            output = scorer[-1]
            assert isinstance(output, nn.Linear)
            nn.init.zeros_(output.weight)
            nn.init.constant_(output.bias, torch.log(torch.tensor(priors[name])).item())
        null_output = self.null_scorer[-1]
        assert isinstance(null_output, nn.Linear)
        nn.init.zeros_(null_output.weight)
        nn.init.constant_(null_output.bias, torch.log(torch.tensor(priors["null"])).item())

        self.amplitude = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, width),
            nn.GELU(),
            nn.Linear(width, d),
        )
        amplitude_output = self.amplitude[-1]
        assert isinstance(amplitude_output, nn.Linear)
        nn.init.zeros_(amplitude_output.weight)
        nn.init.zeros_(amplitude_output.bias)
        self.output_norm = nn.LayerNorm(d)
        nn.init.normal_(self.zone_option_kind, std=0.02)

    def _zone_tokens(self, batch: dict[str, Tensor]) -> Tensor:
        values = batch["zone_inventory_num"] / self.zone_scale
        owners = self.zone_owner(
            torch.arange(2, device=values.device).view(1, 2).expand(values.size(0), -1)
        )
        return self.zone_reader(values) + owners + self.zone_option_kind

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        board_state, options = super().encode(batch)
        resources = self._resource_tokens(batch)
        ledger_read, _ = self.ledger_attention(
            options,
            resources,
            resources,
            key_padding_mask=~batch["registered_mask"],
            need_weights=False,
        )
        zones = self._zone_tokens(batch)
        zone_read, _ = self.zone_attention(options, zones, zones, need_weights=False)
        goals = self.goal_qkv(board_state, resources, batch["registered_mask"])
        role_weight = torch.softmax(self.role_router(options), dim=-1)
        goal_read = torch.einsum("bor,brd->bod", role_weight, self.role_value(goals))
        event = self._event_context(batch).unsqueeze(1)
        hand = self._hand_context(batch).unsqueeze(1)
        evidence = {
            "ledger": ledger_read,
            "goal": goal_read,
            "zone": zone_read,
            "event": event * self.event_relevance(options),
            "hand": hand * self.hand_relevance(options),
        }
        expert_values = [self.experts[name](evidence[name]) for name in self.family_names]
        logits = [self.null_scorer(options)]
        logits.extend(
            self.family_scorers[name](torch.cat((options, evidence[name]), dim=-1))
            for name in self.family_names
        )
        weights = torch.softmax(torch.cat(logits, dim=-1), dim=-1)
        mixture = sum(
            (
                weights[..., index + 1 : index + 2] * value
                for index, value in enumerate(expert_values)
            ),
            start=torch.zeros_like(options),
        )
        amplitude = 2.0 * torch.sigmoid(
            self.amplitude(torch.cat((options, mixture), dim=-1))
        )
        return board_state, self.output_norm(options + amplitude * mixture)


def parameter_count(model: R6SparseFamilyMixturePolicy) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = ["R6ModelConfig", "R6SparseFamilyMixturePolicy", "parameter_count"]
