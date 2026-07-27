"""R4 option-centric family router derived from R2 ScaleGate diagnostics."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from .ac_model import ACModelConfig, AllFeatureCausalPolicy


@dataclass(frozen=True)
class R4ModelConfig:
    """Separate option routes for evidence families; no global scenario state gate."""

    ac: ACModelConfig = field(default_factory=ACModelConfig)
    route_ffn_multiplier: int = 2

    def validate(self) -> None:
        self.ac.validate()
        if self.route_ffn_multiplier < 1:
            raise ValueError("invalid R4 route width")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class OptionFamilyRoute(nn.Module):
    """Project one evidence family through its own positive dynamic 0-2 gate."""

    def __init__(
        self, d_model: int, multiplier: int, *, initial_scale: float
    ) -> None:
        super().__init__()
        if not 0.0 < initial_scale < 2.0:
            raise ValueError("initial family scale must be in (0, 2)")
        self.projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * multiplier),
            nn.GELU(),
            nn.Linear(d_model * multiplier, d_model),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        output = self.gate[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        initial_probability = torch.tensor(initial_scale / 2.0)
        nn.init.constant_(output.bias, float(torch.logit(initial_probability)))

    def forward(self, evidence: Tensor) -> Tensor:
        scale = 2.0 * torch.sigmoid(self.gate(evidence))
        return scale * self.projection(evidence)


class R4OptionFamilyRouterPolicy(AllFeatureCausalPolicy):
    """Route ledger/Goal/card/zone/event/hand evidence directly into legal options."""

    family_names = ("ledger", "goal", "card", "zone", "event", "hand")

    def __init__(self, config: R4ModelConfig, *, ontology_path: Path | str) -> None:
        config.validate()
        super().__init__(config.ac, ontology_path=ontology_path)
        self.r4_config = config
        d = self.config.d_model
        heads = self.config.heads

        # R2 collapsed global state scale to ~0.03. Freeze the duplicated AC
        # state/option auxiliary gates closed and allocate R4 capacity only to
        # explicit, auditable option-family routes.
        for parameter in (
            self.global_zone_gate,
            self.state_aux_gate,
            self.option_semantic_gate,
            self.option_aux_gate,
        ):
            parameter.requires_grad_(False)

        self.zone_option_kind = nn.Parameter(torch.empty(d))
        self.ledger_attention = nn.MultiheadAttention(
            d, heads, dropout=self.config.dropout, batch_first=True
        )
        self.zone_attention = nn.MultiheadAttention(
            d, heads, dropout=self.config.dropout, batch_first=True
        )
        self.role_router = nn.Linear(d, len(self.goal_role_names), bias=False)
        self.role_value = nn.Linear(d, d, bias=False)
        self.event_relevance = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
        self.hand_relevance = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())

        initial_scales = {
            "ledger": 1.0,
            "goal": 1.0,
            "card": 1.0,
            "zone": 0.5,
            "event": 0.25,
            "hand": 0.25,
        }
        self.family_routes = nn.ModuleDict(
            {
                name: OptionFamilyRoute(
                    d, config.route_ffn_multiplier, initial_scale=initial_scales[name]
                )
                for name in self.family_names
            }
        )
        self.family_output_norm = nn.LayerNorm(d)
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
        goal_read = torch.einsum(
            "bor,brd->bod", role_weight, self.role_value(goals)
        )

        option = batch["option_cat"]
        card_read = self._card_features(option[..., 4]) + self._card_features(
            option[..., 5]
        )
        event = self._event_context(batch).unsqueeze(1)
        hand = self._hand_context(batch).unsqueeze(1)
        event_read = event * self.event_relevance(options)
        hand_read = hand * self.hand_relevance(options)
        evidence = {
            "ledger": ledger_read,
            "goal": goal_read,
            "card": card_read,
            "zone": zone_read,
            "event": event_read,
            "hand": hand_read,
        }
        routed = sum(
            (self.family_routes[name](evidence[name]) for name in self.family_names),
            start=torch.zeros_like(options),
        ) / math.sqrt(len(self.family_names))
        return board_state, self.family_output_norm(options + routed)


def parameter_count(model: R4OptionFamilyRouterPolicy) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = [
    "OptionFamilyRoute",
    "R4ModelConfig",
    "R4OptionFamilyRouterPolicy",
    "parameter_count",
]
