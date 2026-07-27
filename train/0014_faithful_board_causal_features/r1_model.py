"""Structured conditional all-feature policy for the post-R1 comparison."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from .ac_model import ACModelConfig, AllFeatureCausalPolicy


@dataclass(frozen=True)
class R1ModelConfig:
    """AC trunk plus explicit scenario-memory conditioning of state and options."""

    ac: ACModelConfig = field(default_factory=ACModelConfig)
    conditioning_layers: int = 2
    conditioning_ffn_multiplier: int = 3
    initial_conditioning_strength: float = 0.10

    def validate(self) -> None:
        self.ac.validate()
        if self.conditioning_layers < 1 or self.conditioning_ffn_multiplier < 2:
            raise ValueError("invalid R1 conditioning depth or width")
        if not 0.0 < self.initial_conditioning_strength < 1.0:
            raise ValueError("initial conditioning strength must be in (0, 1)")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class R1StructuredCausalPolicy(AllFeatureCausalPolicy):
    """Actor-visible scenario memory with role-routed state/action conditioning.

    The parent AC representation remains an input-compatible encoder.  R1 makes
    the information flow non-incidental: resources are separate memory tokens,
    public zones remain separate player tokens, and each action retrieves and
    routes evidence through the four audited goal roles before pointer scoring.
    """

    def __init__(self, config: R1ModelConfig, *, ontology_path: Path | str) -> None:
        config.validate()
        super().__init__(config.ac, ontology_path=ontology_path)
        self.r1_config = config
        d = self.config.d_model
        heads = self.config.heads
        width = d * config.conditioning_ffn_multiplier

        self.zone_memory_kind = nn.Parameter(torch.empty(d))
        self.event_memory_kind = nn.Parameter(torch.empty(d))
        self.hand_memory_kind = nn.Parameter(torch.empty(d))
        self.goal_memory_kind = nn.Parameter(torch.empty(d))
        self.memory_norm = nn.LayerNorm(d)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=heads,
            dim_feedforward=width,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.memory_encoder = nn.TransformerEncoder(
            layer, num_layers=config.conditioning_layers, norm=nn.LayerNorm(d)
        )
        self.state_memory_attention = nn.MultiheadAttention(
            d, heads, dropout=self.config.dropout, batch_first=True
        )
        self.option_memory_attention = nn.MultiheadAttention(
            d, heads, dropout=self.config.dropout, batch_first=True
        )
        self.role_router = nn.Linear(d, len(self.goal_role_names), bias=False)
        self.role_value = nn.Linear(d, d, bias=False)
        self.state_film = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, 2 * d))
        self.option_film = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 2 * d))
        self.state_output_norm = nn.LayerNorm(d)
        self.option_output_norm = nn.LayerNorm(d)

        initial_logit = torch.logit(torch.tensor(config.initial_conditioning_strength))
        self.state_conditioning_logit = nn.Parameter(initial_logit.clone())
        self.option_conditioning_logit = nn.Parameter(initial_logit.clone())
        nn.init.normal_(self.zone_memory_kind, std=0.02)
        nn.init.normal_(self.event_memory_kind, std=0.02)
        nn.init.normal_(self.hand_memory_kind, std=0.02)
        nn.init.normal_(self.goal_memory_kind, std=0.02)

    def _zone_tokens(self, batch: dict[str, Tensor]) -> Tensor:
        values = batch["zone_inventory_num"] / self.zone_scale
        owners = self.zone_owner(
            torch.arange(2, device=values.device).view(1, 2).expand(values.size(0), -1)
        )
        return self.zone_reader(values) + owners + self.zone_memory_kind

    def _scenario_memory(
        self, batch: dict[str, Tensor], board_state: Tensor
    ) -> tuple[Tensor, Tensor]:
        resources = self._resource_tokens(batch)
        goals = self.goal_qkv(board_state, resources, batch["registered_mask"])
        zones = self._zone_tokens(batch)
        event = self._event_context(batch).unsqueeze(1) + self.event_memory_kind
        hand = self._hand_context(batch).unsqueeze(1) + self.hand_memory_kind
        goal_tokens = goals + self.goal_memory_kind
        memory = torch.cat((zones, resources, event, hand, goal_tokens), dim=1)
        mask = torch.cat(
            (
                torch.ones((memory.size(0), 2), dtype=torch.bool, device=memory.device),
                batch["registered_mask"],
                torch.ones((memory.size(0), 2 + len(self.goal_role_names)), dtype=torch.bool, device=memory.device),
            ),
            dim=1,
        )
        memory = self.memory_encoder(self.memory_norm(memory), src_key_padding_mask=~mask)
        return memory, memory[:, -len(self.goal_role_names) :]

    @staticmethod
    def _film(value: Tensor, parameters: Tensor) -> Tensor:
        scale, shift = parameters.chunk(2, dim=-1)
        return value * (1.0 + torch.tanh(scale)) + shift

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        board_state, options = super().encode(batch)
        memory, goal_tokens = self._scenario_memory(batch, board_state)

        state_read, _ = self.state_memory_attention(
            board_state.unsqueeze(1), memory, memory, need_weights=False
        )
        state_read = state_read.squeeze(1)
        state_strength = torch.sigmoid(self.state_conditioning_logit)
        conditioned_state = self.state_output_norm(
            board_state + state_strength * self._film(state_read, self.state_film(state_read))
        )

        option_read, _ = self.option_memory_attention(
            options, memory, memory, need_weights=False
        )
        role_weight = torch.softmax(self.role_router(options), dim=-1)
        role_read = torch.einsum("bor,brd->bod", role_weight, self.role_value(goal_tokens))
        option_context = torch.cat((option_read, role_read), dim=-1)
        option_strength = torch.sigmoid(self.option_conditioning_logit)
        conditioned_options = self.option_output_norm(
            options
            + option_strength * self._film(options, self.option_film(option_context))
        )
        return conditioned_state, conditioned_options


def parameter_count(model: R1StructuredCausalPolicy) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = ["R1ModelConfig", "R1StructuredCausalPolicy", "parameter_count"]
