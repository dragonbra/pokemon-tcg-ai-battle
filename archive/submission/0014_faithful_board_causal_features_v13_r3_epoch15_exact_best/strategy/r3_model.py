"""R3 policy with typed scenario tokens fused inside the board Transformer."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from .ac_model import ACModelConfig, AllFeatureCausalPolicy


@dataclass(frozen=True)
class R3ModelConfig:
    """Strong early fusion that makes scenario visible to every board layer."""

    ac: ACModelConfig = field(default_factory=ACModelConfig)
    scenario_layers: int = 1
    scenario_ffn_multiplier: int = 2

    def validate(self) -> None:
        self.ac.validate()
        if self.scenario_layers < 1 or self.scenario_ffn_multiplier < 2:
            raise ValueError("invalid R3 scenario encoder depth or width")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class R3EarlyScenarioFusionPolicy(AllFeatureCausalPolicy):
    """Fuse typed scenario tokens before board encoding and entity gathering."""

    def __init__(self, config: R3ModelConfig, *, ontology_path: Path | str) -> None:
        config.validate()
        super().__init__(config.ac, ontology_path=ontology_path)
        self.r3_config = config
        d = self.config.d_model
        self.zone_scenario_kind = nn.Parameter(torch.empty(d))
        self.event_scenario_kind = nn.Parameter(torch.empty(d))
        self.hand_scenario_kind = nn.Parameter(torch.empty(d))
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=self.config.heads,
            dim_feedforward=d * config.scenario_ffn_multiplier,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.scenario_encoder = nn.TransformerEncoder(
            layer, num_layers=config.scenario_layers, norm=nn.LayerNorm(d)
        )
        self.scenario_input_norm = nn.LayerNorm(d)
        self.early_goal_norm = nn.LayerNorm(d)
        nn.init.normal_(self.zone_scenario_kind, std=0.02)
        nn.init.normal_(self.event_scenario_kind, std=0.02)
        nn.init.normal_(self.hand_scenario_kind, std=0.02)

    def _scenario_tokens(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, slice]:
        values = batch["zone_inventory_num"] / self.zone_scale
        owners = self.zone_owner(
            torch.arange(2, device=values.device).view(1, 2).expand(values.size(0), -1)
        )
        zones = self.zone_reader(values) + owners + self.zone_scenario_kind
        resources = self._resource_tokens(batch)
        event = self._event_context(batch).unsqueeze(1) + self.event_scenario_kind
        hand = self._hand_context(batch).unsqueeze(1) + self.hand_scenario_kind
        tokens = torch.cat((zones, resources, event, hand), dim=1)
        mask = torch.cat(
            (
                torch.ones((tokens.size(0), 2), dtype=torch.bool, device=tokens.device),
                batch["registered_mask"],
                torch.ones((tokens.size(0), 2), dtype=torch.bool, device=tokens.device),
            ),
            dim=1,
        )
        encoded = self.scenario_encoder(
            self.scenario_input_norm(tokens), src_key_padding_mask=~mask
        )
        return encoded, mask, slice(2, 2 + resources.size(1))

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        entity = batch["entity_cat"]
        # R3 deliberately removes AC's zero gate: official card capability is
        # present from the first update in both entity and option identity paths.
        entity_repr = (
            self.card(entity[..., 0])
            + self.owner(entity[..., 1])
            + self.zone(entity[..., 2])
            + self.slot(entity[..., 3])
            + self.kind(entity[..., 4])
            + self.status(entity[..., 5])
            + self.parent_slot(entity[..., 6].clamp_max(192))
            + self.entity_num(batch["entity_num"])
            + self._card_features(entity[..., 0])
        )
        entity_repr = self.entity_norm(entity_repr)

        global_values = batch["global_cat"]
        global_repr = self.global_norm(
            self.select_type(global_values[:, 0])
            + self.select_context(global_values[:, 1])
            + self.first_player(global_values[:, 2])
            + self.flags(global_values[:, 3])
            + self.global_num(batch["global_num"])
        )
        cls = self.cls.expand(entity.size(0), -1, -1) + global_repr.unsqueeze(1)
        scenario, scenario_mask, resource_slice = self._scenario_tokens(batch)
        sequence = torch.cat((cls, entity_repr, scenario), dim=1)
        padding = torch.cat(
            (
                torch.zeros((entity.size(0), 1), dtype=torch.bool, device=entity.device),
                ~batch["entity_mask"],
                ~scenario_mask,
            ),
            dim=1,
        )
        encoded = self.encoder(sequence, src_key_padding_mask=padding)
        entity_end = 1 + entity.size(1)
        state_repr = encoded[:, 0]
        encoded_entities = encoded[:, 1:entity_end]
        encoded_scenario = encoded[:, entity_end:]

        option = batch["option_cat"]
        option_repr = (
            self.option_type(option[..., 0])
            + self.area(option[..., 1])
            + self.area(option[..., 2])
            + self.option_owner(option[..., 3])
            + self.card(option[..., 4])
            + self.card(option[..., 5])
            + self.number(option[..., 6])
            + self.slot(option[..., 7])
            + self.slot(option[..., 8])
            + self.option_position(option[..., 11])
            + self._gather_entities(encoded_entities, option[..., 9])
            + self._gather_entities(encoded_entities, option[..., 10])
            + self._card_features(option[..., 4])
            + self._card_features(option[..., 5])
        )
        option_repr = self.option_norm(option_repr)
        attended, _ = self.option_to_state(
            option_repr,
            encoded,
            encoded,
            key_padding_mask=padding,
            need_weights=False,
        )
        option_repr = self.option_state_norm(option_repr + attended)
        option_repr = self.option_ff_norm(option_repr + self.option_ff(option_repr))

        resources = encoded_scenario[:, resource_slice]
        goals = self.goal_qkv(state_repr, resources, batch["registered_mask"])
        goal_context, _ = self.option_goal_attention(
            option_repr, goals, goals, need_weights=False
        )
        return state_repr, self.early_goal_norm(option_repr + goal_context)


def parameter_count(model: R3EarlyScenarioFusionPolicy) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = ["R3EarlyScenarioFusionPolicy", "R3ModelConfig", "parameter_count"]
