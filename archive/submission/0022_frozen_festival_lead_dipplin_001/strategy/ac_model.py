"""All-feature causal policy built around the faithful 0010 board trunk."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from .card_features import CARD_FEATURE_WIDTH, build_card_feature_table
from .model import Faithful0010PointerPolicy, IDOnlyConfig


@dataclass(frozen=True)
class ACModelConfig:
    """Lightweight auxiliary readers; the 0010 trunk remains unchanged."""

    base: IDOnlyConfig = field(default_factory=IDOnlyConfig)
    event_layers: int = 1
    auxiliary_ffn_multiplier: int = 2
    goal_roles: int = 4

    def validate(self) -> None:
        self.base.validate()
        if self.event_layers < 1 or self.auxiliary_ffn_multiplier < 2:
            raise ValueError("invalid AC auxiliary depth or width")
        if self.goal_roles != 4:
            raise ValueError("0020 AC requires exactly four audited goal roles")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class BoardConditionedGoalQKV(nn.Module):
    """Retrieve role-specific evidence from the registered-card resource memory."""

    def __init__(self, d_model: int, heads: int, role_count: int) -> None:
        super().__init__()
        self.roles = nn.Parameter(torch.empty(role_count, d_model))
        self.query = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.attention = nn.MultiheadAttention(
            d_model, heads, dropout=0.0, batch_first=True
        )
        nn.init.normal_(self.roles, std=0.02)

    def forward(self, board_state: Tensor, resources: Tensor, mask: Tensor) -> Tensor:
        roles = self.roles.unsqueeze(0).expand(board_state.size(0), -1, -1)
        board = board_state.unsqueeze(1).expand(-1, roles.size(1), -1)
        queries = self.query(torch.cat((board, roles), dim=-1))
        goals, _ = self.attention(
            queries,
            resources,
            resources,
            key_padding_mask=~mask,
            need_weights=False,
        )
        return goals


class AllFeatureCausalPolicy(Faithful0010PointerPolicy):
    """Faithful board encoder plus actor-visible semantic and causal readers."""

    goal_role_names = (
        "setup_board",
        "attack_prize",
        "resource_recovery",
        "tempo_survival",
    )

    def __init__(self, config: ACModelConfig, *, ontology_path: Path | str) -> None:
        config.validate()
        super().__init__(config.base)
        self.ac_config = config
        d = config.base.d_model
        heads = config.base.heads
        auxiliary_ffn = d * config.auxiliary_ffn_multiplier

        self.register_buffer(
            "card_feature_table",
            build_card_feature_table(
                ontology_path, max_card_id=config.base.max_card_id
            ),
            persistent=True,
        )
        self.card_semantic = nn.Sequential(
            nn.Linear(CARD_FEATURE_WIDTH, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

        zone_scale = (
            60, 60, 6, 1, 5, 60, 20, 6, 2, 2000,
            5, 6, 60, 60, 2, 30,
        )
        self.register_buffer(
            "zone_scale",
            torch.tensor(zone_scale, dtype=torch.float32).view(1, 1, -1),
            persistent=True,
        )
        self.zone_owner = nn.Embedding(2, d)
        self.zone_reader = nn.Sequential(
            nn.Linear(16, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d)
        )
        self.zone_pool = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.LayerNorm(d)
        )

        self.ledger_state = nn.Embedding(8, d)
        self.deck_order_state = nn.Embedding(2, d)
        ledger_scale = (
            4, 1, 5, 4, 60, 2, 2, 60, 60, 60, 6, 6, 6, 128, 128,
        )
        self.register_buffer(
            "ledger_scale",
            torch.tensor(ledger_scale, dtype=torch.float32).view(1, 1, -1),
            persistent=True,
        )
        self.ledger_numeric = nn.Sequential(
            nn.Linear(15, d), nn.GELU(), nn.Linear(d, d)
        )
        self.multiplicity = nn.Linear(1, d)
        self.resource_kind = nn.Parameter(torch.zeros(d))
        self.resource_norm = nn.LayerNorm(d)

        self.event_type = nn.Embedding(256, d)
        self.event_actor = nn.Embedding(3, d)
        self.event_area = nn.Embedding(64, d)
        self.event_flag = nn.Embedding(2, d)
        self.event_numeric = nn.Sequential(
            nn.Linear(4, d), nn.GELU(), nn.Linear(d, d)
        )
        self.register_buffer(
            "event_scale",
            torch.tensor((64, 1000, 1000, 1), dtype=torch.float32).view(1, 1, -1),
            persistent=True,
        )
        self.event_kind = nn.Parameter(torch.zeros(d))
        self.event_summary = nn.Parameter(torch.zeros(1, 1, d))
        event_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=heads,
            dim_feedforward=auxiliary_ffn,
            dropout=config.base.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.event_encoder = nn.TransformerEncoder(
            event_layer,
            num_layers=config.event_layers,
            norm=nn.LayerNorm(d),
        )

        self.known_hand_kind = nn.Parameter(torch.zeros(d))
        self.unknown_hand_reader = nn.Sequential(
            nn.Linear(1, d), nn.GELU(), nn.Linear(d, d)
        )
        self.hand_norm = nn.LayerNorm(d)

        self.goal_qkv = BoardConditionedGoalQKV(d, heads, config.goal_roles)
        self.scenario_norm = nn.LayerNorm(d)
        self.option_goal_attention = nn.MultiheadAttention(
            d, heads, dropout=config.base.dropout, batch_first=True
        )
        self.option_aux_norm = nn.LayerNorm(d)

        # Zero gates make an AC instance exactly reproduce its copied 0010
        # trunk before the new information paths learn to open.
        self.entity_semantic_gate = nn.Parameter(torch.zeros(d))
        self.global_zone_gate = nn.Parameter(torch.zeros(d))
        self.state_aux_gate = nn.Parameter(torch.zeros(d))
        self.option_semantic_gate = nn.Parameter(torch.zeros(d))
        self.option_aux_gate = nn.Parameter(torch.zeros(d))

        nn.init.normal_(self.resource_kind, std=0.02)
        nn.init.normal_(self.event_kind, std=0.02)
        nn.init.normal_(self.event_summary, std=0.02)
        nn.init.normal_(self.known_hand_kind, std=0.02)

    def _card_features(self, card_ids: Tensor) -> Tensor:
        ids = card_ids.clamp(0, self.config.max_card_id)
        return self.card_semantic(self.card_feature_table[ids])

    def _zone_context(self, batch: dict[str, Tensor]) -> Tensor:
        values = batch["zone_inventory_num"] / self.zone_scale
        owners = self.zone_owner(
            torch.arange(2, device=values.device).view(1, 2).expand(values.size(0), -1)
        )
        zones = self.zone_reader(values) + owners
        return self.zone_pool(zones.flatten(1))

    def _resource_tokens(self, batch: dict[str, Tensor]) -> Tensor:
        ids = batch["registered_card_ids"].clamp(0, self.config.max_card_id)
        ledger_cat = batch["ledger_cat"]
        token = (
            self.card(ids)
            + self._card_features(ids)
            + self.ledger_state(ledger_cat[..., 1].clamp(0, 7))
            + self.ledger_state(ledger_cat[..., 2].clamp(0, 7))
            + self.deck_order_state(ledger_cat[..., 3].clamp(0, 1))
            + self.ledger_numeric(batch["ledger_num"] / self.ledger_scale)
            + self.multiplicity(batch["registered_multiplicity"].unsqueeze(-1) / 4.0)
            + self.resource_kind
        )
        return self.resource_norm(token)

    def _event_context(self, batch: dict[str, Tensor]) -> Tensor:
        cats = batch["event_cat"]
        ids = cats[..., 2].clamp(0, self.config.max_card_id)
        events = (
            self.event_type(cats[..., 0].clamp(0, 255))
            + self.event_actor(cats[..., 1].clamp(0, 2))
            + self.card(ids)
            + self._card_features(ids)
            + self.event_area(cats[..., 3].clamp(0, 63))
            + self.event_area(cats[..., 4].clamp(0, 63))
            + self.event_flag(cats[..., 5].clamp(0, 1))
            + self.event_flag(cats[..., 6].clamp(0, 1))
            + self.event_flag(cats[..., 7].clamp(0, 1))
            + self.event_numeric(batch["event_num"] / self.event_scale)
            + self.event_kind
        )
        summary = self.event_summary.expand(events.size(0), -1, -1)
        sequence = torch.cat((summary, events), dim=1)
        padding = torch.cat(
            (
                torch.zeros(
                    events.size(0), 1, dtype=torch.bool, device=events.device
                ),
                ~batch["event_mask"],
            ),
            dim=1,
        )
        encoded = self.event_encoder(sequence, src_key_padding_mask=padding)
        return encoded[:, 0]

    def _hand_context(self, batch: dict[str, Tensor]) -> Tensor:
        ids = batch["known_opponent_hand_card_ids"].clamp(
            0, self.config.max_card_id
        )
        known = self.card(ids) + self._card_features(ids) + self.known_hand_kind
        mask = batch["known_opponent_hand_mask"]
        known_mean = (known * mask.unsqueeze(-1)).sum(1) / mask.sum(
            1, keepdim=True
        ).clamp_min(1)
        unknown = self.unknown_hand_reader(
            (batch["unknown_opponent_hand_count"] / 60.0).unsqueeze(-1)
        )
        return self.hand_norm(known_mean + unknown)

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        entity = batch["entity_cat"]
        entity_semantic = self._card_features(entity[..., 0])
        entity_repr = (
            self.card(entity[..., 0])
            + self.owner(entity[..., 1])
            + self.zone(entity[..., 2])
            + self.slot(entity[..., 3])
            + self.kind(entity[..., 4])
            + self.status(entity[..., 5])
            + self.parent_slot(entity[..., 6].clamp_max(192))
            + self.entity_num(batch["entity_num"])
            + torch.tanh(self.entity_semantic_gate) * entity_semantic
        )
        entity_repr = self.entity_norm(entity_repr)

        zone_context = self._zone_context(batch)
        global_values = batch["global_cat"]
        global_repr = self.global_norm(
            self.select_type(global_values[:, 0])
            + self.select_context(global_values[:, 1])
            + self.first_player(global_values[:, 2])
            + self.flags(global_values[:, 3])
            + self.global_num(batch["global_num"])
            + torch.tanh(self.global_zone_gate) * zone_context
        )
        cls = self.cls.expand(entity.size(0), -1, -1) + global_repr.unsqueeze(1)
        sequence = torch.cat((cls, entity_repr), dim=1)
        padding = torch.cat(
            (
                torch.zeros(
                    entity.size(0), 1, dtype=torch.bool, device=entity.device
                ),
                ~batch["entity_mask"],
            ),
            dim=1,
        )
        encoded = self.encoder(sequence, src_key_padding_mask=padding)
        state_repr, encoded_entities = encoded[:, 0], encoded[:, 1:]

        resources = self._resource_tokens(batch)
        goals = self.goal_qkv(state_repr, resources, batch["registered_mask"])
        scenario = self.scenario_norm(
            zone_context
            + self._event_context(batch)
            + self._hand_context(batch)
            + goals.mean(1)
        )
        state_repr = state_repr + torch.tanh(self.state_aux_gate) * scenario

        option = batch["option_cat"]
        option_semantic = self._card_features(option[..., 4]) + self._card_features(
            option[..., 5]
        )
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
            + torch.tanh(self.option_semantic_gate) * option_semantic
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
        goal_context, _ = self.option_goal_attention(
            option_repr, goals, goals, need_weights=False
        )
        option_aux = self.option_aux_norm(goal_context + scenario.unsqueeze(1))
        option_repr = option_repr + torch.tanh(self.option_aux_gate) * option_aux
        return state_repr, option_repr


def parameter_count(model: AllFeatureCausalPolicy) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = [
    "ACModelConfig",
    "AllFeatureCausalPolicy",
    "BoardConditionedGoalQKV",
    "parameter_count",
]
