"""Explicit M0-M5 Semantic Goal Policy variants."""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ModelConfig:
    d_model: int = 384
    heads: int = 8
    state_layers: int = 6
    ffn_dim: int = 1536
    option_layers: int = 2
    dropout: float = 0.1
    max_card_id: int = 2048
    categorical_vocab: int = 4096
    relation_types: int = 16


class PolicyEncoding(NamedTuple):
    state: Tensor
    options: Tensor
    goals: Tensor
    value: Tensor


class GoalQKV(nn.Module):
    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        self.roles = nn.Parameter(torch.empty(4, d_model))
        self.query = nn.Linear(2 * d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.attention = nn.MultiheadAttention(d_model, heads, batch_first=True)
        nn.init.normal_(self.roles, std=0.02)

    def forward(self, state: Tensor, resources: Tensor, mask: Tensor) -> Tensor:
        roles = self.roles.unsqueeze(0).expand(state.size(0), -1, -1)
        queries = self.query(torch.cat((state.unsqueeze(1).expand_as(roles), roles), dim=-1))
        result, _ = self.attention(queries, self.key(resources), self.value(resources), key_padding_mask=~mask, need_weights=False)
        return result


class SemanticGoalPolicy(nn.Module):
    goal_roles = ("setup_board", "attack_prize", "resource_recovery", "tempo_survival")

    def __init__(self, variant: str, config: ModelConfig | None = None) -> None:
        super().__init__()
        if variant not in {f"M{index}" for index in range(6)}:
            raise ValueError(f"unknown model variant {variant}")
        self.variant = variant
        self.level = int(variant[1])
        self.config = config or ModelConfig()
        c, d = self.config, self.config.d_model
        self.categorical = nn.Embedding(c.categorical_vocab, d, padding_idx=0)
        self.card_identity = nn.Embedding(c.max_card_id + 3, d, padding_idx=0)
        self.semantic_projection = nn.Sequential(nn.Linear(64, d), nn.GELU(), nn.Linear(d, d))
        self.state_numeric = nn.Sequential(nn.Linear(16, d), nn.GELU(), nn.Linear(d, d))
        self.entity_numeric = nn.Sequential(nn.Linear(12, d), nn.GELU(), nn.Linear(d, d))
        self.ledger_numeric = nn.Sequential(nn.Linear(12, d), nn.GELU(), nn.Linear(d, d))
        self.event_numeric = nn.Sequential(nn.Linear(8, d), nn.GELU(), nn.Linear(d, d))
        self.option_numeric = nn.Sequential(nn.Linear(8, d), nn.GELU(), nn.Linear(d, d))
        self.state_token = nn.Parameter(torch.zeros(1, 1, d))
        self.deck_kind = nn.Parameter(torch.zeros(d))
        self.ledger_kind = nn.Parameter(torch.zeros(d))
        self.event_kind = nn.Parameter(torch.zeros(d))
        self.goal_kind = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerEncoderLayer(d, c.heads, c.ffn_dim, c.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.state_encoder = nn.TransformerEncoder(layer, c.state_layers, norm=nn.LayerNorm(d))
        self.goal_qkv = GoalQKV(d, c.heads)
        self.relation_bias = nn.Embedding(c.relation_types, c.heads)
        self.option_type = nn.Embedding(256, d)
        self.option_attention = nn.ModuleList(nn.MultiheadAttention(d, c.heads, dropout=c.dropout, batch_first=True) for _ in range(c.option_layers))
        self.option_norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(c.option_layers))
        self.option_ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.value_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.decoder_init = nn.Linear(d, d)
        self.decoder_cell = nn.GRUCell(d, d)
        self.pointer_query = nn.Linear(d, d, bias=False)
        self.pointer_key = nn.Linear(d, d, bias=False)
        self.option_bias = nn.Linear(d, 1)
        self.stop_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.normal_(self.state_token, std=0.02)

    def _cats(self, value: Tensor) -> Tensor:
        return self.categorical(value.clamp(0, self.config.categorical_vocab - 1)).sum(dim=-2)

    def _deck(self, batch: dict[str, Tensor]) -> Tensor:
        cards = batch["deck_card_ids"].clamp(0, self.config.max_card_id + 2)
        return self.card_identity(cards) + self.deck_kind + batch["deck_multiplicity"].unsqueeze(-1) / 4.0

    def _options(self, batch: dict[str, Tensor]) -> Tensor:
        categorical = batch["options_cat"]
        type_ids = categorical[..., 0].clamp(0, 255)
        result = self.option_type(type_ids) + self._cats(categorical[..., 1:])
        if self.level >= 1:
            result = result + self.option_numeric(batch["options_num"])
        return result

    def encode(self, batch: dict[str, Tensor]) -> PolicyEncoding:
        batch_size = batch["state_num"].size(0)
        state = self.state_token.expand(batch_size, -1, -1) + self._cats(batch["state_cat"]).unsqueeze(1) + self.state_numeric(batch["state_num"]).unsqueeze(1)
        entities = self._cats(batch["entities_cat"])
        if self.level >= 1:
            entities = entities + self.entity_numeric(batch["entities_num"])
        if self.level >= 5:
            relations = batch["relations"].clamp(0, self.config.relation_types - 1)
            relation_summary = self.relation_bias(relations).mean(dim=(-2, -1))
            relation_projection = relation_summary.mean(-1, keepdim=True).unsqueeze(-1)
            entities = entities + relation_projection
        pieces, masks = [state, entities], [torch.ones(batch_size, 1, dtype=torch.bool, device=state.device), batch["entity_mask"]]
        deck = self._deck(batch)
        deck_mask = batch["deck_mask"]
        if self.level == 2:
            mean = (deck * deck_mask.unsqueeze(-1)).sum(1) / deck_mask.sum(1, keepdim=True).clamp_min(1)
            pieces[0] = pieces[0] + mean.unsqueeze(1)
        goals = torch.zeros(batch_size, 4, self.config.d_model, device=state.device, dtype=state.dtype)
        if self.level >= 3:
            goals = self.goal_qkv(pieces[0][:, 0], deck, deck_mask)
            if self.level >= 4:
                ledger = self._cats(batch["ledger_cat"]) + self.ledger_numeric(batch["ledger_num"]) + self.ledger_kind
                resources = torch.cat((deck, ledger), dim=1)
                resource_mask = torch.cat((deck_mask, batch["ledger_mask"]), dim=1)
                goals = self.goal_qkv(pieces[0][:, 0], resources, resource_mask)
                pieces.append(ledger); masks.append(batch["ledger_mask"])
            pieces.append(goals + self.goal_kind); masks.append(torch.ones(batch_size, 4, dtype=torch.bool, device=state.device))
        if self.level >= 5:
            events = self._cats(batch["events_cat"]) + self.event_numeric(batch["events_num"]) + self.event_kind
            pieces.append(events); masks.append(batch["event_mask"])
        sequence = torch.cat(pieces, dim=1)
        mask = torch.cat(masks, dim=1)
        encoded = self.state_encoder(sequence, src_key_padding_mask=~mask)
        contextual_state = encoded[:, 0]
        options = self._options(batch)
        for attention, norm in zip(self.option_attention, self.option_norms):
            attended, _ = attention(options, encoded, encoded, key_padding_mask=~mask, need_weights=False)
            options = norm(options + attended)
        options = options + self.option_ff(options)
        value = self.value_head(contextual_state).squeeze(-1)
        return PolicyEncoding(contextual_state, options, goals, value)


__all__ = ["GoalQKV", "ModelConfig", "PolicyEncoding", "SemanticGoalPolicy"]
