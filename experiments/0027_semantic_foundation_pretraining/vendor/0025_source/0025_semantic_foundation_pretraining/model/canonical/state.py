"""State memory encoder for global, card-instance, ledger, and event tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ...features.canonical.schema import (
    CARD_CAT_VOCABS,
    EVENT_CAT_VOCABS,
    GLOBAL_CAT_VOCABS,
    RESOURCE_CAT_VOCABS,
    WIDTHS,
)
from .config import CanonicalModelConfig
from .prototypes import PrototypeBank
from .typed import CategoricalFields, NumericFields, gather_relation


@dataclass(frozen=True)
class StateMemory:
    tokens: Tensor
    mask: Tensor
    summary: Tensor
    cards: Tensor


class CanonicalStateEncoder(nn.Module):
    def __init__(self, config: CanonicalModelConfig, prototypes: PrototypeBank):
        super().__init__()
        d = config.d_model
        self.prototypes = prototypes
        self.global_cat = CategoricalFields(GLOBAL_CAT_VOCABS, d)
        self.global_num = NumericFields(WIDTHS.global_num, d)
        self.card_cat = CategoricalFields(CARD_CAT_VOCABS, d)
        self.card_num = NumericFields(WIDTHS.card_num, d)
        self.card_parent = nn.Linear(d, d, bias=False)
        self.resource_cat = CategoricalFields(RESOURCE_CAT_VOCABS, d)
        self.resource_num = NumericFields(WIDTHS.resource_num, d)
        self.event_cat = CategoricalFields(EVENT_CAT_VOCABS, d)
        self.event_num = NumericFields(WIDTHS.event_num, d)
        self.segment = nn.Embedding(5, d, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            d,
            config.heads,
            d * config.ffn_multiplier,
            config.dropout,
            "gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, config.state_layers, norm=nn.LayerNorm(d)
        )
        self.summary_query = nn.Parameter(torch.zeros(1, 1, d))
        self.summary_attention = nn.MultiheadAttention(
            d, config.heads, dropout=config.dropout, batch_first=True
        )
        self.summary_norm = nn.LayerNorm(d)

    def forward(self, batch: dict[str, Tensor]) -> StateMemory:
        global_token = self.global_cat(batch["global_cat"]) + self.global_num(batch["global_num"])
        global_token = global_token.unsqueeze(1) + self.segment.weight[1]
        global_mask = torch.ones(
            (global_token.size(0), 1), dtype=torch.bool, device=global_token.device
        )

        cards = (
            self.card_cat(batch["card_cat"])
            + self.card_num(batch["card_num"])
            + self.prototypes.card(batch["card_cat"][..., 0])
        )
        cards = cards + self.card_parent(gather_relation(cards, batch["card_parent"]))
        cards = cards + self.segment.weight[2]

        resources = (
            self.resource_cat(batch["resource_cat"])
            + self.resource_num(batch["resource_num"])
            + self.prototypes.card(batch["resource_cat"][..., 0])
            + self.segment.weight[3]
        )
        events = (
            self.event_cat(batch["event_cat"])
            + self.event_num(batch["event_num"])
            + self.prototypes.card(batch["event_cat"][..., 2])
            + self.segment.weight[4]
        )
        tokens = torch.cat((global_token, cards, resources, events), dim=1)
        mask = torch.cat(
            (global_mask, batch["card_mask"], batch["resource_mask"], batch["event_mask"]),
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=~mask)
        encoded = encoded * mask.unsqueeze(-1)
        card_start = 1
        card_end = card_start + cards.size(1)
        encoded_cards = encoded[:, card_start:card_end] * batch["card_mask"].unsqueeze(-1)
        query = self.summary_query.expand(encoded.size(0), -1, -1)
        summary, _ = self.summary_attention(
            query, encoded, encoded, key_padding_mask=~mask, need_weights=False
        )
        return StateMemory(encoded, mask, self.summary_norm(summary.squeeze(1)), encoded_cards)


__all__ = ["CanonicalStateEncoder", "StateMemory"]
