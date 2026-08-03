"""Encode global, card-instance, exact-deck ledger, and event tokens as full memory."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..contracts.batch import DecisionBatch
from ..contracts.fields import (
    CARD_CAT_VOCABS,
    EVENT_CAT_VOCABS,
    GLOBAL_CAT_VOCABS,
    RESOURCE_CAT_VOCABS,
    WIDTHS,
)
from .config import ModelConfig
from .prototype_encoder import OfficialPrototypeEncoder, PrototypeEmbeddings
from .typed_fields import CategoricalFields, NumericFields, gather_one_based, sum_children_by_parent


@dataclass(frozen=True, slots=True)
class EncodedState:
    tokens: Tensor
    mask: Tensor
    summary: Tensor
    cards: Tensor

    def gather_cards(self, one_based_index: Tensor) -> Tensor:
        return gather_one_based(self.cards, one_based_index)


class StateEncoder(nn.Module):
    """Preserve variable-length state memory for option-specific retrieval."""

    def __init__(self, config: ModelConfig, prototypes: OfficialPrototypeEncoder):
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
            d_model=d,
            nhead=config.heads,
            dim_feedforward=d * config.ffn_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.state_layers,
            norm=nn.LayerNorm(d),
        )
        self.summary_query = nn.Parameter(torch.zeros(1, 1, d))
        self.summary_attention = nn.MultiheadAttention(
            d,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.summary_norm = nn.LayerNorm(d)

    def forward(
        self,
        batch: DecisionBatch,
        prototype_memory: PrototypeEmbeddings | None = None,
    ) -> EncodedState:
        prototype_memory = prototype_memory or self.prototypes.encode_all()
        global_token = self.global_cat(batch.global_cat) + self.global_num(batch.global_num, batch.global_state)
        global_token = global_token.unsqueeze(1) + self.segment.weight[1]
        global_mask = torch.ones(
            (batch.batch_size, 1),
            dtype=torch.bool,
            device=global_token.device,
        )

        cards = (
            self.card_cat(batch.card_cat)
            + self.card_num(batch.card_num, batch.card_state)
            + prototype_memory.card(batch.card_cat[..., 0])
        )
        parent_context = gather_one_based(cards, batch.card_parent)
        child_context = sum_children_by_parent(cards, batch.card_parent, batch.card_mask)
        cards = cards + self.card_parent(parent_context + child_context)
        cards = cards + self.segment.weight[2]

        resources = (
            self.resource_cat(batch.resource_cat)
            + self.resource_num(batch.resource_num, batch.resource_state)
            + prototype_memory.card(batch.resource_cat[..., 0])
            + self.segment.weight[3]
        )
        events = (
            self.event_cat(batch.event_cat)
            + self.event_num(batch.event_num, batch.event_state)
            + prototype_memory.card(batch.event_cat[..., 2])
            + self.segment.weight[4]
        )
        events = events + gather_one_based(cards, batch.event_source) + gather_one_based(cards, batch.event_target)

        tokens = torch.cat((global_token, cards, resources, events), dim=1)
        mask = torch.cat(
            (global_mask, batch.card_mask, batch.resource_mask, batch.event_mask),
            dim=1,
        )
        encoded = self.transformer(tokens, src_key_padding_mask=~mask)
        encoded = encoded * mask.unsqueeze(-1)

        card_start = 1
        card_end = card_start + cards.shape[1]
        encoded_cards = encoded[:, card_start:card_end] * batch.card_mask.unsqueeze(-1)

        query = self.summary_query.expand(batch.batch_size, -1, -1)
        summary, _ = self.summary_attention(
            query,
            encoded,
            encoded,
            key_padding_mask=~mask,
            need_weights=False,
        )
        return EncodedState(
            tokens=encoded,
            mask=mask,
            summary=self.summary_norm(summary.squeeze(1)),
            cards=encoded_cards,
        )


__all__ = ["EncodedState", "StateEncoder"]
