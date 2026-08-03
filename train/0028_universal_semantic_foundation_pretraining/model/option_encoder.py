"""Bind legal options to state instances and official prototypes, then retrieve state memory."""

from __future__ import annotations

from torch import Tensor, nn

from ..contracts.batch import DecisionBatch
from ..contracts.fields import (
    EFFECT_ROLE_VOCAB,
    OPTION_CAT_VOCABS,
    SKILL_ROLE_VOCAB,
    WIDTHS,
)
from .config import ModelConfig
from .prototype_encoder import OfficialPrototypeEncoder, PrototypeEmbeddings
from .state_encoder import EncodedState
from .typed_fields import CategoricalFields, NumericFields, mean_pool_by_parent


class OptionEncoder(nn.Module):
    """Every legal option asks its own question of the full state memory."""

    def __init__(self, config: ModelConfig, prototypes: OfficialPrototypeEncoder):
        super().__init__()
        d = config.d_model
        self.prototypes = prototypes
        self.categorical = CategoricalFields(OPTION_CAT_VOCABS, d)
        self.numeric = NumericFields(WIDTHS.option_num, d)
        self.field_state = nn.Embedding(4, d, padding_idx=0)
        self.skill_role = nn.Embedding(SKILL_ROLE_VOCAB, d, padding_idx=0)
        self.effect_role = nn.Embedding(EFFECT_ROLE_VOCAB, d, padding_idx=0)
        self.source_relation = nn.Linear(d, d, bias=False)
        self.target_relation = nn.Linear(d, d, bias=False)
        self.skill_relation = nn.Linear(d, d, bias=False)
        self.effect_relation = nn.Linear(d, d, bias=False)
        self.input_norm = nn.LayerNorm(d)

        layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=config.heads,
            dim_feedforward=d * config.ffn_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_attention_transformer = nn.TransformerDecoder(
            layer,
            num_layers=config.option_layers,
            norm=nn.LayerNorm(d),
        )

    def forward(
        self,
        batch: DecisionBatch,
        state: EncodedState,
        prototype_memory: PrototypeEmbeddings | None = None,
    ) -> Tensor:
        prototype_memory = prototype_memory or self.prototypes.encode_all()
        option_cat = batch.option_cat
        options = self.categorical(option_cat) + self.numeric(batch.option_num)
        options = options + self.field_state(batch.option_state).sum(dim=-2)

        for card_column in (5, 6, 12, 13):
            options = options + prototype_memory.card(option_cat[..., card_column])
        options = options + prototype_memory.attack(option_cat[..., 7])
        options = options + self.source_relation(state.gather_cards(batch.option_source))
        options = options + self.target_relation(state.gather_cards(batch.option_target))

        skill_tokens = (
            prototype_memory.skill(batch.option_skill_id)
            + self.skill_role(batch.option_skill_role)
        )
        skill_context = mean_pool_by_parent(
            skill_tokens,
            batch.option_skill_parent,
            batch.option_skill_mask,
            batch.option_count,
        )
        effect_tokens = (
            prototype_memory.effect(batch.option_effect_id)
            + self.effect_role(batch.option_effect_role)
        )
        effect_context = mean_pool_by_parent(
            effect_tokens,
            batch.option_effect_parent,
            batch.option_effect_mask,
            batch.option_count,
        )
        options = self.input_norm(
            options
            + self.skill_relation(skill_context)
            + self.effect_relation(effect_context)
        )

        encoded = self.cross_attention_transformer(
            tgt=options,
            memory=state.tokens,
            tgt_key_padding_mask=~batch.option_mask,
            memory_key_padding_mask=~state.mask,
        )
        return encoded * batch.option_mask.unsqueeze(-1)


__all__ = ["OptionEncoder"]
