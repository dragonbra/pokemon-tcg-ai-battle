"""Legal-option encoder with explicit state and prototype relations."""

from __future__ import annotations

from torch import Tensor, nn

from ...features.canonical.schema import (
    EFFECT_ROLE_VOCAB,
    OPTION_CAT_VOCABS,
    SKILL_ROLE_VOCAB,
    WIDTHS,
)
from .config import CanonicalModelConfig
from .prototypes import PrototypeBank
from .state import StateMemory
from .typed import (
    CategoricalFields,
    NumericFields,
    aggregate_relations,
    gather_relation,
)


class CanonicalOptionEncoder(nn.Module):
    def __init__(self, config: CanonicalModelConfig, prototypes: PrototypeBank):
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
        self.norm = nn.LayerNorm(d)
        layer = nn.TransformerDecoderLayer(
            d,
            config.heads,
            d * config.ffn_multiplier,
            config.dropout,
            "gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer, config.option_layers, norm=nn.LayerNorm(d)
        )

    def forward(self, batch: dict[str, Tensor], state: StateMemory) -> Tensor:
        option_cat = batch["option_cat"]
        options = self.categorical(option_cat) + self.numeric(batch["option_num"])
        options = options + self.field_state(batch["option_state"]).sum(dim=-2)
        for column in (5, 6, 12, 13):
            options = options + self.prototypes.card(option_cat[..., column])
        options = options + self.prototypes.attack(option_cat[..., 7])
        options = options + self.source_relation(
            gather_relation(state.cards, batch["option_source"])
        )
        options = options + self.target_relation(
            gather_relation(state.cards, batch["option_target"])
        )

        skill_values = self.prototypes.skill(batch["option_skill_id"])
        skill_values = skill_values + self.skill_role(batch["option_skill_role"])
        skill_context = aggregate_relations(
            skill_values,
            batch["option_skill_parent"],
            batch["option_skill_mask"],
            options.size(1),
        )
        effect_values = self.prototypes.effect(batch["option_effect_id"])
        effect_values = effect_values + self.effect_role(batch["option_effect_role"])
        effect_context = aggregate_relations(
            effect_values,
            batch["option_effect_parent"],
            batch["option_effect_mask"],
            options.size(1),
        )
        options = self.norm(
            options
            + self.skill_relation(skill_context)
            + self.effect_relation(effect_context)
        )
        encoded = self.decoder(
            options,
            state.tokens,
            tgt_key_padding_mask=~batch["option_mask"],
            memory_key_padding_mask=~state.mask,
        )
        return encoded * batch["option_mask"].unsqueeze(-1)


__all__ = ["CanonicalOptionEncoder"]
