"""Bind legal options to state instances and official prototypes, then retrieve state memory."""

from __future__ import annotations

from torch import Tensor, nn

from ..contracts.batch import DecisionBatch
from ..contracts.fields import (
    OPTION_CAT_VOCABS,
    WIDTHS,
)
from .config import ModelConfig
from .prototype_encoder import OfficialPrototypeEncoder, PrototypeEmbeddings
from .state_encoder import EncodedState
from .typed_fields import CategoricalFields, NumericFields


class OptionEncoder(nn.Module):
    """Every legal option asks its own question of the full state memory."""

    def __init__(self, config: ModelConfig, prototypes: OfficialPrototypeEncoder):
        super().__init__()
        d = config.d_model
        self.prototypes = prototypes
        self.categorical = CategoricalFields(
            OPTION_CAT_VOCABS,
            d,
            skip_indices=(5, 6, 7, 9, 10),
        )
        self.numeric = NumericFields(WIDTHS.option_num, d)
        self.card_roles = nn.ModuleList(
            nn.Linear(d, d, bias=False) for _ in range(4)
        )
        self.source_relation = nn.Linear(d, d, bias=False)
        self.target_relation = nn.Linear(d, d, bias=False)
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
        options = self.categorical(option_cat) + self.numeric(batch.option_num, batch.option_state)

        for projection, card_column in zip(
            self.card_roles,
            (5, 6, 9, 10),
            strict=True,
        ):
            options = options + projection(
                prototype_memory.card(option_cat[..., card_column])
            )
        options = options + prototype_memory.attack(option_cat[..., 7])
        options = options + self.source_relation(state.gather_cards(batch.option_source))
        options = options + self.target_relation(state.gather_cards(batch.option_target))
        options = self.input_norm(options)

        encoded = self.cross_attention_transformer(
            tgt=options,
            memory=state.tokens,
            tgt_key_padding_mask=~batch.option_mask,
            memory_key_padding_mask=~state.mask,
        )
        return encoded * batch.option_mask.unsqueeze(-1)


__all__ = ["OptionEncoder"]
