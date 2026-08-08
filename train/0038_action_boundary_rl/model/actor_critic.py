"""POD-native actor-critic over the resident CUDA PolicyCodecV1 surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from ..contract import PodNativeBatch
from .action_decoder import ActionDecoder, ActionEvaluation, OrderedActions
from .config import (
    ENTITY_CAT_VOCABS,
    GLOBAL_CAT_VOCABS,
    ModelConfig,
    OPTION_CAT_VOCABS,
)


class FieldEmbeddings(nn.Module):
    def __init__(self, vocabularies: tuple[int, ...], d_model: int) -> None:
        super().__init__()
        self.fields = nn.ModuleList(
            nn.Embedding(size, d_model, padding_idx=0) for size in vocabularies
        )

    def forward(self, values: Tensor) -> Tensor:
        encoded = self.fields[0](values[..., 0])
        for index, embedding in enumerate(self.fields[1:], start=1):
            encoded = encoded + embedding(values[..., index])
        return encoded


@dataclass(frozen=True, slots=True)
class ActorEncoding:
    state_tokens: Tensor
    state_mask: Tensor
    state_summary: Tensor
    entity_tokens: Tensor
    option_tokens: Tensor
    value: Tensor


class PodNativeActorCritic(nn.Module):
    """Field-aware resident actor with an ordered pointer head and value head."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.config.validate()
        d = self.config.d_model

        self.global_cat_embeddings = FieldEmbeddings(GLOBAL_CAT_VOCABS, d)
        self.global_num_projection = nn.Sequential(nn.Linear(16, d), nn.GELU(), nn.Linear(d, d))
        self.global_norm = nn.LayerNorm(d)

        self.entity_cat_embeddings = FieldEmbeddings(ENTITY_CAT_VOCABS, d)
        self.entity_num_projection = nn.Sequential(nn.Linear(10, d), nn.GELU(), nn.Linear(d, d))
        self.entity_parent_relation = nn.Linear(d, d, bias=False)
        self.entity_child_relation = nn.Linear(d, d, bias=False)
        self.entity_norm = nn.LayerNorm(d)
        self.state_segment = nn.Embedding(2, d)
        state_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=self.config.heads,
            dim_feedforward=d * self.config.ffn_multiplier,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.state_encoder = nn.TransformerEncoder(
            state_layer, self.config.state_layers, norm=nn.LayerNorm(d)
        )
        self.summary_norm = nn.LayerNorm(d)

        self.option_cat_embeddings = FieldEmbeddings(OPTION_CAT_VOCABS, d)
        self.option_num_projection = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.option_source_relation = nn.Linear(d, d, bias=False)
        self.option_target_relation = nn.Linear(d, d, bias=False)
        self.option_norm = nn.LayerNorm(d)
        option_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=self.config.heads,
            dim_feedforward=d * self.config.ffn_multiplier,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.option_encoder = nn.TransformerDecoder(
            option_layer, self.config.option_layers, norm=nn.LayerNorm(d)
        )

        self.action_decoder = ActionDecoder(self.config)
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    @staticmethod
    def _gather_entities(entities: Tensor, one_based_index: Tensor) -> Tensor:
        valid = one_based_index.gt(0) & one_based_index.le(entities.shape[1])
        indices = (one_based_index - 1).clamp(min=0, max=entities.shape[1] - 1)
        gathered = entities.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, entities.shape[-1])
        )
        return gathered * valid.unsqueeze(-1)

    @staticmethod
    def _parent_context(tokens: Tensor, parent: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        valid = parent.ge(0) & parent.lt(tokens.shape[1]) & mask
        indices = parent.clamp(min=0, max=tokens.shape[1] - 1)
        gathered = tokens.gather(1, indices.unsqueeze(-1).expand_as(tokens))
        parent_tokens = gathered * valid.unsqueeze(-1)
        children = torch.zeros_like(tokens)
        children.scatter_add_(
            1,
            indices.unsqueeze(-1).expand_as(tokens),
            tokens * valid.unsqueeze(-1),
        )
        return parent_tokens, children

    def validate_batch(
        self, batch: PodNativeBatch | Mapping[str, Tensor]
    ) -> PodNativeBatch:
        return batch if isinstance(batch, PodNativeBatch) else PodNativeBatch.from_mapping(batch)

    def encode(self, batch: PodNativeBatch | Mapping[str, Tensor]) -> ActorEncoding:
        batch = self.validate_batch(batch)
        global_token = self.global_norm(
            self.global_cat_embeddings(batch.global_cat)
            + self.global_num_projection(batch.global_num)
        )
        entities = self.entity_cat_embeddings(batch.entity_cat) + self.entity_num_projection(
            batch.entity_num
        )
        parent, children = self._parent_context(
            entities * batch.entity_mask.unsqueeze(-1), batch.entity_parent, batch.entity_mask
        )
        entities = self.entity_norm(
            entities
            + self.entity_parent_relation(parent)
            + self.entity_child_relation(children)
        )
        entities = entities * batch.entity_mask.unsqueeze(-1)
        state_tokens = torch.cat(
            (
                global_token.unsqueeze(1) + self.state_segment.weight[0],
                entities + self.state_segment.weight[1],
            ),
            dim=1,
        )
        state_mask = torch.cat(
            (
                torch.ones(batch.batch_size, 1, dtype=torch.bool, device=batch.device),
                batch.entity_mask,
            ),
            dim=1,
        )
        state_tokens = self.state_encoder(
            state_tokens, src_key_padding_mask=~state_mask
        ) * state_mask.unsqueeze(-1)
        entity_tokens = state_tokens[:, 1:]
        state_summary = self.summary_norm(state_tokens[:, 0])

        option_cat = batch.option_cat
        options = self.option_cat_embeddings(option_cat) + self.option_num_projection(
            batch.option_num
        )
        options = self.option_norm(
            options
            + self.option_source_relation(
                self._gather_entities(entity_tokens, option_cat[..., 8])
            )
            + self.option_target_relation(
                self._gather_entities(entity_tokens, option_cat[..., 9])
            )
        )
        options = self.option_encoder(
            tgt=options,
            memory=state_tokens,
            tgt_key_padding_mask=~batch.option_mask,
            memory_key_padding_mask=~state_mask,
        ) * batch.option_mask.unsqueeze(-1)
        value = self.value_head(state_summary).squeeze(-1)
        return ActorEncoding(
            state_tokens=state_tokens,
            state_mask=state_mask,
            state_summary=state_summary,
            entity_tokens=entity_tokens,
            option_tokens=options,
            value=value,
        )

    def forward(
        self,
        batch: PodNativeBatch | Mapping[str, Tensor],
        selected_prefix: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch = self.validate_batch(batch)
        encoded = self.encode(batch)
        state = self.action_decoder.initialize(batch, encoded.state_summary)
        state = self.action_decoder.consume_prefix(encoded.option_tokens, state, selected_prefix)
        return self.action_decoder.logits(batch, encoded.option_tokens, state), encoded.value

    def greedy(
        self, batch: PodNativeBatch | Mapping[str, Tensor]
    ) -> OrderedActions:
        batch = self.validate_batch(batch)
        encoded = self.encode(batch)
        return self.action_decoder.greedy(batch, encoded.option_tokens, encoded.state_summary)

    def sample(
        self, batch: PodNativeBatch | Mapping[str, Tensor]
    ) -> OrderedActions:
        batch = self.validate_batch(batch)
        encoded = self.encode(batch)
        return self.action_decoder.generate(
            batch, encoded.option_tokens, encoded.state_summary, stochastic=True
        )

    def act_device(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        actions = self.greedy(batch)
        return actions.sequences, actions.lengths

    def value(self, batch: PodNativeBatch | Mapping[str, Tensor]) -> Tensor:
        return self.encode(batch).value

    def evaluate_actions(
        self,
        batch: PodNativeBatch | Mapping[str, Tensor],
        sequences: Tensor,
        lengths: Tensor,
    ) -> tuple[ActionEvaluation, Tensor]:
        batch = self.validate_batch(batch)
        encoded = self.encode(batch)
        evaluated = self.action_decoder.evaluate(
            batch, encoded.option_tokens, encoded.state_summary, sequences, lengths
        )
        return evaluated, encoded.value


__all__ = ["ActorEncoding", "FieldEmbeddings", "PodNativeActorCritic"]
