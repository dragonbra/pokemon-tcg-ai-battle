from __future__ import annotations

import torch
from torch import Tensor, nn

from rl_environment.model import ModelConfig
from train.alakazam_bc_rl.full_action_model import FullActionPolicyValueNet


class CategoryAugmentedFullActionPolicyValueNet(FullActionPolicyValueNet):
    """Stage-two model with exact card-category embeddings beside card IDs."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        card_category_lookup: Tensor | list[int],
        category_vocab_size: int,
        category_scale: float = 1.0,
        max_selection_count: int = 64,
    ) -> None:
        super().__init__(config, max_selection_count=max_selection_count)
        lookup = torch.as_tensor(card_category_lookup, dtype=torch.long)
        if lookup.shape != (config.card_vocab_size + 1,):
            raise ValueError(
                "card_category_lookup must have shape "
                f"[{config.card_vocab_size + 1}]"
            )
        if category_vocab_size < 1:
            raise ValueError("category_vocab_size must be positive")
        if lookup.min().item() < 0 or lookup.max().item() > category_vocab_size:
            raise ValueError("card category lookup contains an out-of-range category")
        if category_scale < 0:
            raise ValueError("category_scale must be non-negative")
        self.category_vocab_size = int(category_vocab_size)
        self.category_scale = float(category_scale)
        self.register_buffer("card_category_lookup", lookup, persistent=True)
        self.card_category_embedding = nn.Embedding(
            self.category_vocab_size + 1,
            config.d_model,
            padding_idx=0,
        )

    def _category_semantics(self, card_ids: Tensor) -> Tensor:
        categories = self.card_category_lookup[card_ids]
        return self.card_category_embedding(categories) * self.category_scale

    def _state_card_embedding(self, card_ids: Tensor) -> Tensor:
        return self.state_card_embedding(card_ids) + self._category_semantics(card_ids)

    def _encode_state(
        self,
        state_numeric: Tensor,
        state_card_ids: Tensor,
        deck_card_ids: Tensor | None = None,
        deck_card_numeric: Tensor | None = None,
        entity_card_ids: Tensor | None = None,
        entity_numeric: Tensor | None = None,
        history_card_ids: Tensor | None = None,
        history_numeric: Tensor | None = None,
        expert_ids: Tensor | None = None,
    ) -> Tensor:
        if state_card_ids.shape[1] != self.config.state_token_count:
            raise ValueError(
                "state_card_ids has an unexpected token count: "
                f"{state_card_ids.shape[1]} != {self.config.state_token_count}"
            )
        positions = torch.arange(
            self.config.state_token_count,
            device=state_card_ids.device,
        )
        tokens = self._state_card_embedding(state_card_ids)
        tokens = tokens + self.state_position_embedding(positions).unsqueeze(0)
        padding_mask = state_card_ids.eq(0)
        safe_padding_mask = padding_mask.clone()
        empty_rows = safe_padding_mask.all(dim=1)
        safe_padding_mask[empty_rows, 0] = False
        encoded = self.state_transformer(tokens, src_key_padding_mask=safe_padding_mask)
        present = (~padding_mask).unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * present).sum(dim=1) / present.sum(dim=1).clamp_min(1.0)
        numeric = self.state_numeric_encoder(state_numeric)
        base = self.state_norm(pooled + numeric)
        if not self.universal:
            return base
        required = (
            deck_card_ids,
            deck_card_numeric,
            entity_card_ids,
            entity_numeric,
            history_card_ids,
            history_numeric,
            expert_ids,
        )
        if any(value is None for value in required):
            raise ValueError("universal model inputs are incomplete")
        deck = self._pool_auxiliary(
            deck_card_ids,
            deck_card_numeric,
            self._state_card_embedding,
            self.deck_numeric_encoder,
            self.deck_position_embedding,
        )
        entity = self._pool_auxiliary(
            entity_card_ids,
            entity_numeric,
            self._state_card_embedding,
            self.entity_numeric_encoder,
            self.entity_position_embedding,
        )
        history = self._pool_auxiliary(
            history_card_ids,
            history_numeric,
            self._state_card_embedding,
            self.history_numeric_encoder,
            self.history_position_embedding,
        )
        expert = self.expert_embedding(expert_ids.reshape(-1))
        return self.universal_fusion(torch.cat([base, deck, entity, history, expert], dim=-1))

    def _encode_candidates(
        self,
        action_type_ids: Tensor,
        action_card_ids: Tensor,
        action_target_ids: Tensor,
        action_numeric: Tensor,
    ) -> Tensor:
        return (
            self.action_type_embedding(action_type_ids)
            + self.action_card_embedding(action_card_ids)
            + self._category_semantics(action_card_ids)
            + self.action_target_embedding(action_target_ids)
            + self._category_semantics(action_target_ids)
            + self.action_numeric_encoder(action_numeric)
        )
