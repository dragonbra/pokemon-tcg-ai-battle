from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    """Shape contract for the candidate-scoring policy/value model.

    Card and action ids use zero as padding. Encoders therefore allocate one
    extra embedding row and callers should pass ``id + 1`` for real ids.
    """

    state_numeric_dim: int = 24
    state_token_count: int = 24
    candidate_numeric_dim: int = 10
    max_candidates: int = 64
    card_vocab_size: int = 4096
    action_type_vocab_size: int = 32
    d_model: int = 128
    hidden_dim: int = 256
    num_heads: int = 2
    num_transformer_layers: int = 1
    dropout: float = 0.0
    deck_token_count: int = 0
    deck_numeric_dim: int = 0
    entity_token_count: int = 0
    entity_numeric_dim: int = 0
    history_token_count: int = 0
    history_numeric_dim: int = 0
    expert_vocab_size: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class CandidatePolicyValueNet(nn.Module):
    """Policy/value model for a variable set of legal action candidates.

    The model does not invent actions. The simulator supplies the candidates;
    this module scores them and masks padding or unavailable options. This
    keeps the model compatible with the PTCG ``select.option`` contract.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        embedding_size = config.card_vocab_size + 1
        action_type_size = config.action_type_vocab_size + 1

        self.state_card_embedding = nn.Embedding(embedding_size, config.d_model, padding_idx=0)
        self.state_position_embedding = nn.Embedding(
            config.state_token_count, config.d_model
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.state_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_transformer_layers,
        )
        self.state_numeric_encoder = nn.Sequential(
            nn.Linear(config.state_numeric_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )
        self.state_norm = nn.LayerNorm(config.d_model)

        self.universal = config.deck_token_count > 0
        if self.universal:
            if not all(
                value > 0
                for value in (
                    config.deck_numeric_dim,
                    config.entity_token_count,
                    config.entity_numeric_dim,
                    config.history_token_count,
                    config.history_numeric_dim,
                    config.expert_vocab_size,
                )
            ):
                raise ValueError("universal model dimensions must all be positive")
            self.deck_position_embedding = nn.Embedding(
                config.deck_token_count, config.d_model
            )
            self.deck_numeric_encoder = nn.Sequential(
                nn.Linear(config.deck_numeric_dim, config.d_model),
                nn.LayerNorm(config.d_model),
                nn.GELU(),
            )
            self.entity_position_embedding = nn.Embedding(
                config.entity_token_count, config.d_model
            )
            self.entity_numeric_encoder = nn.Sequential(
                nn.Linear(config.entity_numeric_dim, config.d_model),
                nn.LayerNorm(config.d_model),
                nn.GELU(),
            )
            self.history_position_embedding = nn.Embedding(
                config.history_token_count, config.d_model
            )
            self.history_numeric_encoder = nn.Sequential(
                nn.Linear(config.history_numeric_dim, config.d_model),
                nn.LayerNorm(config.d_model),
                nn.GELU(),
            )
            self.expert_embedding = nn.Embedding(config.expert_vocab_size, config.d_model)
            self.universal_fusion = nn.Sequential(
                nn.Linear(config.d_model * 5, config.d_model),
                nn.LayerNorm(config.d_model),
                nn.GELU(),
            )

        self.action_type_embedding = nn.Embedding(action_type_size, config.d_model, padding_idx=0)
        self.action_card_embedding = nn.Embedding(embedding_size, config.d_model, padding_idx=0)
        self.action_target_embedding = nn.Embedding(embedding_size, config.d_model, padding_idx=0)
        self.action_numeric_encoder = nn.Sequential(
            nn.Linear(config.candidate_numeric_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(config.d_model * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    @staticmethod
    def _pool_auxiliary(
        card_ids: Tensor,
        numeric: Tensor,
        card_embedding: nn.Embedding,
        numeric_encoder: nn.Module,
        position_embedding: nn.Embedding,
    ) -> Tensor:
        positions = torch.arange(card_ids.shape[1], device=card_ids.device)
        encoded = card_embedding(card_ids)
        encoded = encoded + position_embedding(positions).unsqueeze(0)
        encoded = encoded + numeric_encoder(numeric)
        present = (card_ids.ne(0) | numeric.abs().sum(dim=-1).gt(0)).unsqueeze(-1)
        present_float = present.to(encoded.dtype)
        return (encoded * present_float).sum(dim=1) / present_float.sum(dim=1).clamp_min(1.0)

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
        tokens = self.state_card_embedding(state_card_ids)
        tokens = tokens + self.state_position_embedding(positions).unsqueeze(0)
        padding_mask = state_card_ids.eq(0)
        # PyTorch's attention can produce NaNs when every token in a sample is
        # masked. Keep one neutral token visible to the transformer while the
        # original mask still prevents it from contributing to the pooled card
        # representation. Numeric state features remain available in that case.
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
            self.state_card_embedding,
            self.deck_numeric_encoder,
            self.deck_position_embedding,
        )
        entity = self._pool_auxiliary(
            entity_card_ids,
            entity_numeric,
            self.state_card_embedding,
            self.entity_numeric_encoder,
            self.entity_position_embedding,
        )
        history = self._pool_auxiliary(
            history_card_ids,
            history_numeric,
            self.state_card_embedding,
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
        candidates = (
            self.action_type_embedding(action_type_ids)
            + self.action_card_embedding(action_card_ids)
            + self.action_target_embedding(action_target_ids)
            + self.action_numeric_encoder(action_numeric)
        )
        return candidates

    def forward(
        self,
        state_numeric: Tensor,
        state_card_ids: Tensor,
        action_type_ids: Tensor,
        action_card_ids: Tensor,
        action_target_ids: Tensor,
        action_numeric: Tensor,
        action_mask: Tensor,
        deck_card_ids: Tensor | None = None,
        deck_card_numeric: Tensor | None = None,
        entity_card_ids: Tensor | None = None,
        entity_numeric: Tensor | None = None,
        history_card_ids: Tensor | None = None,
        history_numeric: Tensor | None = None,
        expert_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(value, masked_action_logits)``.

        ``action_mask`` is boolean with shape ``[batch, candidate_count]``.
        Every row must contain at least one true entry.
        """
        if action_mask.ndim != 2:
            raise ValueError("action_mask must have shape [batch, candidate_count]")
        if not torch.all(action_mask.any(dim=1)):
            raise ValueError("every sample must have at least one legal candidate")
        if action_type_ids.shape[1] != self.config.max_candidates:
            raise ValueError("candidate tensors must use ModelConfig.max_candidates")

        state = self._encode_state(
            state_numeric,
            state_card_ids,
            deck_card_ids,
            deck_card_numeric,
            entity_card_ids,
            entity_numeric,
            history_card_ids,
            history_numeric,
            expert_ids,
        )
        candidates = self._encode_candidates(
            action_type_ids,
            action_card_ids,
            action_target_ids,
            action_numeric,
        )
        state_for_candidates = state.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        logits = self.policy_head(torch.cat([state_for_candidates, candidates], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        value = torch.tanh(self.value_head(state).squeeze(-1))
        return value, logits

    @staticmethod
    def masked_probabilities(logits: Tensor, action_mask: Tensor) -> Tensor:
        """Convert masked candidate logits to a probability distribution."""
        masked = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        return torch.softmax(masked, dim=-1)
