"""Candidate-scoring model with a selection-cardinality head.

The simulator still supplies and masks the legal candidates.  The additional
head predicts how many distinct candidates the current select call requires;
the policy head then scores the candidates selected by a deterministic top-k
decoder.  This covers the variable-length list action used by effect choices
without delegating any choice to a rule teacher.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from rl.core.model import CandidatePolicyValueNet, ModelConfig


class FullActionPolicyValueNet(CandidatePolicyValueNet):
    """Policy/value network for single- and multi-candidate selections."""

    def __init__(self, config: ModelConfig, *, max_selection_count: int = 64) -> None:
        super().__init__(config)
        if max_selection_count < 1:
            raise ValueError("max_selection_count must be positive")
        self.max_selection_count = int(max_selection_count)
        self.selection_count_head = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, self.max_selection_count + 1),
        )

    def forward_with_count(
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
    ) -> tuple[Tensor, Tensor, Tensor]:
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
        count_logits = self.selection_count_head(state)
        return value, logits, count_logits
