"""One-encoder resident policy routing focal player 0 against frozen heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from ..contract import PodNativeBatch
from .actor_critic import PodNativeActorCritic
from .frozen_heads import FrozenDeckHeads


@dataclass(frozen=True, slots=True)
class RoutedActions:
    sequences: Tensor
    lengths: Tensor
    legal: Tensor
    logprob: Tensor
    value: Tensor
    focal_mask: Tensor


class RoutedResidentPolicy(nn.Module):
    def __init__(self, learner: PodNativeActorCritic, frozen_heads: FrozenDeckHeads) -> None:
        super().__init__()
        self.learner = learner
        self.frozen_heads = frozen_heads
        self.frozen_heads.requires_grad_(False)

    def act(
        self,
        raw_batch: PodNativeBatch | Mapping[str, Tensor],
        opponent_head_indices: Tensor,
        *,
        stochastic_focal: bool,
    ) -> RoutedActions:
        batch = self.learner.validate_batch(raw_batch)
        encoded = self.learner.encode(batch)
        focal = batch.global_cat[:, 3].eq(1)
        learner_actions = self.learner.action_decoder.generate(
            batch,
            encoded.option_tokens,
            encoded.state_summary,
            stochastic=stochastic_focal,
        )
        frozen_actions = self.frozen_heads.grouped_greedy(
            batch,
            encoded.option_tokens,
            encoded.state_summary,
            max_action_steps=self.learner.config.max_action_steps,
        )
        return RoutedActions(
            sequences=torch.where(
                focal.unsqueeze(1), learner_actions.sequences, frozen_actions.sequences
            ),
            lengths=torch.where(focal, learner_actions.lengths, frozen_actions.lengths),
            legal=torch.where(focal, learner_actions.legal, frozen_actions.legal),
            logprob=torch.where(
                focal,
                learner_actions.logprob
                if learner_actions.logprob is not None
                else torch.zeros_like(encoded.value),
                torch.zeros_like(encoded.value),
            ),
            value=torch.where(focal, encoded.value, torch.zeros_like(encoded.value)),
            focal_mask=focal,
        )


__all__ = ["RoutedActions", "RoutedResidentPolicy"]
