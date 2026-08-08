"""Semi-MDP GAE over real policy transitions, independent of callback count."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..rollout.protocol import PolicyTransition


@dataclass(frozen=True, slots=True)
class CompoundTargets:
    advantages: Tensor
    returns: Tensor
    policy_mask: Tensor
    value_mask: Tensor


def compound_gae(transitions: list[PolicyTransition], *, gae_lambda: float) -> CompoundTargets:
    if not transitions:
        raise ValueError("compound GAE requires at least one transition")
    if not 0.0 < gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in (0, 1]")
    advantages = [0.0] * len(transitions)
    continuation = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        transition = transitions[index]
        bootstrap = 0.0 if transition.done else transition.next_value
        delta = (
            transition.accumulated_reward
            + transition.accumulated_discount * bootstrap
            - transition.pre_action_value
        )
        advantages[index] = delta + (
            0.0 if transition.done else
            transition.accumulated_discount * gae_lambda * continuation
        )
        continuation = advantages[index]
    advantage = torch.tensor(advantages, dtype=torch.float32)
    values = torch.tensor([item.pre_action_value for item in transitions], dtype=torch.float32)
    valid = torch.tensor([item.valid for item in transitions], dtype=torch.bool)
    strategic = torch.tensor([item.valid and item.strategic for item in transitions], dtype=torch.bool)
    return CompoundTargets(advantage, advantage + values, strategic, valid)


def strategic_metric(values: Tensor, policy_mask: Tensor) -> Tensor:
    """Entropy/KL/clip denominator is exactly the strategic valid set."""
    selected = values[policy_mask]
    if selected.numel() == 0:
        raise ValueError("no strategic policy samples")
    return selected.mean()


__all__ = ["CompoundTargets", "compound_gae", "strategic_metric"]
