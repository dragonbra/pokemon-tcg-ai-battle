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


def compound_gae(
    transitions: list[PolicyTransition], *, gae_lambda: float,
    credit_clock: str = "selection",
) -> CompoundTargets:
    if not transitions:
        raise ValueError("compound GAE requires at least one transition")
    if not 0.0 < gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in (0, 1]")
    if credit_clock not in {"selection", "turn"}:
        raise ValueError(f"unsupported credit clock: {credit_clock}")
    turns = [item.metadata.get("turn") for item in transitions]
    if credit_clock == "turn" and any(
        isinstance(turn, bool) or not isinstance(turn, int) or turn < 0
        for turn in turns
    ):
        raise ValueError("turn clock requires nonnegative transition turn metadata")
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
        if transition.done:
            continuation_weight = 0.0
        else:
            lambda_weight = (
                gae_lambda
                if (
                    credit_clock == "selection"
                    or index + 1 == len(transitions)
                    or turns[index] != turns[index + 1]
                )
                else 1.0
            )
            continuation_weight = transition.accumulated_discount * lambda_weight
        advantages[index] = delta + continuation_weight * continuation
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
