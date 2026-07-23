from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import LossConfig


def per_sample_full_action_losses(
    logits: Tensor,
    count_logits: Tensor,
    *,
    target_mask: Tensor,
    target_count: Tensor,
    action_mask: Tensor,
    selection_min_count: Tensor,
    selection_max_count: Tensor,
) -> tuple[Tensor, Tensor]:
    """Match the BC V4 single/multi/count contract without reducing the batch."""
    single_target = target_mask.argmax(dim=1)
    single_loss = F.cross_entropy(logits.clamp(-30.0, 30.0), single_target, reduction="none")
    binary = F.binary_cross_entropy_with_logits(
        logits.clamp(-30.0, 30.0), target_mask, reduction="none"
    )
    multi_loss = (binary * action_mask.float()).sum(dim=1) / (
        action_mask.float().sum(dim=1).clamp_min(1.0)
    )
    policy_loss = torch.where(target_count.eq(1), single_loss, multi_loss)

    positions = torch.arange(count_logits.shape[1], device=count_logits.device).unsqueeze(0)
    maximum = selection_max_count.clamp_max(count_logits.shape[1] - 1)
    valid = (positions >= selection_min_count.unsqueeze(1)) & (positions <= maximum.unsqueeze(1))
    masked_count_logits = count_logits.masked_fill(~valid, torch.finfo(count_logits.dtype).min)
    count_loss = F.cross_entropy(
        masked_count_logits,
        target_count.clamp_max(count_logits.shape[1] - 1),
        reduction="none",
    )
    return policy_loss, count_loss


def weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same shape")
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def combined_loss(
    policy_losses: Tensor,
    count_losses: Tensor,
    values: Tensor,
    rewards: Tensor,
    weights: Tensor,
    config: LossConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    weighted_policy = weighted_mean(policy_losses, weights)
    weighted_count = weighted_mean(count_losses, weights)
    value = F.smooth_l1_loss(values, rewards)
    total = config.policy * weighted_policy + config.count * weighted_count + config.value * value
    return total, {
        "policy": weighted_policy,
        "count": weighted_count,
        "value": value,
    }
