"""Loss adapter for hierarchical root/allocation PPO probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .compound_gae import strategic_metric


@dataclass(frozen=True, slots=True)
class CompoundPPOLoss:
    policy_loss: Tensor
    entropy: Tensor
    approximate_kl: Tensor
    clip_fraction: Tensor
    ratio: Tensor


def compound_policy_loss(
    *,
    current_root_logprob: Tensor,
    current_parameter_logprob: Tensor,
    old_joint_logprob: Tensor,
    root_entropy: Tensor,
    parameter_entropy: Tensor,
    advantage: Tensor,
    policy_mask: Tensor,
    clip_ratio: float,
) -> CompoundPPOLoss:
    current_joint = current_root_logprob + current_parameter_logprob
    log_ratio = current_joint - old_joint_logprob.detach()
    ratio = log_ratio.exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
    policy = strategic_metric(-torch.minimum(unclipped, clipped), policy_mask)
    entropy = strategic_metric(root_entropy + parameter_entropy, policy_mask)
    approximate_kl = strategic_metric((ratio - 1.0) - log_ratio, policy_mask)
    clip_fraction = strategic_metric(
        ((ratio - 1.0).abs() > clip_ratio).to(ratio.dtype), policy_mask
    )
    return CompoundPPOLoss(policy, entropy, approximate_kl, clip_fraction, ratio)


__all__ = ["CompoundPPOLoss", "compound_policy_loss"]
