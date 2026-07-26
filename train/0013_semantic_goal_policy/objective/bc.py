"""Teacher-forced behavioral-cloning objective over full actions."""
from __future__ import annotations

import torch
from torch import Tensor


def sequence_cross_entropy(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    if logits.ndim != 3 or targets.shape != logits.shape[:2] or mask.shape != targets.shape:
        raise ValueError("BC tensor shapes disagree")
    losses = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").view_as(targets)
    denominator = mask.sum().clamp_min(1)
    return (losses * mask).sum() / denominator


def exact_action_rate(predicted: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    return (((predicted == targets) | ~mask).all(dim=1)).float().mean()


__all__ = ["exact_action_rate", "sequence_cross_entropy"]
