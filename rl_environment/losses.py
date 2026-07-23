from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_cross_entropy(logits: Tensor, target: Tensor, action_mask: Tensor) -> Tensor:
    """Cross entropy over the current legal candidate set."""
    if target.ndim != 1:
        raise ValueError("target must have shape [batch]")
    rows = torch.arange(target.shape[0], device=target.device)
    if not action_mask[rows, target].all():
        raise ValueError("cross-entropy target contains an illegal action")
    masked_logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
    return F.cross_entropy(masked_logits, target)


def masked_cross_entropy_per_sample(
    logits: Tensor,
    target: Tensor,
    action_mask: Tensor,
) -> Tensor:
    """Return one legal-candidate cross-entropy value per sample."""
    if target.ndim != 1:
        raise ValueError("target must have shape [batch]")
    rows = torch.arange(target.shape[0], device=target.device)
    if not action_mask[rows, target].all():
        raise ValueError("cross-entropy target contains an illegal action")
    masked_logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
    return F.cross_entropy(masked_logits, target, reduction="none")


def masked_soft_cross_entropy_per_sample(
    logits: Tensor,
    target_distribution: Tensor,
    action_mask: Tensor,
) -> Tensor:
    """Cross entropy against a probability target over legal candidates."""
    if target_distribution.shape != logits.shape or action_mask.shape != logits.shape:
        raise ValueError("soft target, logits, and action_mask must have the same shape")
    if (target_distribution < 0).any():
        raise ValueError("soft policy target cannot contain negative probabilities")
    if (target_distribution.masked_fill(~action_mask, 0.0) != target_distribution).any():
        raise ValueError("soft policy target contains an illegal action")
    mass = target_distribution.sum(dim=-1)
    if not torch.isfinite(mass).all() or (mass <= 0).any():
        raise ValueError("each soft policy target must have positive finite mass")
    target = target_distribution / mass.unsqueeze(-1)
    masked_logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
    log_probabilities = F.log_softmax(masked_logits, dim=-1)
    return -(target * log_probabilities).sum(dim=-1)


def masked_huber_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    delta: float = 0.1,
) -> Tensor:
    """Huber loss for padded candidate targets."""
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have the same shape")
    loss = F.huber_loss(prediction, target, reduction="none", delta=delta)
    mask_float = mask.to(loss.dtype)
    return (loss * mask_float).sum() / mask_float.sum().clamp_min(1.0)


def value_huber_loss(prediction: Tensor, target: Tensor, delta: float = 0.2) -> Tensor:
    """Huber loss for value targets in the ``[-1, 1]`` range."""
    return F.huber_loss(prediction, target, delta=delta)
