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
