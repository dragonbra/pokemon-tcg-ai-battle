"""Positive episode-outcome weights for behavior-cloning optimization only."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def _validate_weight(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive finite")


def outcome_weights(
    outcome_ids: Tensor,
    *,
    win_weight: float,
    loss_weight: float,
    draw_weight: float,
) -> Tensor:
    """Map the frozen ``1/-1/0`` outcome contract to decision weights."""
    for name, value in (
        ("win_weight", win_weight),
        ("loss_weight", loss_weight),
        ("draw_weight", draw_weight),
    ):
        _validate_weight(name, value)
    if outcome_ids.ndim != 1:
        raise ValueError("outcome_id must be a one-dimensional tensor")
    valid = outcome_ids.eq(1) | outcome_ids.eq(-1) | outcome_ids.eq(0)
    if not bool(valid.all()):
        invalid = sorted(int(value) for value in outcome_ids[~valid].unique().tolist())
        raise ValueError(f"outcome_id contains unsupported values: {invalid}")
    result = torch.full_like(outcome_ids, draw_weight, dtype=torch.float32)
    result = torch.where(outcome_ids.eq(1), result.new_tensor(win_weight), result)
    return torch.where(outcome_ids.eq(-1), result.new_tensor(loss_weight), result)


def weighted_token_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    decision_weights: Tensor,
    *,
    label_smoothing: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Weight valid BC tokens by episode outcome while preserving the token-mean scale."""
    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("logits and targets must have shapes [B,T,C] and [B,T]")
    if logits.shape[:2] != targets.shape or decision_weights.shape != targets.shape[:1]:
        raise ValueError("logits, targets, and decision weights have incompatible shapes")
    if not bool(torch.isfinite(decision_weights).all()) or bool((decision_weights <= 0).any()):
        raise ValueError("decision weights must be positive finite")
    if not math.isfinite(label_smoothing) or not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label smoothing must be finite and in [0, 1)")
    valid = targets.ne(-100)
    if label_smoothing == 0.0:
        losses = F.cross_entropy(
            logits.flatten(0, 1),
            targets.flatten(),
            ignore_index=-100,
            reduction="none",
        ).view_as(targets)
    else:
        mask_floor = torch.finfo(logits.dtype).min / 2
        legal = torch.isfinite(logits) & logits.gt(mask_floor)
        legal_count = legal.sum(dim=-1)
        if bool((legal_count[valid] == 0).any()):
            raise ValueError("smoothed training token contains no legal class")
        safe_targets = targets.masked_fill(~valid, 0)
        target_legal = legal.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        if bool((~target_legal & valid).any()):
            raise ValueError("smoothed training target is not a legal finite class")
        log_probabilities = torch.log_softmax(logits, dim=-1)
        negative_log_likelihood = -log_probabilities.gather(
            -1, safe_targets.unsqueeze(-1)
        ).squeeze(-1)
        legal_log_sum = torch.where(
            legal, log_probabilities, torch.zeros_like(log_probabilities)
        ).sum(dim=-1)
        smooth_loss = -legal_log_sum / legal_count.clamp_min(1)
        losses = (
            (1.0 - label_smoothing) * negative_log_likelihood
            + label_smoothing * smooth_loss
        ).masked_fill(~valid, 0.0)
    token_weights = decision_weights.to(dtype=losses.dtype).unsqueeze(1) * valid
    numerator = (losses * token_weights).sum()
    denominator = token_weights.sum()
    if not bool(denominator > 0):
        raise ValueError("weighted training batch contains no valid target token")
    return numerator / denominator, numerator.detach(), denominator.detach()


__all__ = ["outcome_weights", "weighted_token_cross_entropy"]
