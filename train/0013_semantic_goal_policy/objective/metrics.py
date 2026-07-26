"""Representation, counterfactual, and invariance evidence metrics."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class InvarianceMetrics:
    aligned_agreement: float
    mean_kl: float


def aligned_invariance(reference_logits: Tensor, permuted_logits: Tensor, old_to_new: Tensor) -> InvarianceMetrics:
    aligned = permuted_logits.index_select(-1, old_to_new)
    reference_log = torch.log_softmax(reference_logits, dim=-1)
    aligned_log = torch.log_softmax(aligned, dim=-1)
    probabilities = reference_log.exp()
    kl = (probabilities * (reference_log - aligned_log)).sum(-1).mean()
    agreement = (reference_logits.argmax(-1) == aligned.argmax(-1)).float().mean()
    return InvarianceMetrics(float(agreement.item()), float(kl.item()))


def expected_direction_rate(before: Tensor, after: Tensor, expected_sign: Tensor) -> float:
    if before.shape != after.shape or before.shape != expected_sign.shape:
        raise ValueError("counterfactual metric shapes disagree")
    direction = torch.sign(after - before)
    return float((direction == expected_sign).float().mean().item())


def unknown_vs_zero_rate(unknown_logits: Tensor, zero_logits: Tensor, *, epsilon: float = 1e-6) -> float:
    if unknown_logits.shape != zero_logits.shape:
        raise ValueError("unknown-vs-zero metric shapes disagree")
    return float(((unknown_logits - zero_logits).abs() > epsilon).float().mean().item())


def irrelevant_goal_stability(before: Tensor, after: Tensor) -> float:
    return float((after - before).abs().mean().item())


__all__ = ["InvarianceMetrics", "aligned_invariance", "expected_direction_rate", "irrelevant_goal_stability", "unknown_vs_zero_rate"]
