"""0015 optimization callback for bounded terminal-outcome-weighted BC."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import Tensor, nn

from .outcome_weighting import outcome_weights, weighted_token_cross_entropy


OptimizationResult = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Mapping[str, Tensor]]


def make_optimization_step(
    *,
    win_weight: float,
    loss_weight: float,
    draw_weight: float,
    label_smoothing: float = 0.0,
) -> Callable[[nn.Module, dict[str, Tensor]], OptimizationResult]:
    """Create the weighted train-only step while leaving validation untouched."""

    def optimization_step(
        model: nn.Module,
        batch: dict[str, Tensor],
    ) -> OptimizationResult:
        logits = model.teacher_logits(batch)
        mask = batch["targets"] != -100
        weights = outcome_weights(
            batch["outcome_id"],
            win_weight=win_weight,
            loss_weight=loss_weight,
            draw_weight=draw_weight,
        ).to(device=logits.device)
        loss, numerator, denominator = weighted_token_cross_entropy(
            logits,
            batch["targets"],
            weights,
            label_smoothing=label_smoothing,
        )
        correct = logits.argmax(-1).eq(batch["targets"]) & mask
        exact = (correct | ~mask).all(1)
        outcome_ids = batch["outcome_id"]
        diagnostics = {
            "weighted_loss_numerator": numerator,
            "weighted_loss_denominator": denominator,
            "decision_weight_sum": weights.sum(),
            "win_decisions": outcome_ids.eq(1).sum(),
            "loss_decisions": outcome_ids.eq(-1).sum(),
            "draw_decisions": outcome_ids.eq(0).sum(),
            "label_smoothing": logits.new_tensor(label_smoothing),
        }
        return loss, mask.sum(), correct.sum(), exact.sum(), logits, diagnostics

    return optimization_step


__all__ = ["make_optimization_step"]
