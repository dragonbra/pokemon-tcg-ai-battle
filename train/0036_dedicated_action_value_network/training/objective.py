"""Episode-balanced multi-task objective with Value BCE as selection loss."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor
from ..model.value_network import ValueOutputs


@dataclass(frozen=True, slots=True)
class LossWeights:
    archetype: float = 0.0
    hand: float = 0.0
    final_diff: float = 0.0

    def validate(self) -> None:
        if self.hand != 0.0:
            raise ValueError("0036 V1 hidden-hand loss is reserved but not implemented")
        if self.archetype < 0.0 or self.final_diff < 0.0:
            raise ValueError("0036 auxiliary weights must be nonnegative")


@dataclass(frozen=True, slots=True)
class LossResult:
    total: Tensor
    value: Tensor
    archetype: Tensor
    final_diff: Tensor


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("0036 loss values and weights must be aligned vectors")
    return (values * weights).sum() / weights.sum().clamp_min(torch.finfo(values.dtype).eps)


def value_objective(outputs: ValueOutputs, *, value_target: Tensor, archetype_target: Tensor,
                    final_diff_target: Tensor, episode_weight: Tensor,
                    weights: LossWeights = LossWeights()) -> LossResult:
    weights.validate()
    sample_weight = episode_weight.to(outputs.value_logit.dtype)
    value = _weighted_mean(
        torch.nn.functional.binary_cross_entropy_with_logits(
            outputs.value_logit, value_target.to(outputs.value_logit.dtype), reduction="none"
        ), sample_weight,
    )
    archetype = _weighted_mean(
        torch.nn.functional.cross_entropy(outputs.archetype_logits, archetype_target.long(), reduction="none"),
        sample_weight,
    )
    final_diff = _weighted_mean(
        torch.nn.functional.cross_entropy(outputs.final_diff_logits, final_diff_target.long(), reduction="none"),
        sample_weight,
    )
    total = value + weights.archetype * archetype + weights.final_diff * final_diff
    return LossResult(total, value, archetype, final_diff)


__all__ = ["LossResult", "LossWeights", "value_objective"]
