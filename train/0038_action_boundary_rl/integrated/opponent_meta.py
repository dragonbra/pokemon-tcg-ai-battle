"""Opponent archetype prediction from deployable public representations only."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class OpponentMetaHead(nn.Module):
    def __init__(self, width: int, archetypes: int) -> None:
        super().__init__()
        if archetypes < 2:
            raise ValueError("taxonomy must include at least unknown and one archetype")
        self.classifier = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, archetypes)
        )

    def forward(self, public_state_summary: Tensor) -> Tensor:
        return self.classifier(public_state_summary)


class OpponentMetaConditioner(nn.Module):
    """Detached predicted distribution -> zero-initialized Action-query residual."""

    def __init__(self, width: int, archetypes: int) -> None:
        super().__init__()
        self.archetype_embeddings = nn.Parameter(torch.empty(archetypes, width))
        nn.init.normal_(self.archetype_embeddings, std=width ** -0.5)
        self.output = nn.Linear(width, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state_summary: Tensor, meta_logits: Tensor, *, detach: bool = True) -> Tensor:
        probabilities = meta_logits.softmax(dim=-1)
        if detach:
            probabilities = probabilities.detach()
        z_opp = probabilities @ self.archetype_embeddings
        return state_summary + self.output(z_opp)


@dataclass(frozen=True, slots=True)
class MetaCalibration:
    accuracy: float
    macro_f1: float
    ece: float
    brier: float


def meta_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Labels exist only at training/evaluation call sites, never in inference tensors."""
    return torch.nn.functional.cross_entropy(logits, labels)


def calibration(logits: Tensor, labels: Tensor, *, bins: int = 10) -> MetaCalibration:
    probabilities = logits.softmax(-1)
    confidence, predicted = probabilities.max(-1)
    accuracy = predicted.eq(labels).float().mean()
    classes = logits.shape[-1]
    f1s = []
    for category in range(classes):
        pred = predicted.eq(category)
        true = labels.eq(category)
        tp = (pred & true).sum().float()
        denominator = 2 * tp + (pred & ~true).sum() + (~pred & true).sum()
        f1s.append(torch.where(denominator > 0, 2 * tp / denominator, denominator.new_zeros(())))
    ece = logits.new_zeros(())
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = confidence.ge(low) & (confidence.lt(high) if index + 1 < bins else confidence.le(high))
        if mask.any():
            ece += mask.float().mean() * (
                confidence[mask].mean() - predicted[mask].eq(labels[mask]).float().mean()
            ).abs()
    one_hot = torch.nn.functional.one_hot(labels, classes).to(probabilities.dtype)
    brier = (probabilities - one_hot).square().sum(-1).mean()
    return MetaCalibration(float(accuracy), float(torch.stack(f1s).mean()), float(ece), float(brier))


__all__ = [
    "MetaCalibration", "OpponentMetaConditioner", "OpponentMetaHead", "calibration", "meta_loss",
]
