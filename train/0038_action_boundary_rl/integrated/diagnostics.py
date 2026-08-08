"""Sparse fixed-minibatch gradient diagnostics; never part of PPO hot loop."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import torch
from torch import Tensor, nn


def gradient_diagnostics(
    losses: dict[str, Tensor], parameters: Iterable[nn.Parameter]
) -> dict[str, float]:
    params = tuple(parameter for parameter in parameters if parameter.requires_grad)
    vectors = {}
    for index, (name, loss) in enumerate(losses.items()):
        gradients = torch.autograd.grad(
            loss, params, retain_graph=index + 1 < len(losses), allow_unused=True
        )
        vectors[name] = torch.cat([
            (gradient if gradient is not None else torch.zeros_like(parameter)).reshape(-1)
            for gradient, parameter in zip(gradients, params, strict=True)
        ])
    metrics = {f"gradient/{name}/norm": float(vector.norm())
               for name, vector in vectors.items()}
    for left, right in combinations(vectors, 2):
        metrics[f"gradient/{left}_{right}/cosine"] = float(
            torch.nn.functional.cosine_similarity(vectors[left], vectors[right], dim=0)
        )
    return metrics


def allocation_margin(logits: Tensor) -> Tensor:
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("allocation margin requires at least two legal allocations")
    top = logits.softmax(-1).topk(2, dim=-1).values
    return top[:, 0] - top[:, 1]


__all__ = ["allocation_margin", "gradient_diagnostics"]
