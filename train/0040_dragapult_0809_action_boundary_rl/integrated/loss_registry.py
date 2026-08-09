from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class LossTerm:
    name: str
    value: Tensor
    weight: float
    optimizer_group: str
    enabled: bool = True


class LossRegistry:
    def __init__(self) -> None:
        self._terms: dict[str, LossTerm] = {}

    def register(self, term: LossTerm) -> None:
        if term.name in self._terms:
            raise ValueError(f"duplicate loss term: {term.name}")
        if term.weight < 0 or term.value.ndim != 0:
            raise ValueError("loss terms require nonnegative weight and scalar value")
        self._terms[term.name] = term

    def total(self) -> Tensor:
        active = [term.value * term.weight for term in self._terms.values() if term.enabled]
        if not active:
            return torch.zeros(())
        return torch.stack(active).sum()

    def metrics(self) -> dict[str, float]:
        return {
            f"loss/{name}": float(term.value.detach())
            for name, term in self._terms.items()
        } | {
            f"loss_weight/{name}": term.weight if term.enabled else 0.0
            for name, term in self._terms.items()
        }

    def optimizer_groups(self) -> dict[str, tuple[str, ...]]:
        output: dict[str, list[str]] = {}
        for term in self._terms.values():
            if term.enabled:
                output.setdefault(term.optimizer_group, []).append(term.name)
        return {key: tuple(value) for key, value in output.items()}


CORE_LOSS_NAMES = (
    "L_policy_win", "L_value_win", "L_value_prize", "A_prize",
    "L_opponent_meta", "L_entropy_root", "L_entropy_allocation", "L_tempo_aux",
)

__all__ = ["CORE_LOSS_NAMES", "LossRegistry", "LossTerm"]
