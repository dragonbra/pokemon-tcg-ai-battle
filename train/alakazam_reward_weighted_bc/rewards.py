from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from .config import RewardConfig, WeightingConfig


def _field(record: Mapping[str, Any], path: str) -> object:
    value: object = record
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(path)
        value = value[part]
    return value


def compose_reward(
    record: Mapping[str, Any], config: RewardConfig
) -> tuple[float, dict[str, float]]:
    """Compose one scalar reward while retaining each weighted contribution."""
    contributions: dict[str, float] = {}
    total = 0.0
    for component in config.components:
        if not component.enabled:
            continue
        try:
            raw = _field(record, component.field)
        except KeyError:
            if component.default is None:
                raise ValueError(
                    f"record is missing reward field {component.field!r} for {component.name!r}"
                ) from None
            raw = component.default
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"reward field {component.field!r} must be numeric")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"reward field {component.field!r} must be finite")
        if component.clip_min is not None:
            value = max(component.clip_min, value)
        if component.clip_max is not None:
            value = min(component.clip_max, value)
        contribution = component.weight * value
        contributions[component.name] = contribution
        total += contribution
    if config.clip_min is not None:
        total = max(config.clip_min, total)
    if config.clip_max is not None:
        total = min(config.clip_max, total)
    return total, contributions


def compose_rewards(
    records: Sequence[Mapping[str, Any]], config: RewardConfig
) -> tuple[Tensor, dict[str, Tensor]]:
    totals: list[float] = []
    components: dict[str, list[float]] = {
        component.name: [] for component in config.components if component.enabled
    }
    for record in records:
        total, row = compose_reward(record, config)
        totals.append(total)
        for name in components:
            components[name].append(row[name])
    return (
        torch.tensor(totals, dtype=torch.float32),
        {name: torch.tensor(values, dtype=torch.float32) for name, values in components.items()},
    )


def imitation_weights(
    rewards: Tensor,
    config: WeightingConfig,
    *,
    dataset_mean: float = 0.0,
    values: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return detached advantages and safe non-negative sample weights."""
    if config.baseline == "zero":
        baseline: Tensor | float = 0.0
    elif config.baseline == "dataset_mean":
        baseline = float(dataset_mean)
    else:
        if values is None or values.shape != rewards.shape:
            raise ValueError("value baseline requires one value prediction per reward")
        baseline = values.detach()
    advantage = (rewards - baseline).detach()
    if config.mode == "uniform":
        weights = torch.ones_like(advantage)
    elif config.mode == "linear":
        weights = 1.0 + config.linear_scale * advantage
    else:
        weights = torch.exp((advantage / config.temperature).clamp(-20.0, 20.0))
    weights = weights.clamp(config.min_weight, config.max_weight).detach()
    return advantage, weights
