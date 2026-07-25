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
                    f"record is missing reward field {component.field!r} "
                    f"for {component.name!r}"
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
    component_values: dict[str, list[float]] = {
        component.name: [] for component in config.components if component.enabled
    }
    for record in records:
        total, contributions = compose_reward(record, config)
        totals.append(total)
        for name in component_values:
            component_values[name].append(contributions[name])
    return (
        torch.tensor(totals, dtype=torch.float32),
        {
            name: torch.tensor(values, dtype=torch.float32)
            for name, values in component_values.items()
        },
    )


def imitation_weights(
    rewards: Tensor,
    config: WeightingConfig,
    *,
    dataset_mean: float,
) -> tuple[Tensor, Tensor]:
    baseline = dataset_mean if config.baseline == "dataset_mean" else 0.0
    advantage = (rewards - baseline).detach()
    if config.mode == "uniform":
        weights = torch.ones_like(advantage)
    elif config.mode == "linear":
        weights = 1.0 + config.linear_scale * advantage
    else:
        weights = torch.exp((advantage / config.temperature).clamp(-20.0, 20.0))
    return advantage, weights.clamp(config.min_weight, config.max_weight).detach()


def weighted_token_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    weights: Tensor,
) -> Tensor:
    """Apply one decision weight to each valid token while preserving BC scale."""
    if logits.shape[:2] != targets.shape or weights.shape != targets.shape[:1]:
        raise ValueError("logits, targets, and weights have incompatible shapes")
    valid = targets.ne(-100)
    token_losses = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        targets.flatten(),
        ignore_index=-100,
        reduction="none",
    ).view_as(targets)
    token_weights = weights.unsqueeze(1) * valid
    return (token_losses * token_weights).sum() / token_weights.sum().clamp_min(1e-8)
