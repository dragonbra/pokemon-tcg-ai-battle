"""Explicit model registry."""
from __future__ import annotations

from collections.abc import Mapping

from .variants import ModelConfig, SemanticGoalPolicy

MODEL_REGISTRY: Mapping[str, type[SemanticGoalPolicy]] = {
    **{f"M{index}": SemanticGoalPolicy for index in range(6)},
    "M5.1": SemanticGoalPolicy,
}


def create_model(name: str, config: ModelConfig | None = None) -> SemanticGoalPolicy:
    try:
        implementation = MODEL_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"unknown model variant: {name}") from error
    return implementation(name, config)


__all__ = ["MODEL_REGISTRY", "create_model"]
