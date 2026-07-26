from __future__ import annotations

from typing import Any

try:
    from .base_model import IDOnlyCodec
except ImportError:  # Repository training path; candidate packages include base_model.py.
    from train.0010_alakazam_sota_model.model import IDOnlyCodec

from .model import FeatureModelConfig


def _int(value: Any, default: int = 0) -> int:
    try:
        return default if value is None else int(value)
    except (TypeError, ValueError):
        return default


def _card_id(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    return max(0, _int(value.get("id", value.get("cardId", 0))))


class FeatureEngineeringCodec(IDOnlyCodec):
    """0010 codec plus audited action primitive/context fields for E3+."""

    def __init__(self, config: FeatureModelConfig):
        super().__init__(config)
        self.config = config

    def encode(
        self,
        observation: dict[str, Any],
        action: list[int] | None,
    ) -> dict[str, Any] | None:
        encoded = super().encode(observation, action)
        if encoded is None or not self.config.action_primitive_context:
            return encoded
        select = observation.get("select") or {}
        options = select.get("option") or []
        primitive: list[list[int]] = []
        for raw in options:
            option = raw if isinstance(raw, dict) else {}
            primitive.append(
                [
                    min(2048, max(0, _int(option.get("attackId"))) + 1)
                    if option.get("attackId") is not None
                    else 0,
                    min(128, max(0, _int(option.get("count"))) + 1)
                    if option.get("count") is not None
                    else 0,
                    min(128, max(0, _int(option.get("energyIndex"))) + 1)
                    if option.get("energyIndex") is not None
                    else 0,
                    min(32, max(0, _int(option.get("toolIndex"))) + 1)
                    if option.get("toolIndex") is not None
                    else 0,
                ]
            )
        encoded["option_primitive_cat"] = primitive
        encoded["action_context_cat"] = [
            min(self.config.max_card_id, _card_id(select.get("effect"))),
            min(self.config.max_card_id, _card_id(select.get("contextCard"))),
            min(16, max(0, _int(select.get("minCount"))) + 1),
            min(16, max(0, _int(select.get("maxCount"))) + 1),
        ]
        return encoded
