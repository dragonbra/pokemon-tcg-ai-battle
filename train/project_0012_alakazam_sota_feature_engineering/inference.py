from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

try:
    from .base_model import collate_id_only
except ImportError:
    from train.project_0010_alakazam_sota_model.model import collate_id_only

from .codec import FeatureEngineeringCodec
from .model import FeatureEngineeringPolicy, FeatureModelConfig


class FeatureEngineeringInference:
    def __init__(
        self,
        model: FeatureEngineeringPolicy,
        codec: FeatureEngineeringCodec,
    ) -> None:
        self.model = model.eval()
        self.codec = codec

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path) -> "FeatureEngineeringInference":
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        metadata = payload.get("metadata") or {}
        if metadata.get("model_version") != "alakazam_sota_feature_engineering_bc_v1":
            raise ValueError("wrong 0012 feature-engineering checkpoint model version")
        config_value = metadata.get("model_config")
        if not isinstance(config_value, dict):
            raise ValueError("checkpoint is missing model_config")
        config = FeatureModelConfig(**config_value)
        model = FeatureEngineeringPolicy(config)
        model.load_state_dict(payload["model"], strict=True)
        return cls(model, FeatureEngineeringCodec(config))

    def reset(self) -> None:
        return None

    def select(self, observation: dict[str, Any]) -> list[int]:
        encoded = self.codec.encode(observation, None)
        if encoded is None:
            raise RuntimeError("observation exceeds the 0012 codec contract")
        batch = collate_id_only([encoded])
        # The frozen daily-winner dataset builder filtered labels beyond max_action_steps,
        # while the live simulator can require longer mandatory selections (for example
        # Xerosic discards from a large hand). Inference must preserve the live legality
        # contract instead of silently truncating minCount/maxCount to the train cap.
        select = observation.get("select") or {}
        minimum = max(0, int(select.get("minCount", 0) or 0))
        maximum = max(minimum, int(select.get("maxCount", minimum) or minimum))
        batch["min_count"][0] = minimum
        batch["max_count"][0] = min(maximum, len(encoded["option_cat"]))
        if self.codec.config.action_primitive_context:
            primitive = encoded["option_primitive_cat"]
            batch["option_primitive_cat"] = torch.tensor([primitive], dtype=torch.long)
            batch["action_context_cat"] = torch.tensor(
                [encoded["action_context_cat"]], dtype=torch.long
            )
        with torch.inference_mode():
            return self.model.greedy_action(batch)
