from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import IDOnlyCodec, IDOnlyConfig, IDOnlyPointerPolicy, collate_id_only


class IDOnlyPolicy:
    """Stateless CPU inference wrapper for one official simulator observation."""

    def __init__(self, model: IDOnlyPointerPolicy, codec: IDOnlyCodec) -> None:
        self.model = model.eval()
        self.codec = codec

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path) -> "IDOnlyPolicy":
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        metadata = payload.get("metadata") or {}
        config_value = metadata.get("model_config")
        if not isinstance(config_value, dict):
            raise ValueError("checkpoint metadata is missing model_config")
        config = IDOnlyConfig(**config_value)
        model = IDOnlyPointerPolicy(config)
        model.load_state_dict(payload["model"], strict=True)
        return cls(model, IDOnlyCodec(config))

    def reset(self) -> None:
        return None

    def select(self, observation: dict[str, Any]) -> list[int]:
        encoded = self.codec.encode(observation, None)
        if encoded is None:
            raise RuntimeError("observation exceeds the frozen ID-only codec contract")
        batch = collate_id_only([encoded])
        with torch.inference_mode():
            return self.model.greedy_action(batch)
