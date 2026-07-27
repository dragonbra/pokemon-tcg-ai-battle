"""CPU inference session for the 0014 AC policy."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from .ac_model import ACModelConfig, AllFeatureCausalPolicy
from .base_model import IDOnlyConfig
from .online_runtime import OnlineCausalEncoder


class AllFeaturePolicy:
    def __init__(
        self,
        model: AllFeatureCausalPolicy,
        deck: Sequence[int],
        config: IDOnlyConfig,
    ) -> None:
        torch.set_num_threads(1)
        self.model = model.eval()
        self.deck = tuple(deck)
        self.config = config
        self.encoder: OnlineCausalEncoder | None = None

    @classmethod
    def from_checkpoint(
        cls, checkpoint: str | Path, ontology_path: str | Path, deck: Sequence[int]
    ) -> "AllFeaturePolicy":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata") or {}
        raw = metadata.get("model")
        if not isinstance(raw, dict) or not isinstance(raw.get("base"), dict):
            raise ValueError("AC checkpoint metadata is missing model.base")
        base = IDOnlyConfig(**raw["base"])
        config = ACModelConfig(
            base=base,
            event_layers=int(raw["event_layers"]),
            auxiliary_ffn_multiplier=int(raw["auxiliary_ffn_multiplier"]),
            goal_roles=int(raw["goal_roles"]),
        )
        model = AllFeatureCausalPolicy(config, ontology_path=ontology_path)
        model.load_state_dict(payload["model"], strict=True)
        return cls(model, deck, base)

    def reset(self) -> None:
        self.encoder = None

    def select(self, observation: dict[str, Any]) -> list[int]:
        current = observation.get("current") or {}
        actor = current.get("yourIndex")
        if actor not in (0, 1):
            raise ValueError("observation has no valid actor")
        if self.encoder is None or self.encoder.actor != actor:
            self.encoder = OnlineCausalEncoder(actor, self.deck, self.config)
        batch = self.encoder.encode(observation)
        with torch.inference_mode():
            return self.model.greedy_action(batch)


__all__ = ["AllFeaturePolicy"]
