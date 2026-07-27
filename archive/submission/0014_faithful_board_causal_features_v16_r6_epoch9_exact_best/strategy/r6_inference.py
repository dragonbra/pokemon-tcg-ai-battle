"""CPU inference session for the 0014 R6 sparse family mixture."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from .ac_model import ACModelConfig
from .base_model import IDOnlyConfig
from .online_runtime import OnlineCausalEncoder
from .r6_model import R6ModelConfig, R6SparseFamilyMixturePolicy


class R6Policy:
    def __init__(
        self,
        model: R6SparseFamilyMixturePolicy,
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
    ) -> "R6Policy":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata") or {}
        raw = metadata.get("model")
        if not isinstance(raw, dict) or not isinstance(raw.get("ac"), dict):
            raise ValueError("R6 checkpoint metadata is missing model.ac")
        ac_raw = raw["ac"]
        if not isinstance(ac_raw.get("base"), dict):
            raise ValueError("R6 checkpoint metadata is missing model.ac.base")
        base = IDOnlyConfig(**ac_raw["base"])
        ac = ACModelConfig(
            base=base,
            event_layers=int(ac_raw["event_layers"]),
            auxiliary_ffn_multiplier=int(ac_raw["auxiliary_ffn_multiplier"]),
            goal_roles=int(ac_raw["goal_roles"]),
        )
        config = R6ModelConfig(
            ac=ac, expert_ffn_multiplier=int(raw["expert_ffn_multiplier"])
        )
        model = R6SparseFamilyMixturePolicy(config, ontology_path=ontology_path)
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


__all__ = ["R6Policy"]
