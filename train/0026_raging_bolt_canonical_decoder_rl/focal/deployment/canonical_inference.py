"""Portable greedy inference for an exported 0026 canonical actor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from ..features.prototypes import PrototypeIndex
from ..model.canonical import CanonicalModelConfig, CanonicalSemanticPolicy
from .canonical_online_runtime import OnlineCausalEncoder, _prototype_path
from .inference import legal_fallback


class PortableCanonicalPolicy:
    requires_source_id = False
    fail_closed_inference_errors = True

    def __init__(self, model: CanonicalSemanticPolicy, deck: Sequence[int]) -> None:
        torch.set_num_threads(1)
        self.model = model.eval()
        self.actor = model
        self.config = model.config
        self.deck = tuple(int(card_id) for card_id in deck)
        if len(self.deck) != 60:
            raise ValueError("deck must contain exactly 60 cards")
        self.encoder: OnlineCausalEncoder | None = None

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Path, deck: Sequence[int]
    ) -> "PortableCanonicalPolicy":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if set(payload) != {"schema_version", "state_dict", "metadata"}:
            raise ValueError("portable actor checkpoint schema mismatch")
        metadata = payload["metadata"]
        if not isinstance(metadata, dict) or metadata.get("arm") != "canonical_semantic":
            raise ValueError("portable actor is not canonical_semantic")
        model_contract = metadata.get("model_config")
        if not isinstance(model_contract, dict) or not isinstance(
            model_contract.get("config"), dict
        ):
            raise ValueError("portable actor has no canonical model config")
        config = CanonicalModelConfig(**model_contract["config"])
        model = CanonicalSemanticPolicy(config, PrototypeIndex.load(_prototype_path()))
        model.load_state_dict(payload["state_dict"], strict=True)
        return cls(model, deck)

    def reset(self) -> None:
        self.encoder = None

    def select(self, observation: dict[str, Any]) -> list[int]:
        actor = (observation.get("current") or {}).get("yourIndex")
        if actor not in (0, 1):
            raise ValueError("observation has no valid actor")
        if self.encoder is None or self.encoder.actor != actor:
            self.encoder = OnlineCausalEncoder(actor, self.deck, self.config)
        try:
            batch = self.encoder.encode(observation)
            with torch.inference_mode():
                return self.model.greedy_action(batch)
        except (IndexError, RuntimeError, ValueError):
            self.encoder = None
            return legal_fallback(observation)


__all__ = ["PortableCanonicalPolicy"]
