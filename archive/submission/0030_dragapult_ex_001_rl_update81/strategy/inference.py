"""Portable inference contract for the 0028 semantic policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from .domain.prototypes import PrototypeIndex
from .model import ModelConfig, SemanticPolicy
from .online_runtime import OnlineCausalEncoder


def legal_fallback(observation: dict[str, Any]) -> list[int]:
    select = observation.get("select") or {}
    options = select.get("option")
    if not isinstance(options, list):
        raise ValueError("select.option is not a list")
    minimum = int(select.get("minCount", 0))
    maximum = int(select.get("maxCount", len(options)))
    if not 0 <= minimum <= maximum <= len(options):
        raise ValueError("invalid selection bounds")
    return list(range(minimum))


class SemanticPortablePolicy:
    """Model-only 0028 checkpoint wrapper used by both local and shared GPU eval."""

    requires_source_id = False
    fail_closed_inference_errors = False

    def __init__(
        self,
        model: SemanticPolicy,
        deck: Sequence[int],
        prototypes: PrototypeIndex,
        metadata: dict[str, Any],
    ) -> None:
        torch.set_num_threads(1)
        self.model = model.eval()
        self.config = model.config
        self.deck = tuple(int(card_id) for card_id in deck)
        self.prototypes = prototypes
        self.metadata = dict(metadata)
        self.encoder: OnlineCausalEncoder | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path,
        public_prototypes: Path,
        full_engine_prototypes: Path,
        deck: Sequence[int],
    ) -> "SemanticPortablePolicy":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != "0028_model_only_checkpoint_v1":
            raise ValueError("unsupported 0028 checkpoint schema")
        metadata = payload.get("metadata") or {}
        config_payload = ((metadata.get("model_config") or {}).get("config") or {})
        config = ModelConfig(**config_payload)
        prototypes = PrototypeIndex.load(public_prototypes, full_engine_prototypes)
        model = SemanticPolicy(config, prototypes)
        model.load_state_dict(payload["state_dict"], strict=True)
        return cls(model, deck, prototypes, dict(metadata))

    def reset(self) -> None:
        self.encoder = None

    def select(self, observation: dict[str, Any]) -> list[int]:
        actor_index = (observation.get("current") or {}).get("yourIndex")
        if actor_index not in (0, 1):
            raise ValueError("observation has no valid actor")
        select = observation.get("select") or {}
        options = select.get("option") or []
        minimum = int(select.get("minCount", 0))
        if len(options) > self.config.max_options or minimum > self.config.max_action_steps:
            return legal_fallback(observation)
        if self.encoder is None or self.encoder.actor != actor_index:
            self.encoder = OnlineCausalEncoder(actor_index, self.deck, self.config, self.prototypes)
        try:
            batch = self.encoder.encode(observation)
            with torch.inference_mode():
                return self.model.greedy_action(batch)
        except (IndexError, RuntimeError, ValueError):
            self.encoder = None
            return legal_fallback(observation)


__all__ = ["SemanticPortablePolicy", "legal_fallback"]
