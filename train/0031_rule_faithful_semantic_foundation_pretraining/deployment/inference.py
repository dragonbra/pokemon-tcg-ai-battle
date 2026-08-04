"""Strict model-only inference wrapper for the 0031 policy."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from ..domain.prototypes import PrototypeIndex
from ..model import ModelConfig, SemanticPolicy
from .online_runtime import OnlineCausalEncoder, _prototype_paths


def legal_fallback(observation: dict[str, Any]) -> list[int]:
    select = observation.get("select") or {}
    options = select.get("option")
    if not isinstance(options, list):
        raise ValueError("select.option is not a list")
    minimum = select.get("minCount", 0)
    maximum = select.get("maxCount", len(options))
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or not 0 <= minimum <= maximum <= len(options)
    ):
        raise ValueError("invalid selection bounds")
    return list(range(minimum))


class PortableSemanticPolicy:
    requires_source_id = False
    fail_closed_inference_errors = True

    def __init__(
        self,
        model: SemanticPolicy,
        deck: Sequence[int],
        metadata: dict[str, Any],
    ) -> None:
        torch.set_num_threads(1)
        self.model = model.eval()
        self.actor = self.model
        self.config = model.config
        self.deck = tuple(int(card_id) for card_id in deck)
        if len(self.deck) != 60 or any(card_id <= 0 for card_id in self.deck):
            raise ValueError("deck must contain exactly 60 positive card IDs")
        self.metadata = dict(metadata)
        self.encoder: OnlineCausalEncoder | None = None

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Path, deck: Sequence[int]
    ) -> "PortableSemanticPolicy":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if set(payload) != {"schema_version", "state_dict", "metadata"}:
            raise ValueError("checkpoint is not the 0031 model-only contract")
        if payload["schema_version"] != "0031_model_only_checkpoint_v1":
            raise ValueError("unsupported 0031 checkpoint schema")
        metadata = payload["metadata"]
        if (
            not isinstance(metadata, dict)
            or metadata.get("project_id")
            != "0031_rule_faithful_semantic_foundation_pretraining"
            or metadata.get("arm") != "rule_faithful_semantic"
        ):
            raise ValueError("checkpoint is not the 0031 rule-faithful semantic arm")
        contract = metadata.get("model_config")
        config_payload = contract.get("config") if isinstance(contract, dict) else None
        if not isinstance(config_payload, dict) or contract.get("model") != "SemanticPolicy":
            raise ValueError("checkpoint has no 0031 SemanticPolicy config")
        config = ModelConfig(**config_payload)
        public_path, engine_path = _prototype_paths()
        model = SemanticPolicy(
            config, PrototypeIndex.load(public_path, engine_path)
        )
        model.load_state_dict(payload["state_dict"], strict=True)
        return cls(model, deck, metadata)

    def reset(self) -> None:
        self.encoder = None

    def online_encoder(
        self, actor: int, deck: Sequence[int] | None = None
    ) -> OnlineCausalEncoder:
        return OnlineCausalEncoder(actor, deck or self.deck, self.config)

    def select(self, observation: dict[str, Any]) -> list[int]:
        actor = (observation.get("current") or {}).get("yourIndex")
        if actor not in (0, 1):
            raise ValueError("observation has no valid actor")
        if self.encoder is None or self.encoder.actor != actor:
            self.encoder = self.online_encoder(actor)
        batch = self.encoder.encode(observation)
        with torch.inference_mode():
            return self.model.greedy_action(batch)


__all__ = ["PortableSemanticPolicy", "legal_fallback"]
