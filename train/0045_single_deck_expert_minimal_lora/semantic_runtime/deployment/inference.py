"""Strict model-only inference wrapper for the 0031 policy."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from ..domain.prototypes import PrototypeIndex
from ..contracts.batch import DecisionBatch
from ..model import ModelConfig, SemanticPolicy
from .online_runtime import OnlineCausalEncoder, _prototype_paths


PORTABLE_SCHEMA_VERSION = "0031_shared_prototype_candidate_checkpoint_v1"
FP16_STORAGE_SCHEMA_VERSION = (
    "0031_shared_prototype_fp16_storage_candidate_checkpoint_v1"
)
FP16_STORAGE_FP32_RUNTIME_SCHEMA_VERSION = (
    "0031_shared_prototype_fp16_storage_fp32_runtime_candidate_checkpoint_v1"
)
_PROTOTYPE_ALIASES = ("state_encoder.prototypes.", "option_encoder.prototypes.")


def _expanded_portable_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    expanded = dict(state_dict)
    prototype = {
        key.removeprefix("prototype_encoder."): value
        for key, value in state_dict.items()
        if key.startswith("prototype_encoder.")
    }
    if not prototype:
        raise ValueError("portable checkpoint has no canonical prototype encoder")
    if any(key.startswith(_PROTOTYPE_ALIASES) for key in state_dict):
        raise ValueError("portable checkpoint contains redundant prototype aliases")
    for alias in _PROTOTYPE_ALIASES:
        expanded.update(
            {f"{alias}{suffix}": value for suffix, value in prototype.items()}
        )
    return expanded


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
        self.runtime_dtype = torch.float32
        self.encoder: OnlineCausalEncoder | None = None

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Path, deck: Sequence[int]
    ) -> "PortableSemanticPolicy":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if set(payload) != {"schema_version", "state_dict", "metadata"}:
            raise ValueError("checkpoint is not the 0031 model-only contract")
        if payload["schema_version"] not in {
            "0031_model_only_checkpoint_v1",
            PORTABLE_SCHEMA_VERSION,
            FP16_STORAGE_SCHEMA_VERSION,
            FP16_STORAGE_FP32_RUNTIME_SCHEMA_VERSION,
        }:
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
        state_dict = payload["state_dict"]
        if payload["schema_version"] in {
            PORTABLE_SCHEMA_VERSION,
            FP16_STORAGE_SCHEMA_VERSION,
            FP16_STORAGE_FP32_RUNTIME_SCHEMA_VERSION,
        }:
            state_dict = _expanded_portable_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=True)
        policy = cls(model, deck, metadata)
        if payload["schema_version"] == FP16_STORAGE_SCHEMA_VERSION:
            policy.model = policy.model.half().eval()
            policy.actor = policy.model
            policy.runtime_dtype = torch.float16
        return policy

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
        batch = DecisionBatch.from_mapping(self.encoder.encode(observation))
        if self.runtime_dtype is torch.float16:
            batch = DecisionBatch.from_mapping(
                {
                    name: value.half() if value.dtype.is_floating_point else value
                    for name, value in batch.items()
                }
            )
        with torch.inference_mode():
            return self.model.greedy_action(batch)


__all__ = ["PortableSemanticPolicy", "legal_fallback"]
