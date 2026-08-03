"""Strict loader for the current 0028 pretrained release."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from model_source.domain import PrototypeIndex
from model_source.model import ModelConfig, SemanticPolicy


ARCHIVE_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ARCHIVE_ROOT / "model.pt"
PUBLIC_PROTOTYPES = ARCHIVE_ROOT / "assets/official_public_prototypes_v1.json"
ENGINE_PROTOTYPES = ARCHIVE_ROOT / "assets/official_full_engine_prototypes_v1.json"


def load_checkpoint() -> dict[str, Any]:
    payload = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    if set(payload) != {"schema_version", "state_dict", "metadata"}:
        raise ValueError("0028 archive checkpoint payload is not model-only")
    if payload["schema_version"] != "0028_model_only_checkpoint_v1":
        raise ValueError("unsupported 0028 checkpoint schema")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("0028 checkpoint metadata is absent")
    if metadata.get("project_id") != "0028_universal_semantic_foundation_pretraining":
        raise ValueError("checkpoint project identity does not match this archive")
    if metadata.get("arm") != "universal_semantic":
        raise ValueError("checkpoint is not the universal_semantic arm")
    return payload


def load_model(
    device: str | torch.device = "cpu",
    *,
    eval_mode: bool = True,
) -> SemanticPolicy:
    """Rebuild the bound model and strictly load the released weights."""
    payload = load_checkpoint()
    contract = payload["metadata"].get("model_config")
    if not isinstance(contract, dict) or not isinstance(contract.get("config"), dict):
        raise ValueError("checkpoint has no model configuration")
    config = ModelConfig(**contract["config"])
    prototypes = PrototypeIndex.load(PUBLIC_PROTOTYPES, ENGINE_PROTOTYPES)
    model = SemanticPolicy(config, prototypes)
    model.load_state_dict(payload["state_dict"], strict=True)
    expected = int(contract.get("parameter_count", -1))
    actual = sum(parameter.numel() for parameter in model.parameters())
    if expected != 21_837_082 or actual != expected:
        raise ValueError(f"model parameter contract mismatch: {actual} != {expected}")
    model.to(device)
    if eval_mode:
        model.eval()
    return model


__all__ = ["load_checkpoint", "load_model"]

