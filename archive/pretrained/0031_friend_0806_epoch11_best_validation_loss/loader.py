"""Strict loader for the external 0031 friend-0806 pretrained release."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from model_source.domain import PrototypeIndex
from model_source.model import ModelConfig, SemanticPolicy


ARCHIVE_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ARCHIVE_ROOT / "model.pt"
PUBLIC_PROTOTYPES = ARCHIVE_ROOT / "assets/official_public_prototypes_v1.json"
ENGINE_PROTOTYPES = ARCHIVE_ROOT / "assets/official_full_engine_prototypes_v2.json"
PROJECT_ID = "0031_rule_faithful_semantic_foundation_pretraining"
REPOSITORY_VERSION = "V2_full_winners_bs1024_20260616_20260803"
EXPECTED_PARAMETERS = 56_352_322


def load_checkpoint() -> dict[str, Any]:
    payload = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    if set(payload) != {"schema_version", "state_dict", "metadata"}:
        raise ValueError("0031 archive checkpoint payload is not model-only")
    if payload["schema_version"] != "0031_model_only_checkpoint_v1":
        raise ValueError("unsupported 0031 checkpoint schema")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("0031 checkpoint metadata is absent")
    expected = {
        "project_id": PROJECT_ID,
        "version": REPOSITORY_VERSION,
        "arm": "rule_faithful_semantic",
        "epoch": 11,
        "global_step": 141878,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"0031 checkpoint identity mismatch: {mismatches}")
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
    declared = int(contract.get("parameter_count", -1))
    actual = sum(parameter.numel() for parameter in model.parameters())
    if declared != EXPECTED_PARAMETERS or actual != EXPECTED_PARAMETERS:
        raise ValueError(
            f"model parameter contract mismatch: declared={declared}, actual={actual}"
        )
    model.to(device)
    if eval_mode:
        model.eval()
    return model


__all__ = ["load_checkpoint", "load_model"]
