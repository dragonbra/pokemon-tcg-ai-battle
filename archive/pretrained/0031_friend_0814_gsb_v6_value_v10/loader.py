"""Strict loaders for the paired Policy-0814 BC actor and pretrained Value network."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from model_source.domain import PrototypeIndex
from model_source.model import ModelConfig, SemanticPolicy
from value_source.checkpoints import load_value_checkpoint as _load_value_payload
from value_source.value_network import FrozenEncoderValueNetwork


ARCHIVE_ROOT = Path(__file__).resolve().parent
POLICY_WEIGHTS_PATH = ARCHIVE_ROOT / "model.pt"
VALUE_WEIGHTS_PATH = ARCHIVE_ROOT / "value_head.pt"
PUBLIC_PROTOTYPES = ARCHIVE_ROOT / "assets/official_public_prototypes_v1.json"
ENGINE_PROTOTYPES = ARCHIVE_ROOT / "assets/official_full_engine_prototypes_v2.json"

POLICY_ID = "Policy-0814"
POLICY_PROJECT_ID = "0031_rule_faithful_semantic_foundation_pretraining"
POLICY_VERSION = "V6_medal_zone_gsb_all_days_to_0810"
POLICY_SHA256 = "d7921f420c8f12155119d6caa0fef414f51c0a8368cd5ecf07cd7d71312c897b"
POLICY_PARAMETERS = 56_352_322
POLICY_DATASET_SHA256 = "7dfb2cebc01f70893ad517a21bee8e26a9f665b83e0f0a9012154eb8e4e1eaae"

VALUE_PROJECT_ID = "0036_dedicated_action_value_network"
VALUE_VERSION = "V10_v6_e23_best_val_encoder_latent_value_archetype_diff"
VALUE_SHA256 = "0ad6f57a32d37942cccdc78a8a9c8ef2f6f8784d08ea8ed77ca1513628c77d2d"
VALUE_PARAMETERS = 3_407_389


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy_checkpoint() -> dict[str, Any]:
    if _sha256(POLICY_WEIGHTS_PATH) != POLICY_SHA256:
        raise ValueError("Policy-0814 actor checkpoint SHA-256 mismatch")
    payload = torch.load(POLICY_WEIGHTS_PATH, map_location="cpu", weights_only=True)
    if set(payload) != {"schema_version", "state_dict", "metadata"}:
        raise ValueError("Policy-0814 actor checkpoint is not model-only")
    if payload["schema_version"] != "0031_model_only_checkpoint_v1":
        raise ValueError("unsupported Policy-0814 actor checkpoint schema")
    metadata = payload["metadata"]
    expected = {
        "project_id": POLICY_PROJECT_ID,
        "version": POLICY_VERSION,
        "arm": "rule_faithful_semantic",
        "epoch": 23,
        "global_step": 143198,
        "dataset_manifest_sha256": POLICY_DATASET_SHA256,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Policy-0814 actor checkpoint identity mismatch: {mismatches}")
    return payload


def load_value_checkpoint() -> dict[str, Any]:
    if _sha256(VALUE_WEIGHTS_PATH) != VALUE_SHA256:
        raise ValueError("Policy-0814 Value checkpoint SHA-256 mismatch")
    payload = _load_value_payload(VALUE_WEIGHTS_PATH)
    metadata = payload["metadata"]
    expected = {
        "project_id": VALUE_PROJECT_ID,
        "version": VALUE_VERSION,
        "architecture": "latent_queries",
        "epoch": 13,
        "source_checkpoint_sha256": POLICY_SHA256,
        "optimizer_state_saved": False,
        "resumable_training_state_saved": False,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Policy-0814 Value checkpoint identity mismatch: {mismatches}")
    return payload


def load_policy(
    device: str | torch.device = "cpu",
    *,
    eval_mode: bool = True,
) -> SemanticPolicy:
    payload = load_policy_checkpoint()
    contract = payload["metadata"].get("model_config")
    if not isinstance(contract, dict) or not isinstance(contract.get("config"), dict):
        raise ValueError("Policy-0814 actor checkpoint has no model configuration")
    config = ModelConfig(**contract["config"])
    prototypes = PrototypeIndex.load(PUBLIC_PROTOTYPES, ENGINE_PROTOTYPES)
    model = SemanticPolicy(config, prototypes)
    model.load_state_dict(payload["state_dict"], strict=True)
    declared = int(contract.get("parameter_count", -1))
    actual = sum(parameter.numel() for parameter in model.parameters())
    if declared != POLICY_PARAMETERS or actual != POLICY_PARAMETERS:
        raise ValueError(
            "Policy-0814 actor parameter contract mismatch: "
            f"declared={declared}, actual={actual}"
        )
    model.to(device)
    if eval_mode:
        model.eval()
    return model


def load_value_network(
    device: str | torch.device = "cpu",
) -> FrozenEncoderValueNetwork:
    policy = load_policy(device)
    payload = load_value_checkpoint()
    config = payload["metadata"].get("value_config")
    if not isinstance(config, dict):
        raise ValueError("Policy-0814 Value checkpoint has no network configuration")
    network = FrozenEncoderValueNetwork(
        policy,
        architecture=config["architecture"],
        queries=int(config["queries"]),
        layers=int(config["layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    network.value_head.load_state_dict(payload["value_head_state_dict"], strict=True)
    actual = sum(parameter.numel() for parameter in network.value_head.parameters())
    if actual != VALUE_PARAMETERS:
        raise ValueError(f"Policy-0814 Value parameter contract mismatch: {actual}")
    network.eval()
    network.assert_frozen_encoder()
    return network


def load_pair(
    device: str | torch.device = "cpu",
) -> tuple[SemanticPolicy, FrozenEncoderValueNetwork]:
    """Load independent actor and critic modules bound to identical actor weights."""
    return load_policy(device), load_value_network(device)


load_model = load_policy


__all__ = [
    "load_model",
    "load_pair",
    "load_policy",
    "load_policy_checkpoint",
    "load_value_checkpoint",
    "load_value_network",
]
