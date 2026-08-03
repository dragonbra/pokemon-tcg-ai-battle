"""Fail-closed binding to the 0028 V3 epoch-4 model-only checkpoint."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .domain.prototypes import PrototypeIndex
from .model import ModelConfig, SemanticPolicy


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_PATH = REPOSITORY_ROOT / (
    "rl_runs/0028_universal_semantic_foundation_pretraining/versions/"
    "V3_shared_prototype_batch512/checkpoint/universal_semantic/latest.pt"
)
PUBLIC_PROTOTYPES = Path(__file__).with_name("official_public_prototypes_v1.json")
FULL_PROTOTYPES = Path(__file__).with_name("official_full_engine_prototypes_v1.json")
EXPECTED_CHECKPOINT_SHA256 = "5e0a6eea42bf228a9bd977cf56fdd14ad360fbceffcf5c19e2d9fe1b39713d98"
EXPECTED_EPOCH = 4
EXPECTED_GLOBAL_STEP = 63196
EXPECTED_DATASET_SHA256 = "7b8bd85a33e756e41212f57ff43a57abd59a0626a47f10fa2101c715c42adc15"
EXPECTED_PARAMETER_COUNT = 21_837_082


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FoundationIdentity:
    checkpoint_path: str
    checkpoint_sha256: str
    epoch: int
    global_step: int
    dataset_manifest_sha256: str
    parameter_count: int


def _payload() -> tuple[dict[str, Any], FoundationIdentity]:
    actual = _sha256(CHECKPOINT_PATH)
    if actual != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"0028 checkpoint SHA mismatch: {actual}")
    payload = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "state_dict", "metadata"}:
        raise ValueError("0028 checkpoint payload contract mismatch")
    if payload["schema_version"] != "0028_model_only_checkpoint_v1":
        raise ValueError("0028 checkpoint schema mismatch")
    metadata = payload.get("metadata") or {}
    expected = {
        "epoch": EXPECTED_EPOCH,
        "global_step": EXPECTED_GLOBAL_STEP,
        "dataset_manifest_sha256": EXPECTED_DATASET_SHA256,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"0028 checkpoint {key} mismatch")
    contract = metadata.get("model_config") or {}
    if contract.get("parameter_count") != EXPECTED_PARAMETER_COUNT:
        raise ValueError("0028 checkpoint parameter count mismatch")
    identity = FoundationIdentity(
        str(CHECKPOINT_PATH.relative_to(REPOSITORY_ROOT)), actual, EXPECTED_EPOCH,
        EXPECTED_GLOBAL_STEP, EXPECTED_DATASET_SHA256, EXPECTED_PARAMETER_COUNT,
    )
    return payload, identity


def verify_foundation() -> FoundationIdentity:
    return _payload()[1]


def load_foundation(device: str | torch.device = "cpu", *, eval_mode: bool = True):
    payload, identity = _payload()
    config = ModelConfig(**payload["metadata"]["model_config"]["config"])
    prototypes = PrototypeIndex.load(PUBLIC_PROTOTYPES, FULL_PROTOTYPES)
    model = SemanticPolicy(config, prototypes)
    model.load_state_dict(payload["state_dict"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("loaded 0028 parameter count mismatch")
    model.to(device)
    if eval_mode:
        model.eval()
    return model, prototypes, identity


__all__ = ["CHECKPOINT_PATH", "EXPECTED_CHECKPOINT_SHA256", "FoundationIdentity", "load_foundation", "verify_foundation"]
