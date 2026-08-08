"""Strictly reconstruct the frozen friend-0806 semantic Encoder."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..domain import PrototypeIndex
from .config import ModelConfig
from .policy import SemanticPolicy


SOURCE_CHECKPOINT_SHA256 = "0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8"
SOURCE_PROJECT_ID = "0031_rule_faithful_semantic_foundation_pretraining"
SOURCE_VERSION = "V2_full_winners_bs1024_20260616_20260803"
SOURCE_PARAMETER_COUNT = 56_352_322
SOURCE_EPOCH = 11
SOURCE_GLOBAL_STEP = 141_878
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PROTOTYPES = PROJECT_ROOT / "assets" / "official_public_prototypes_v1.json"
ENGINE_PROTOTYPES = PROJECT_ROOT / "assets" / "official_full_engine_prototypes_v2.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    checkpoint_sha256: str
    schema_version: str
    project_id: str
    version: str
    epoch: int
    global_step: int
    parameter_count: int
    tensor_count: int


def load_source_policy(
    checkpoint: Path,
    device: str | torch.device = "cpu",
) -> tuple[SemanticPolicy, SourceIdentity]:
    digest = file_sha256(checkpoint)
    if digest != SOURCE_CHECKPOINT_SHA256:
        raise ValueError(f"0036 source checkpoint SHA-256 mismatch: {digest}")
    payload: dict[str, Any] = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(payload) != {"schema_version", "state_dict", "metadata"}:
        raise ValueError("0036 source must be a model-only checkpoint")
    if payload["schema_version"] != "0031_model_only_checkpoint_v1":
        raise ValueError("0036 source checkpoint schema mismatch")
    metadata = payload["metadata"]
    expected = {
        "project_id": SOURCE_PROJECT_ID,
        "version": SOURCE_VERSION,
        "epoch": SOURCE_EPOCH,
        "global_step": SOURCE_GLOBAL_STEP,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"0036 source metadata mismatch: {mismatches}")
    model_contract = metadata.get("model_config")
    if not isinstance(model_contract, dict) or not isinstance(model_contract.get("config"), dict):
        raise ValueError("0036 source model config is absent")
    config = ModelConfig(**model_contract["config"])
    prototypes = PrototypeIndex.load(PUBLIC_PROTOTYPES, ENGINE_PROTOTYPES)
    policy = SemanticPolicy(config, prototypes)
    policy.load_state_dict(payload["state_dict"], strict=True)
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    if parameter_count != SOURCE_PARAMETER_COUNT:
        raise ValueError(f"0036 source parameter count mismatch: {parameter_count}")
    policy.requires_grad_(False).eval().to(device)
    identity = SourceIdentity(
        checkpoint_sha256=digest,
        schema_version=str(payload["schema_version"]),
        project_id=str(metadata["project_id"]),
        version=str(metadata["version"]),
        epoch=int(metadata["epoch"]),
        global_step=int(metadata["global_step"]),
        parameter_count=parameter_count,
        tensor_count=len(payload["state_dict"]),
    )
    return policy, identity


__all__ = [
    "ENGINE_PROTOTYPES",
    "PUBLIC_PROTOTYPES",
    "SOURCE_CHECKPOINT_SHA256",
    "SOURCE_PARAMETER_COUNT",
    "SourceIdentity",
    "file_sha256",
    "load_source_policy",
]
