"""Strict binding to the 0025 V4 best-greedy canonical checkpoint."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .features.prototypes import PrototypeIndex
from .model.canonical import CanonicalModelConfig, CanonicalSemanticPolicy


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_PATH = (
    REPOSITORY_ROOT
    / "rl_runs/0025_semantic_foundation_pretraining/versions/"
    "V4_canonical_semantic_foundation/checkpoint/canonical_semantic/"
    "best_greedy_exact.pt"
)
PROTOTYPE_PATH = Path(__file__).resolve().parent / "assets/official_public_prototypes_v1.json"
EXPECTED_CHECKPOINT_SHA256 = "adc4eaeca1e62a28bbd762e8e513212c94941044bbc85673f02bbeadc1865aa3"
EXPECTED_PARAMETER_COUNT = 21_837_082
EXPECTED_EPOCH = 14
EXPECTED_GLOBAL_STEP = 2576
EXPECTED_DATASET_SHA256 = "619311eb3df8c6f1ebef874670363988c26dd7665325b9d6fef8adc4bef4f8cf"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FocalIdentity:
    checkpoint_path: str
    checkpoint_sha256: str
    epoch: int
    global_step: int
    dataset_manifest_sha256: str
    parameter_count: int

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def _payload() -> tuple[dict[str, Any], FocalIdentity]:
    actual_sha = _sha256(CHECKPOINT_PATH)
    if actual_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"0025 best-greedy checkpoint SHA mismatch: {actual_sha}")
    payload = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "state_dict", "metadata"}:
        raise ValueError("0025 checkpoint payload contract mismatch")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("0025 checkpoint metadata is missing")
    checks = {
        "arm": (metadata.get("arm"), "canonical_semantic"),
        "epoch": (metadata.get("epoch"), EXPECTED_EPOCH),
        "global_step": (metadata.get("global_step"), EXPECTED_GLOBAL_STEP),
        "dataset": (metadata.get("dataset_manifest_sha256"), EXPECTED_DATASET_SHA256),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"0025 checkpoint {label} mismatch: {actual!r} != {expected!r}")
    return payload, FocalIdentity(
        checkpoint_path=str(CHECKPOINT_PATH.relative_to(REPOSITORY_ROOT)),
        checkpoint_sha256=actual_sha,
        epoch=EXPECTED_EPOCH,
        global_step=EXPECTED_GLOBAL_STEP,
        dataset_manifest_sha256=EXPECTED_DATASET_SHA256,
        parameter_count=EXPECTED_PARAMETER_COUNT,
    )


def verify_focal_checkpoint() -> FocalIdentity:
    payload, identity = _payload()
    model_contract = payload["metadata"].get("model_config")
    if not isinstance(model_contract, dict) or model_contract.get("parameter_count") != EXPECTED_PARAMETER_COUNT:
        raise ValueError("0025 checkpoint parameter-count metadata mismatch")
    return identity


def load_focal_actor(
    device: str | torch.device = "cpu", *, eval_mode: bool = True
) -> tuple[CanonicalSemanticPolicy, FocalIdentity]:
    payload, identity = _payload()
    model_contract = payload["metadata"].get("model_config")
    if not isinstance(model_contract, dict) or not isinstance(model_contract.get("config"), dict):
        raise ValueError("0025 checkpoint model config is missing")
    config = CanonicalModelConfig(**model_contract["config"])
    actor = CanonicalSemanticPolicy(config, PrototypeIndex.load(PROTOTYPE_PATH))
    actor.load_state_dict(payload["state_dict"], strict=True)
    if sum(parameter.numel() for parameter in actor.parameters()) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("0025 focal actor parameter count mismatch")
    actor.to(device)
    if eval_mode:
        actor.eval()
    return actor, identity


__all__ = [
    "CHECKPOINT_PATH",
    "EXPECTED_CHECKPOINT_SHA256",
    "FocalIdentity",
    "load_focal_actor",
    "verify_focal_checkpoint",
]
