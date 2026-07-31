"""Strict binding to the immutable 0019 Epoch 13 Foundation weights."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .model_source.ac_model import ACModelConfig
from .model_source.base_model import IDOnlyConfig
from .model_source.r15_model import R15ModelConfig
from .model_source.source_model import SourceModelConfig
from .model_source.source_r15_model import SourceConditionedR15Policy


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_ROOT = REPOSITORY_ROOT / "archive/pretrained/0019_universal_winner_bc_0730_epoch13"
WEIGHTS_PATH = ARCHIVE_ROOT / "model.pt"
MANIFEST_PATH = ARCHIVE_ROOT / "manifest.json"
ONTOLOGY_PATH = Path(__file__).resolve().parent.parent / "assets/card_ontology.json"
EXPECTED_WEIGHTS_SHA256 = "da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb"
EXPECTED_ONTOLOGY_SHA256 = "8144c63e512a2a00fabaaf2b19cd002c5b59c25fb0763a112256848460113a4d"
EXPECTED_SOURCE_SHA256 = "1f103c179fde4fd471f0b8d765d9edd341886112991a7e35ff92bb5c38b041fd"
EXPECTED_PARAMETER_COUNT = 17_756_162
EXPECTED_EPOCH = 13
EXPECTED_GLOBAL_STEP = 337_194


@dataclass(frozen=True)
class FoundationIdentity:
    asset_id: str
    epoch: int
    global_step: int
    weights_sha256: str
    ontology_sha256: str
    parameter_count: int
    deployment_source_id: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "deployment_source_id": self.deployment_source_id,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "ontology_sha256": self.ontology_sha256,
            "parameter_count": self.parameter_count,
            "weights_sha256": self.weights_sha256,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _model_config() -> tuple[R15ModelConfig, SourceModelConfig]:
    base = IDOnlyConfig(
        max_card_id=2048,
        d_model=320,
        heads=8,
        encoder_layers=4,
        ffn_multiplier=3,
        max_entities=192,
        max_options=128,
        max_action_steps=64,
        dropout=0.1,
    )
    ac = ACModelConfig(
        base=base,
        event_layers=1,
        auxiliary_ffn_multiplier=2,
        goal_roles=4,
    )
    return (
        R15ModelConfig(
            ac=ac,
            scenario_layers=2,
            scenario_ffn_multiplier=3,
            scale_gate_ffn_multiplier=2,
            option_initial_scale=0.35,
        ),
        SourceModelConfig(vocabulary_size=509, initial_scale=0.05),
    )


def verify_foundation() -> FoundationIdentity:
    """Verify immutable data provenance before constructing the model."""
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable Foundation manifest: {MANIFEST_PATH}") from error
    weights_sha = _sha256(WEIGHTS_PATH)
    ontology_sha = _sha256(ONTOLOGY_PATH)
    source_sha = _tree_sha256(Path(__file__).with_name("model_source"))
    checks = {
        "asset_id": (manifest.get("asset_id"), "0019-0730-epoch13"),
        "epoch": (manifest.get("epoch"), EXPECTED_EPOCH),
        "weights manifest SHA": (
            (manifest.get("weights") or {}).get("sha256"), EXPECTED_WEIGHTS_SHA256
        ),
        "weights file SHA": (weights_sha, EXPECTED_WEIGHTS_SHA256),
        "ontology SHA": (ontology_sha, EXPECTED_ONTOLOGY_SHA256),
        "local model source SHA": (source_sha, EXPECTED_SOURCE_SHA256),
        "parameter count": (
            (manifest.get("model") or {}).get("parameter_count"),
            EXPECTED_PARAMETER_COUNT,
        ),
        "neutral source ID": (
            (manifest.get("model") or {}).get("neutral_source_id"), 0
        ),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"Foundation {label} mismatch: {actual!r} != {expected!r}")
    return FoundationIdentity(
        asset_id="0019-0730-epoch13",
        epoch=EXPECTED_EPOCH,
        global_step=EXPECTED_GLOBAL_STEP,
        weights_sha256=weights_sha,
        ontology_sha256=ontology_sha,
        parameter_count=EXPECTED_PARAMETER_COUNT,
    )


def load_foundation(
    device: str | torch.device = "cpu", *, eval_mode: bool = True
) -> tuple[SourceConditionedR15Policy, FoundationIdentity]:
    """Instantiate local frozen code and strictly load archived immutable weights."""
    identity = verify_foundation()
    model_config, source_config = _model_config()
    model = SourceConditionedR15Policy(
        model_config, source_config, ontology_path=ONTOLOGY_PATH
    )
    payload: dict[str, Any] = torch.load(
        WEIGHTS_PATH, map_location="cpu", weights_only=True
    )
    if payload.get("epoch") != identity.epoch:
        raise ValueError("Foundation checkpoint epoch mismatch")
    if payload.get("global_step") != identity.global_step:
        raise ValueError("Foundation checkpoint global step mismatch")
    model.load_state_dict(payload.get("model", {}), strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != identity.parameter_count:
        raise ValueError(f"Foundation parameter count mismatch: {parameter_count}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    if eval_mode:
        model.eval()
    return model, identity


def neutral_source_id(
    batch_size: int, device: str | torch.device = "cpu"
) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return torch.zeros(batch_size, dtype=torch.long, device=device)


__all__ = [
    "FoundationIdentity",
    "load_foundation",
    "neutral_source_id",
    "verify_foundation",
]
