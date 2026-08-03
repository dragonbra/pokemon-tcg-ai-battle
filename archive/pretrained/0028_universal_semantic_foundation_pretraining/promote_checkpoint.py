"""Atomically promote a verified 0028 model-only checkpoint into this release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import torch


ARCHIVE_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ARCHIVE_ROOT / "model.pt"
METADATA_PATH = ARCHIVE_ROOT / "checkpoint_metadata.json"
MANIFEST_PATH = ARCHIVE_ROOT / "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("w", encoding="ascii") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_payload(checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(payload) != {"schema_version", "state_dict", "metadata"}:
        raise ValueError("candidate is not a model-only 0028 checkpoint")
    if payload["schema_version"] != "0028_model_only_checkpoint_v1":
        raise ValueError("candidate checkpoint schema is not supported")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("candidate checkpoint metadata is absent")
    expected = {
        "project_id": "0028_universal_semantic_foundation_pretraining",
        "version": "V3_shared_prototype_batch512",
        "arm": "universal_semantic",
        "dataset_manifest_sha256": "7b8bd85a33e756e41212f57ff43a57abd59a0626a47f10fa2101c715c42adc15",
        "implementation_sha256": "c2fac00ae9c9b39bcd28897aea5580168e283e9abeb66845955d84518d09a80b",
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"candidate identity mismatch: {mismatches}")
    return payload


def promote(checkpoint: Path, *, stage: str, criterion: str) -> None:
    checkpoint = checkpoint.resolve(strict=True)
    payload = _validate_payload(checkpoint)
    metadata = payload["metadata"]
    if not stage or not criterion:
        raise ValueError("stage and criterion must be non-empty")

    temporary = WEIGHTS_PATH.with_name("model.pt.tmp")
    try:
        with checkpoint.open("rb") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(WEIGHTS_PATH)
    finally:
        temporary.unlink(missing_ok=True)

    sidecar = {
        "schema_version": payload["schema_version"],
        "release_stage": stage,
        "criterion": criterion,
        "source_checkpoint": str(checkpoint),
        "project_id": metadata["project_id"],
        "version": metadata["version"],
        "arm": metadata["arm"],
        "epoch": metadata["epoch"],
        "global_step": metadata["global_step"],
        "optimizer_state_saved": False,
        "resumable_training_state_saved": False,
        "dataset_manifest_sha256": metadata["dataset_manifest_sha256"],
        "training_config_sha256": metadata["training_config_sha256"],
        "model_contract_sha256": metadata["model_contract_sha256"],
        "implementation_sha256": metadata["implementation_sha256"],
        "validation": metadata["validation"],
    }
    _atomic_json(METADATA_PATH, sidecar)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["release_stage"] = stage
    manifest["epoch"] = int(metadata["epoch"])
    manifest["global_step"] = int(metadata["global_step"])
    manifest["weights"] = {
        "path": "model.pt",
        "git_lfs": True,
        "bytes": WEIGHTS_PATH.stat().st_size,
        "sha256": _sha256(WEIGHTS_PATH),
    }
    manifest["selection"] = {"criterion": criterion, **metadata["validation"]}
    manifest["provenance"]["source_checkpoint"] = str(checkpoint)
    for relative in manifest["file_sha256"]:
        manifest["file_sha256"][relative] = _sha256(ARCHIVE_ROOT / relative)
    _atomic_json(MANIFEST_PATH, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--criterion", required=True)
    args = parser.parse_args()
    promote(args.checkpoint, stage=args.stage, criterion=args.criterion)


if __name__ == "__main__":
    main()
