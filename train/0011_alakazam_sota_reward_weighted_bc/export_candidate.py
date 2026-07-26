from __future__ import annotations

import hashlib
import importlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch

MAIN = importlib.import_module("train.0010_alakazam_sota_model.export_candidate").MAIN


MODEL_VERSION = "alakazam_sota_reward_weighted_pointer_bc_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_candidate(checkpoint: Path, source_package: Path, output: Path) -> dict[str, Any]:
    """Export an 0011 checkpoint without weakening the 0010 exporter contract."""
    if output.exists():
        raise FileExistsError(f"candidate already exists: {output}")
    for required in (source_package / "deck.csv", source_package / "cg"):
        if not required.exists():
            raise FileNotFoundError(required)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    metadata = payload.get("metadata") or {}
    if metadata.get("model_version") != MODEL_VERSION:
        raise ValueError("checkpoint is not an Alakazam SOTA reward-weighted policy")
    if not isinstance(metadata.get("model_config"), dict):
        raise ValueError("checkpoint metadata is missing model_config")

    strategy = output / "strategy"
    strategy.mkdir(parents=True)
    shutil.copy2(source_package / "deck.csv", output / "deck.csv")
    shutil.copytree(source_package / "cg", output / "cg")
    (output / "main.py").write_text(MAIN, encoding="utf-8")
    (strategy / "__init__.py").write_text("", encoding="utf-8")
    model_project = Path(__file__).resolve().parents[1] / "0010_alakazam_sota_model"
    shutil.copy2(model_project / "model.py", strategy / "model.py")
    shutil.copy2(model_project / "inference.py", strategy / "inference.py")

    portable = {
        "model": payload["model"],
        "metadata": {
            **metadata,
            "source_checkpoint_sha256": _sha256(checkpoint),
            "portable_checkpoint": True,
        },
    }
    model_path = strategy / "model.bin"
    torch.save(portable, model_path)
    manifest = {
        "schema_version": "alakazam_sota_reward_weighted_candidate_v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "deck_sha256": _sha256(output / "deck.csv"),
        "model_sha256": _sha256(model_path),
        "model_bytes": model_path.stat().st_size,
        "source_package": str(source_package.resolve()),
    }
    (strategy / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
