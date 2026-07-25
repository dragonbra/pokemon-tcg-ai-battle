from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_candidate(
    template: Path,
    checkpoint: Path,
    output: Path,
    *,
    name: str,
    experiment_id: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"candidate output already exists: {output}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("checkpoint must contain a model state dict")
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be an object")
    category = (metadata.get("stage_two_config") or {}).get("card_category_embedding") or {}
    if category.get("enabled") or any(
        "card_category" in str(key) for key in payload["model"]
    ):
        raise ValueError("the current candidate runtime does not support category augmentation")
    shutil.copytree(template, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    model_path = output / "strategy" / "model.bin"
    inference_payload = {
        "step": int(payload.get("step", 0)),
        "model": payload["model"],
        "metadata": metadata,
    }
    torch.save(inference_payload, model_path)
    manifest_path = output / "strategy" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "name": name,
            "experiment_id": experiment_id,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_step": inference_payload["step"],
            "model_bytes": model_path.stat().st_size,
            "model_sha256": _sha256(model_path),
            "stage_two_config": metadata.get("stage_two_config"),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _main() -> None:
    parser = argparse.ArgumentParser(description="Export a stage-two BC checkpoint candidate")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    manifest = export_candidate(
        args.template,
        args.checkpoint,
        args.output,
        name=args.name,
        experiment_id=args.experiment_id,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    _main()
