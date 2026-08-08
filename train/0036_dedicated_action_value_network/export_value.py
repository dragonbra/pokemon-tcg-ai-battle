"""Export and load a hash-committed 0036 critic for later PPO initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import torch

from .model.source import SOURCE_CHECKPOINT_SHA256, load_source_policy
from .model.value_network import FrozenEncoderValueNetwork
from .training.checkpoints import load_value_checkpoint


EXPORT_SCHEMA_VERSION = "0036_ppo_value_initialization_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_value_checkpoint(checkpoint: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite 0036 Value export: {output}")
    payload = load_value_checkpoint(checkpoint)
    metadata = payload["metadata"]
    if metadata.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("0036 Value checkpoint was not trained against the admitted frozen Encoder")
    config = metadata.get("value_config")
    if not isinstance(config, dict) or config.get("architecture") not in {
        "raw_pool_mlp", "summary_mlp", "latent_queries"
    }:
        raise ValueError("0036 Value checkpoint has no compatible network config")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        target = stage / "value_head.pt"
        shutil.copy2(checkpoint, target)
        manifest = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "source_checkpoint_required_sha256": SOURCE_CHECKPOINT_SHA256,
            "value_checkpoint_sha256": _sha256(target),
            "value_checkpoint_schema": payload["schema_version"],
            "value_config": config,
            "training_metadata": metadata,
            "ppo_contract": {
                "output": "2*sigmoid(value_logit)-1",
                "use": "baseline_and_gae_initialization",
                "on_policy_recalibration_required": True,
                "value_delta_is_environment_reward": False,
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, output)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def load_exported_value(
    export: Path,
    source_checkpoint: Path,
    device: str | torch.device = "cpu",
) -> FrozenEncoderValueNetwork:
    manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != EXPORT_SCHEMA_VERSION:
        raise ValueError("0036 Value export schema mismatch")
    checkpoint = export / "value_head.pt"
    if _sha256(checkpoint) != manifest.get("value_checkpoint_sha256"):
        raise ValueError("0036 exported Value checkpoint hash mismatch")
    encoder, identity = load_source_policy(source_checkpoint, device)
    if identity.checkpoint_sha256 != manifest.get("source_checkpoint_required_sha256"):
        raise ValueError("0036 exported Value source Encoder mismatch")
    config = manifest["value_config"]
    model = FrozenEncoderValueNetwork(
        encoder,
        architecture=config["architecture"],
        queries=int(config.get("queries", 8)),
        layers=int(config.get("layers", 2)),
        dropout=float(config.get("dropout", 0.0)),
    ).to(device)
    payload = load_value_checkpoint(checkpoint)
    model.value_head.load_state_dict(payload["value_head_state_dict"], strict=True)
    model.eval()
    model.assert_frozen_encoder()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_value_checkpoint(args.checkpoint, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["EXPORT_SCHEMA_VERSION", "export_value_checkpoint", "load_exported_value"]
