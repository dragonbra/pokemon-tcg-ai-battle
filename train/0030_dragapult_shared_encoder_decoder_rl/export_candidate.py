"""Export a self-contained 0030 decoder checkpoint as an evaluation package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import torch

from evaluation.packages.loader import validate_kaggle_raw_exec

from .checkpoint import load_model_checkpoint
from .policy import load_actor_critic


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_PACKAGE = (
    REPOSITORY_ROOT
    / "evaluation"
    / "arena"
    / "candidates"
    / "0028_v3_latest_dragapult_ex_001_zero_shot"
)

_UNSAFE_PACKAGE_ROOT = "ROOT = Path(__file__).resolve().parent"
_KAGGLE_PACKAGE_ROOT = """ROOT = Path(globals().get("__file__", Path.cwd())).resolve()
if ROOT.is_file():
    ROOT = ROOT.parent
if not (ROOT / "deck.csv").is_file() and Path("/kaggle_simulations/agent/deck.csv").is_file():
    ROOT = Path("/kaggle_simulations/agent")"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _kaggle_compatible_main(source: str) -> str:
    if _KAGGLE_PACKAGE_ROOT in source:
        return source
    if _UNSAFE_PACKAGE_ROOT not in source:
        raise ValueError("source package main.py has an unknown root resolution contract")
    return source.replace(_UNSAFE_PACKAGE_ROOT, _KAGGLE_PACKAGE_ROOT, 1)


def export_candidate(
    *,
    checkpoint: Path,
    output: Path,
    source_package: Path = DEFAULT_SOURCE_PACKAGE,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    source_package = source_package.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not source_package.is_dir():
        raise FileNotFoundError(source_package)
    if output.exists():
        raise FileExistsError(output)

    model, _, foundation = load_actor_critic("cpu")
    decoder_identity = load_model_checkpoint(checkpoint, model)
    base_model_path = source_package / "strategy" / "model.bin"
    base_payload = torch.load(base_model_path, map_location="cpu", weights_only=True)
    if base_payload.get("schema_version") != "0028_model_only_checkpoint_v1":
        raise ValueError("source package is not a 0028 semantic model package")

    metadata = dict(base_payload.get("metadata") or {})
    metadata.update({
        "project_id": "0030_dragapult_shared_encoder_decoder_rl",
        "version": str(decoder_identity["policy_version"]),
        "update": int(decoder_identity["update"]),
        "foundation_checkpoint_sha256": foundation.checkpoint_sha256,
        "decoder_checkpoint_sha256": decoder_identity["checkpoint_sha256"],
        "decoder_sha256": decoder_identity["decoder_sha256"],
        "representation_sha256": model.representation_sha256(),
        "trainable_contract": ["action_decoder.*", "value_head.*"],
        "deployment_contract": "greedy_actor_without_value_head",
    })
    exported_payload = {
        "schema_version": "0028_model_only_checkpoint_v1",
        "state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.actor.state_dict().items()
        },
        "metadata": metadata,
    }

    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copytree(
            source_package,
            staging,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        main_path = staging / "main.py"
        main_source = main_path.read_text(encoding="utf-8").replace(
            "0028 V3 latest zero-shot Dragapult ex evaluation entrypoint.",
            "0030 Dragapult decoder-RL evaluation entrypoint.",
        )
        main_source = _kaggle_compatible_main(main_source)
        main_path.write_text(main_source, encoding="utf-8")

        model_path = staging / "strategy" / "model.bin"
        temporary_model = model_path.with_suffix(".bin.tmp")
        torch.save(exported_payload, temporary_model)
        temporary_model.replace(model_path)

        manifest = {
            "schema_version": "0030_decoder_rl_candidate_v1",
            "candidate": output.name,
            "project_id": "0030_dragapult_shared_encoder_decoder_rl",
            "version": decoder_identity["policy_version"],
            "update": decoder_identity["update"],
            "deck_id": "dragapult_ex_001",
            "deck_sha256": _sha256(staging / "deck.csv"),
            "foundation_checkpoint_sha256": foundation.checkpoint_sha256,
            "decoder_checkpoint": str(checkpoint.relative_to(REPOSITORY_ROOT)),
            "decoder_checkpoint_sha256": decoder_identity["checkpoint_sha256"],
            "decoder_sha256": decoder_identity["decoder_sha256"],
            "representation_sha256": model.representation_sha256(),
            "exported_model_sha256": _sha256(model_path),
            "model_only_source": True,
            "optimizer_state_saved": False,
            "runtime": "0028_canonical_semantic_online_with_0030_decoder_v1",
        }
        _write_json(staging / "manifest.json", manifest)
        deck = [
            int(card_id)
            for card_id in (staging / "deck.csv").read_text(encoding="utf-8").splitlines()
            if card_id.strip()
        ]
        validate_kaggle_raw_exec(main_path, staging, deck)
        staging.replace(output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-package", type=Path, default=DEFAULT_SOURCE_PACKAGE)
    arguments = parser.parse_args()
    print(json.dumps(export_candidate(**vars(arguments)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
