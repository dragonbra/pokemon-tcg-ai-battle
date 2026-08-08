"""Export a V3 decoder checkpoint as a self-contained semantic candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ID = "0037_dragapult_value_initialized_rl"
SOURCE_SCHEMA = "0031_shared_prototype_fp16_storage_candidate_checkpoint_v1"
OUTPUT_SCHEMA = "0031_shared_prototype_fp16_storage_fp32_runtime_candidate_checkpoint_v1"
SOURCE_SCHEMAS = frozenset({SOURCE_SCHEMA, OUTPUT_SCHEMA})
RL_SCHEMA = "0034_full_semantic_decoder_value_model_only_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deck_hash(deck: list[int]) -> str:
    payload = ",".join(str(card_id) for card_id in sorted(deck)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return payload


def export_candidate(*, source: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    source = source.resolve()
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source_model_path = source / "strategy/model.bin"
    source_payload = _load_payload(source_model_path)
    rl_payload = _load_payload(checkpoint)
    if source_payload.get("schema_version") not in SOURCE_SCHEMAS:
        raise ValueError("source candidate is not the audited fp16 Large Model 0806 package")
    if rl_payload.get("schema_version") != RL_SCHEMA:
        raise ValueError("checkpoint is not a V3 full-semantic model-only checkpoint")

    source_state = source_payload.get("state_dict")
    rl_state = rl_payload.get("state_dict")
    if not isinstance(source_state, dict) or not isinstance(rl_state, dict):
        raise ValueError("checkpoint has no state_dict mapping")
    decoder_state = {
        name: value for name, value in rl_state.items() if name.startswith("action_decoder.")
    }
    expected = {name for name in source_state if name.startswith("action_decoder.")}
    if set(decoder_state) != expected:
        raise ValueError(
            f"decoder tensor mismatch: missing={sorted(expected - set(decoder_state))}, "
            f"unexpected={sorted(set(decoder_state) - expected)}"
        )

    merged_state = dict(source_state)
    for name, value in decoder_state.items():
        if value.shape != source_state[name].shape:
            raise ValueError(f"decoder shape mismatch: {name}")
        merged_state[name] = value.to(dtype=source_state[name].dtype)

    metadata = dict(source_payload["metadata"])
    metadata["rl_finetune"] = {
        "project_id": PROJECT_ID,
        "version": rl_payload.get("metadata", {}).get("version"),
        "checkpoint_update": rl_payload.get("update"),
        "checkpoint_sha256": _sha256(checkpoint),
        "source_policy_update": rl_payload.get("metadata", {}).get("source_policy_update"),
        "trainable_contract": ["actor.action_decoder.*", "value_head.*"],
        "deployed_contract": ["actor.action_decoder.*"],
    }
    portable = {
        "schema_version": OUTPUT_SCHEMA,
        "state_dict": merged_state,
        "metadata": metadata,
    }

    try:
        shutil.copytree(
            source,
            output,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "manifest.json"),
        )
        torch.save(portable, output / "strategy/model.bin")
        deck = [
            int(line)
            for line in (output / "deck.csv").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(deck) != 60:
            raise ValueError("candidate deck is not exactly 60 cards")
        manifest = {
            "schema_version": "0034_full_semantic_rl_candidate_v1",
            "candidate": output.name,
            "project_id": PROJECT_ID,
            "version": rl_payload.get("metadata", {}).get("version"),
            "deck_id": "dragapult_third_ptcg_club",
            "deck_display_name": "Dragapult Third PTCG Club",
            "deck_sha256": _deck_hash(deck),
            "source_candidate": str(source.relative_to(ROOT)),
            "source_portable_checkpoint_sha256": _sha256(source_model_path),
            "rl_checkpoint": str(checkpoint.relative_to(ROOT)),
            "rl_checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_update": rl_payload.get("update"),
            "source_policy_update": rl_payload.get("metadata", {}).get(
                "source_policy_update"
            ),
            "portable_checkpoint_schema_version": portable["schema_version"],
            "portable_checkpoint_sha256": _sha256(output / "strategy/model.bin"),
            "storage_dtype": "fp16",
            "runtime_dtype": "fp32",
            "evaluation_inference_device": "cuda:0",
            "model_only": True,
            "optimizer_state_saved": False,
            "critic_deployed": False,
            "selection": "highest periodic frozen greedy win rate; latest checkpoint on ties",
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if any(path.is_symlink() for path in output.rglob("*")):
            raise ValueError("candidate package contains a symlink")
        return manifest
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT
        / "evaluation/arena/candidates/0034_dragapult_third_large_model_zero_shot",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(export_candidate(**vars(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
