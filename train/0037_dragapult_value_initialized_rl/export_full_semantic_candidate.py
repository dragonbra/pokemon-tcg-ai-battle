"""Export an adapted 0037 actor as a self-contained semantic candidate."""

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
BASE_SCHEMA = "0031_model_only_checkpoint_v1"
BASE_SHA256 = "0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8"
BASE_CHECKPOINT = (
    ROOT / "rl_runs/0037_dragapult_value_initialized_rl/source/friend_0806_epoch11/model.pt"
)
RL_SCHEMA = "0037_value_initialized_adapted_model_only_v2"
EXPECTED_VERSION = "V5_last_option_qv_lora_r4_eval5_50u"
_PROTOTYPE_ALIASES = ("state_encoder.prototypes.", "option_encoder.prototypes.")
_LORA_MODULES = ("self_attn", "multihead_attn")


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


def _merge_qv_weight(
    base: torch.Tensor,
    *,
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    v_a: torch.Tensor,
    v_b: torch.Tensor,
    alpha: float,
    rank: int,
) -> torch.Tensor:
    if base.ndim != 2 or base.shape[0] % 3:
        raise ValueError("merged attention base must have three equal Q/K/V row groups")
    width = base.shape[0] // 3
    columns = base.shape[1]
    expected_a = (rank, columns)
    expected_b = (width, rank)
    if (
        rank < 1
        or alpha <= 0
        or tuple(q_a.shape) != expected_a
        or tuple(v_a.shape) != expected_a
        or tuple(q_b.shape) != expected_b
        or tuple(v_b.shape) != expected_b
    ):
        raise ValueError("LoRA Q/V factor shapes do not match merged attention weight")
    merged = base.clone()
    scale = alpha / rank
    merged[:width].add_((q_b.float() @ q_a.float()).to(base.dtype), alpha=scale)
    merged[2 * width :].add_((v_b.float() @ v_a.float()).to(base.dtype), alpha=scale)
    return merged


def _canonical_base_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = payload.get("state_dict")
    if (
        payload.get("schema_version") != BASE_SCHEMA
        or not isinstance(state, dict)
        or len(state) != 293
    ):
        raise ValueError("0037 base checkpoint does not match the audited 0806 actor")
    canonical = {
        name: value.clone()
        for name, value in state.items()
        if not name.startswith(_PROTOTYPE_ALIASES)
    }
    prototype_keys = {
        name.removeprefix("prototype_encoder.")
        for name in canonical
        if name.startswith("prototype_encoder.")
    }
    for alias in _PROTOTYPE_ALIASES:
        alias_keys = {name.removeprefix(alias) for name in state if name.startswith(alias)}
        if alias_keys != prototype_keys:
            raise ValueError(f"prototype alias inventory differs: {alias}")
        for suffix in prototype_keys:
            if not torch.equal(
                state[f"prototype_encoder.{suffix}"], state[f"{alias}{suffix}"]
            ):
                raise ValueError(f"prototype alias tensor differs: {alias}{suffix}")
    return canonical


def _checkpoint_sidecar(path: Path) -> str:
    digest = _sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().strip() != digest:
        raise ValueError("RL checkpoint SHA-256 sidecar mismatch")
    return digest


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
    metadata = rl_payload.get("metadata") or {}
    adaptation = rl_payload.get("adaptation") or {}
    expected_adaptation = {
        "lora": True,
        "layernorm_tuning": False,
        "rank": 4,
        "alpha": 8.0,
        "option_block": 1,
    }
    if (
        rl_payload.get("schema_version") != RL_SCHEMA
        or metadata.get("project") != PROJECT_ID
        or metadata.get("version") != EXPECTED_VERSION
        or adaptation != expected_adaptation
    ):
        raise ValueError("checkpoint is not the audited 0037 V5 Last Option Q/V LoRA arm")
    checkpoint_sha256 = _checkpoint_sidecar(checkpoint)
    if _sha256(BASE_CHECKPOINT) != BASE_SHA256:
        raise ValueError("0037 audited 0806 base checkpoint SHA-256 mismatch")
    base_payload = _load_payload(BASE_CHECKPOINT)

    source_state = source_payload.get("state_dict")
    rl_state = rl_payload.get("state_dict")
    if not isinstance(source_state, dict) or not isinstance(rl_state, dict):
        raise ValueError("checkpoint has no state_dict mapping")
    merged_state = _canonical_base_state(base_payload)
    if set(merged_state) != set(source_state):
        raise ValueError("audited fp16 source package tensor inventory differs from 0806 base")

    decoder_names = {name for name in source_state if name.startswith("action_decoder.")}
    rl_decoder_names = {f"actor.{name}" for name in decoder_names}
    lora_names: set[str] = set()
    for module_name in _LORA_MODULES:
        prefix = (
            "actor.option_encoder.cross_attention_transformer.layers.1."
            f"{module_name}.parametrizations.in_proj_weight.0."
        )
        lora_names.update(prefix + suffix for suffix in ("q_a", "q_b", "v_a", "v_b"))
    value_names = {name for name in rl_state if name.startswith("value_head.")}
    if set(rl_state) != rl_decoder_names | lora_names | value_names or not value_names:
        raise ValueError("adapted checkpoint trainable tensor inventory is incomplete or unexpected")

    for name in decoder_names:
        value = rl_state[f"actor.{name}"]
        if value.shape != merged_state[name].shape:
            raise ValueError(f"decoder shape mismatch: {name}")
        merged_state[name] = value.clone()

    for module_name in _LORA_MODULES:
        base_name = (
            "option_encoder.cross_attention_transformer.layers.1."
            f"{module_name}.in_proj_weight"
        )
        prefix = (
            "actor.option_encoder.cross_attention_transformer.layers.1."
            f"{module_name}.parametrizations.in_proj_weight.0."
        )
        merged_state[base_name] = _merge_qv_weight(
            merged_state[base_name],
            q_a=rl_state[prefix + "q_a"],
            q_b=rl_state[prefix + "q_b"],
            v_a=rl_state[prefix + "v_a"],
            v_b=rl_state[prefix + "v_b"],
            alpha=float(adaptation["alpha"]),
            rank=int(adaptation["rank"]),
        )

    portable_state = {
        name: value.half() if torch.is_floating_point(value) else value
        for name, value in merged_state.items()
    }

    portable_metadata = dict(source_payload["metadata"])
    portable_metadata["rl_finetune"] = {
        "project_id": PROJECT_ID,
        "version": metadata.get("version"),
        "checkpoint_update": rl_payload.get("update"),
        "checkpoint_sha256": checkpoint_sha256,
        "source_policy_update": metadata.get("source_policy_update"),
        "adaptation": adaptation,
        "trainable_contract": [
            "actor.action_decoder.*",
            "actor.option_encoder.last_block.self_attn.qv_lora",
            "actor.option_encoder.last_block.cross_attn.qv_lora",
            "value_head.*",
        ],
        "deployed_contract": [
            "action_decoder.*",
            "option_encoder.last_block.self_attn.qv_merged",
            "option_encoder.last_block.cross_attn.qv_merged",
        ],
        "critic_deployed": False,
    }
    portable = {
        "schema_version": OUTPUT_SCHEMA,
        "state_dict": portable_state,
        "metadata": portable_metadata,
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
            "schema_version": "0037_adapted_full_semantic_rl_candidate_v1",
            "candidate": output.name,
            "project_id": PROJECT_ID,
            "version": metadata.get("version"),
            "deck_id": "dragapult_ex_007",
            "deck_display_name": "007 · Dragapult ex",
            "deck_sha256": _deck_hash(deck),
            "source_candidate": str(source.relative_to(ROOT)),
            "source_portable_checkpoint_sha256": _sha256(source_model_path),
            "base_checkpoint": str(BASE_CHECKPOINT.relative_to(ROOT)),
            "base_checkpoint_sha256": BASE_SHA256,
            "rl_checkpoint": str(checkpoint.relative_to(ROOT)),
            "rl_checkpoint_sha256": checkpoint_sha256,
            "checkpoint_update": rl_payload.get("update"),
            "source_policy_update": metadata.get("source_policy_update"),
            "adaptation": adaptation,
            "adapter_deployment": "merged_into_last_option_self_and_cross_attention_qv",
            "portable_checkpoint_schema_version": portable["schema_version"],
            "portable_checkpoint_sha256": _sha256(output / "strategy/model.bin"),
            "storage_dtype": "fp16",
            "runtime_dtype": "fp32",
            "evaluation_inference_device": "cuda:0",
            "model_only": True,
            "optimizer_state_saved": False,
            "critic_deployed": False,
            "selection": "update32 official seeded512 frozen greedy 305-207 (59.57%)",
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
