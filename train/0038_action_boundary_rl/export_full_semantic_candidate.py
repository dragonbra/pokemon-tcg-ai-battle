"""Export a 0038 compound-action checkpoint as a self-contained candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import torch

from .action_boundary.contracts import (
    ACTION_BOUNDARY_SCHEMA_VERSION,
    CANONICALIZER_VERSION,
    DECISION_GATE_VERSION,
    FEATURE_PREPROCESSING_VERSION,
    OFFICIAL_PROTOCOL_ADAPTER_VERSION,
    TRAJECTORY_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ID = "0038_action_boundary_rl"
SOURCE_SCHEMA = "0031_shared_prototype_fp16_storage_candidate_checkpoint_v1"
SOURCE_RUNTIME_SCHEMA = (
    "0031_shared_prototype_fp16_storage_fp32_runtime_candidate_checkpoint_v1"
)
OUTPUT_SCHEMA = "0038_compound_kaggle_candidate_v4"
SOURCE_SCHEMAS = frozenset({SOURCE_SCHEMA, SOURCE_RUNTIME_SCHEMA})
BASE_SCHEMA = "0031_model_only_checkpoint_v1"
BASE_SHA256 = "0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8"
BASE_CHECKPOINT = (
    ROOT / "rl_runs/0037_dragapult_value_initialized_rl/source/friend_0806_epoch11/model.pt"
)
RL_SCHEMA = "0038_model_only_checkpoint_v1"
RUNTIME_ROOT = ROOT / "train/0038_action_boundary_rl/kaggle_runtime"
ACTION_BOUNDARY_ROOT = ROOT / "train/0038_action_boundary_rl/action_boundary"
SEMANTIC_RUNTIME_ROOT = ROOT / "train/0038_action_boundary_rl/semantic_policy"
FOCAL_DECK_ID = "dragapult_ex_07bedfffbfad"
FOCAL_DECK_SHA256 = "07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725"
FOCAL_DECK_PATH = (
    ROOT / "train/0038_action_boundary_rl/league/decks" / FOCAL_DECK_ID / "deck.csv"
)
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


def _package_file_hashes(root: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.name == "manifest.json"
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
        ):
            continue
        rows[str(path.relative_to(root))] = _sha256(path)
    if not rows:
        raise ValueError("candidate package file inventory is empty")
    return rows


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


def _frozen_selection(checkpoint: Path, update: int) -> dict[str, Any]:
    result_path = (
        checkpoint.parent.parent
        / "artifact/frozen_results"
        / f"core-update-{update:06d}.json"
    )
    if not result_path.is_file():
        raise ValueError(f"missing canonical Frozen evaluation: {result_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    result_schema = payload.get("schema_version")
    if (
        result_schema not in {
            "0038_frozen_per_game_results_v1",
            "0038_frozen_per_game_results_v2",
        }
        or payload.get("checkpoint_update") != update
        or payload.get("frozen_panel_version")
        != "frozen_0806_seeded_agent_first_player_v3"
        or not isinstance(entries, list)
        or len(entries) != 2048
    ):
        raise ValueError("canonical Frozen evaluation contract mismatch")
    valid = [
        row for row in entries
        if row.get("valid") is True and row.get("error") in (None, "")
    ]
    if len(valid) != 2048:
        raise ValueError("canonical Frozen evaluation contains invalid/error games")
    if result_schema == "0038_frozen_per_game_results_v2":
        malformed = [
            row for row in valid
            if (
                not isinstance(row.get("chance_boundary"), bool)
                or not isinstance(row.get("semantic_fallback"), bool)
                or bool(row.get("fallback")) != bool(
                    row.get("chance_boundary") or row.get("semantic_fallback")
                )
                or bool(row.get("chance_boundary")) != (
                    row.get("fallback_reason") == "chance_boundary_before_allocation"
                )
            )
        ]
        if malformed:
            raise ValueError("canonical Frozen v2 chance/fallback evidence is malformed")
        semantic_fallbacks = sum(bool(row["semantic_fallback"]) for row in valid)
        if semantic_fallbacks:
            raise ValueError("canonical Frozen evaluation contains semantic fallback")
        chance_boundaries = sum(bool(row["chance_boundary"]) for row in valid)
    else:
        semantic_fallbacks = sum(bool(row.get("fallback")) for row in valid)
        chance_boundaries = 0
    wins = sum(row.get("outcome") == 1 for row in valid)
    losses = len(valid) - wins
    first = [row for row in valid if row.get("focal_first") is True]
    second = [row for row in valid if row.get("focal_first") is False]
    if len(first) + len(second) != len(valid) or not first or not second:
        raise ValueError(
            "canonical Frozen evaluation lacks actual Agent-selected turn-order evidence"
        )
    return {
        "result": str(result_path.relative_to(ROOT)),
        "result_sha256": _sha256(result_path),
        "result_schema": result_schema,
        "panel_version": payload["frozen_panel_version"],
        "episodes": len(valid),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(valid),
        "first_wins": sum(row.get("outcome") == 1 for row in first),
        "second_wins": sum(row.get("outcome") == 1 for row in second),
        "chance_boundaries": chance_boundaries,
        "semantic_fallbacks": semantic_fallbacks,
    }


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
    flags = rl_payload.get("integrated_flags") or {}
    checkpoint_update = int(rl_payload.get("update", -1))
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
        or metadata.get("version") != checkpoint.parent.parent.name
        or adaptation != expected_adaptation
        or not flags.get("enable_action_boundary")
        or not flags.get("enable_forced_shortcut")
        or not flags.get("enable_dragapult_macro")
    ):
        raise ValueError("checkpoint is not an audited 0038 compound policy")
    checkpoint_sha256 = _checkpoint_sidecar(checkpoint)
    frozen = _frozen_selection(checkpoint, checkpoint_update)
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
    meta_enabled = bool(flags.get("enable_opponent_meta_conditioning"))
    deployed_prefixes = (
        "allocation_head.", "opponent_meta_head.", "opponent_meta_conditioner."
    )
    value_names = {name for name in rl_state if name.startswith("value_head.")}
    ignored_prefixes = ("prize_aux.",)
    deployed_names = {name for name in rl_state if name.startswith(deployed_prefixes)}
    ignored_names = {name for name in rl_state if name.startswith(ignored_prefixes)}
    if (
        set(rl_state) != (
            rl_decoder_names | lora_names | value_names | deployed_names | ignored_names
        )
        or not value_names
        or not any(name.startswith("allocation_head.") for name in deployed_names)
        or (meta_enabled != any(name.startswith("opponent_meta_head.") for name in deployed_names))
        or (meta_enabled != any(name.startswith("opponent_meta_conditioner.") for name in deployed_names))
    ):
        raise ValueError("0038 trainable tensor inventory is incomplete or unexpected")

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
            merged_state[base_name].float(),
            q_a=rl_state[prefix + "q_a"],
            q_b=rl_state[prefix + "q_b"],
            v_a=rl_state[prefix + "v_a"],
            v_b=rl_state[prefix + "v_b"],
            alpha=float(adaptation["alpha"]),
            rank=int(adaptation["rank"]),
        )

    portable_actor_state = {
        name: value.half() if torch.is_floating_point(value) else value
        for name, value in merged_state.items()
    }

    def portable_substate(prefix: str) -> dict[str, torch.Tensor]:
        return {
            name.removeprefix(prefix): (
                value.half() if torch.is_floating_point(value) else value
            )
            for name, value in rl_state.items() if name.startswith(prefix)
        }

    portable_metadata = {
        "project_id": PROJECT_ID,
        "version": metadata.get("version"),
        "checkpoint_update": checkpoint_update,
        "checkpoint_sha256": checkpoint_sha256,
        "source_policy_update": metadata.get("source_policy_update"),
        "actor_metadata": source_payload["metadata"],
        "opponent_meta_class_count": int(flags["opponent_meta_class_count"]),
        "enable_opponent_meta_conditioning": meta_enabled,
        "action_schema_version": ACTION_BOUNDARY_SCHEMA_VERSION,
        "decision_gate_version": DECISION_GATE_VERSION,
        "canonicalizer_version": CANONICALIZER_VERSION,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "official_protocol_adapter_version": OFFICIAL_PROTOCOL_ADAPTER_VERSION,
        "feature_preprocessing_version": FEATURE_PREPROCESSING_VERSION,
        "source_checkpoint_contracts": {
            key: metadata.get(key) for key in (
                "action_schema_version", "decision_gate_version",
                "canonicalizer_version", "trajectory_schema_version",
                "official_protocol_adapter_version",
            )
        },
        "inference_contract": "greedy_root_plus_allocation_with_primitive_expansion",
    }
    portable = {
        "schema_version": OUTPUT_SCHEMA,
        "actor_state_dict": portable_actor_state,
        "value_head_state_dict": portable_substate("value_head."),
        "allocation_head_state_dict": portable_substate("allocation_head."),
        "opponent_meta_head_state_dict": portable_substate("opponent_meta_head."),
        "opponent_meta_conditioner_state_dict": portable_substate(
            "opponent_meta_conditioner."
        ),
        "metadata": portable_metadata,
    }

    try:
        shutil.copytree(
            source,
            output,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "manifest.json",
            ),
        )
        # The source candidate is only the audited deck/cg/weight provenance.
        # Deploy the project-local 0038 semantic runtime rather than inheriting
        # the stale 0034 package copy.  In particular, this preserves the frozen
        # prototype embedding cache used by training and resident evaluation.
        shutil.rmtree(output / "strategy")
        shutil.copytree(
            SEMANTIC_RUNTIME_ROOT,
            output / "strategy",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        # The historical portable actor package carries a one-card-different
        # Dragapult list.  The RL checkpoint was trained and Frozen-evaluated
        # with the versioned 007 deck below, so the deployment deck must come
        # from that authority rather than from the actor source package.
        shutil.copy2(FOCAL_DECK_PATH, output / "deck.csv")
        shutil.copy2(RUNTIME_ROOT / "main.py", output / "main.py")
        shutil.copy2(
            RUNTIME_ROOT / "compound_inference.py",
            output / "strategy/deployment/compound_inference.py",
        )
        shutil.copy2(
            RUNTIME_ROOT / "value_network.py",
            output / "strategy/deployment/value_network.py",
        )
        action_boundary = output / "strategy/action_boundary"
        action_boundary.mkdir()
        (action_boundary / "__init__.py").write_text("", encoding="utf-8")
        for name in (
            "decision_gate.py", "dragapult.py", "macro_planner.py", "macro_protocol.py",
            "public_card_features.py", "contracts.py",
        ):
            shutil.copy2(ACTION_BOUNDARY_ROOT / name, action_boundary / name)
        torch.save(portable, output / "strategy/model.bin")
        deck = [
            int(line)
            for line in (output / "deck.csv").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(deck) != 60:
            raise ValueError("candidate deck is not exactly 60 cards")
        deck_sha256 = _deck_hash(deck)
        if deck_sha256 != FOCAL_DECK_SHA256:
            raise ValueError("candidate deck does not match the Frozen 007 exact deck")
        manifest = {
            "schema_version": OUTPUT_SCHEMA,
            "candidate": output.name,
            "project_id": PROJECT_ID,
            "version": metadata.get("version"),
            "deck_id": FOCAL_DECK_ID,
            "deck_display_name": "007 · Dragapult ex",
            "deck_sha256": deck_sha256,
            "deck_file_sha256": _sha256(output / "deck.csv"),
            "deck_source": str(FOCAL_DECK_PATH.relative_to(ROOT)),
            "source_candidate": str(source.relative_to(ROOT)),
            "source_portable_checkpoint_sha256": _sha256(source_model_path),
            "base_checkpoint": str(BASE_CHECKPOINT.relative_to(ROOT)),
            "base_checkpoint_sha256": BASE_SHA256,
            "rl_checkpoint": str(checkpoint.relative_to(ROOT)),
            "rl_checkpoint_sha256": checkpoint_sha256,
            "checkpoint_update": checkpoint_update,
            "source_policy_update": metadata.get("source_policy_update"),
            "adaptation": adaptation,
            "adapter_deployment": "merged_into_last_option_self_and_cross_attention_qv",
            "action_boundary_deployment": {
                "decision_gate": True,
                "forced_observe_only": True,
                "phantom_allocation_head": True,
                "official_primitive_transaction": True,
                "opponent_meta_conditioning": meta_enabled,
            },
            "semantic_runtime_source": str(SEMANTIC_RUNTIME_ROOT.relative_to(ROOT)),
            "prototype_embedding_cache": {
                "enabled": True,
                "scope": "once_per_process_across_battles",
                "invalidation": "weights_device_dtype_or_trainable_prototypes",
            },
            "prototype_embedding_cache_version": "frozen_model_owned_v1",
            "prototype_embedding_cache_required": True,
            "prototype_embedding_cache_persistent": False,
            "portable_checkpoint_schema_version": portable["schema_version"],
            "portable_checkpoint_sha256": _sha256(output / "strategy/model.bin"),
            "storage_dtype": "fp16",
            "runtime_dtype": "fp32",
            "evaluation_inference_device": "cuda:0",
            "model_only": True,
            "optimizer_state_saved": False,
            "critic_deployed": True,
            "critic_runtime_role": "strict-loaded diagnostic only; excluded from select()",
            "frozen_evaluation": frozen,
            "selection": (
                f"{metadata.get('version')} update{checkpoint_update} canonical Frozen-0806 greedy "
                f"{frozen['wins']}-{frozen['losses']} "
                f"({frozen['win_rate'] * 100:.8f}%), 0 error"
            ),
        }
        manifest["package_file_sha256"] = _package_file_hashes(output)
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
