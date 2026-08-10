"""Export a 0042 compound-action checkpoint as a self-contained candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import torch

from .policy.own_archetype import OwnArchetypeVocabulary

from .action_boundary.contracts import (
    ACTION_BOUNDARY_SCHEMA_VERSION,
    CANONICALIZER_VERSION,
    DECISION_GATE_VERSION,
    FEATURE_PREPROCESSING_VERSION,
    OFFICIAL_PROTOCOL_ADAPTER_VERSION,
    TRAJECTORY_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ID = "0042_full_model_design"
SOURCE_SCHEMA = "0031_shared_prototype_fp16_storage_candidate_checkpoint_v1"
SOURCE_RUNTIME_SCHEMA = (
    "0031_shared_prototype_fp16_storage_fp32_runtime_candidate_checkpoint_v1"
)
OUTPUT_SCHEMA = "0042_strategy_conditioned_kaggle_candidate_v1"
SOURCE_SCHEMAS = frozenset({SOURCE_SCHEMA, SOURCE_RUNTIME_SCHEMA})
BASE_SCHEMA = "0031_model_only_checkpoint_v1"
BASE_SHA256 = "926321955b6f3144b62e65899b5041ca3a47b17f305202ba9dffc0c92aaa7c7f"
BASE_CHECKPOINT = (
    ROOT / "archive/pretrained/0031_friend_0809_gsb_v5_value_v9/model.pt"
)
RL_SCHEMA = "0042_strategy_conditioned_model_only_v1"
RUNTIME_ROOT = ROOT / "train/0042_full_model_design/kaggle_runtime"
ACTION_BOUNDARY_ROOT = ROOT / "train/0042_full_model_design/action_boundary"
SEMANTIC_RUNTIME_ROOT = ROOT / "train/0042_full_model_design/semantic_policy"
FOCAL_DECK_ID = "dragapult_ex_07bedfffbfad"
FOCAL_DECK_SHA256 = "07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725"
FOCAL_DECK_PATH = (
    ROOT / "train/0042_full_model_design/league/decks/007_dragapult_ex/deck.csv"
)
_PROTOTYPE_ALIASES = ("state_encoder.prototypes.", "option_encoder.prototypes.")
PORTABLE_STATE_FIELDS = (
    "actor_state_dict",
    "value_head_state_dict",
    "allocation_head_state_dict",
    "value_adapter_state_dict",
    "policy_strategy_adapter_state_dict",
)


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


def deployment_effective_sha256(
    payload: Mapping[str, Any], manifest: Mapping[str, Any]
) -> str:
    """Hash semantic deployment content independently of torch.save bytes."""

    digest = hashlib.sha256()
    digest.update(b"kaggle_fp16_storage_fp32_runtime_v1\0")
    semantic_metadata = {
        "schema_version": payload.get("schema_version"),
        "metadata": payload.get("metadata"),
        "action_boundary_deployment": manifest.get("action_boundary_deployment"),
        "storage_dtype": manifest.get("storage_dtype"),
        "runtime_dtype": manifest.get("runtime_dtype"),
    }
    digest.update(json.dumps(
        semantic_metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii"))
    digest.update(b"\0")
    for field in PORTABLE_STATE_FIELDS:
        state = payload.get(field)
        if not isinstance(state, Mapping):
            raise ValueError(f"portable checkpoint is missing {field}")
        for name, value in sorted(state.items()):
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"portable tensor is not a Tensor: {field}.{name}")
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{field}.{name}".encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(json.dumps(
                list(tensor.shape), separators=(",", ":")
            ).encode("ascii"))
            digest.update(b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _canonical_base_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = payload.get("state_dict")
    if (
        payload.get("schema_version") != BASE_SCHEMA
        or not isinstance(state, dict)
        or len(state) != 293
    ):
        raise ValueError("paired 0809 base checkpoint does not match the audited actor")
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
        result_schema != "0040_frozen_per_game_results_policy_identity_v4"
        or payload.get("checkpoint_update") != update
        or payload.get("frozen_panel_version")
        != "frozen_0806_seeded_agent_first_player_v3"
        or not isinstance(entries, list)
        or len(entries) != 2048
    ):
        raise ValueError("canonical Frozen evaluation contract mismatch")
    audit = payload.get("policy_identity_audit") or {}
    if (
        payload.get("opponent_policy_id") != "Policy-0806"
        or audit.get("status") != "PASS"
        or audit.get("requested_policy_id") != "Policy-0806"
    ):
        raise ValueError("canonical Frozen evaluation policy identity audit failed")
    candidate_audit = payload.get("candidate_deployment_identity_audit") or {}
    if (
        candidate_audit.get("status") != "PASS"
        or candidate_audit.get("contract_id")
        != "kaggle_fp16_storage_fp32_runtime_v1"
        or candidate_audit.get("storage_dtype") != "fp16"
        or candidate_audit.get("runtime_dtype") != "fp32"
        or candidate_audit.get("checkpoint_update") != update
        or len(str(candidate_audit.get("portable_checkpoint_sha256", ""))) != 64
        or len(str(candidate_audit.get("effective_candidate_sha256", ""))) != 64
        or candidate_audit.get("source_checkpoint_sha256")
        != _sha256(checkpoint)
    ):
        raise ValueError(
            "canonical Frozen evaluation candidate deployment audit failed"
        )
    valid = [
        row for row in entries
        if row.get("valid") is True and row.get("error") in (None, "")
    ]
    if len(valid) != 2048:
        raise ValueError("canonical Frozen evaluation contains invalid/error games")
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
        raise ValueError("canonical Frozen chance/fallback evidence is malformed")
    semantic_fallbacks = sum(bool(row["semantic_fallback"]) for row in valid)
    if semantic_fallbacks:
        raise ValueError("canonical Frozen evaluation contains semantic fallback")
    chance_boundaries = sum(bool(row["chance_boundary"]) for row in valid)
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
        "candidate_deployment_identity_audit": candidate_audit,
    }


def export_candidate(
    *,
    source: Path,
    checkpoint: Path,
    output: Path,
    require_frozen_selection: bool = True,
) -> dict[str, Any]:
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
        "no_option_lora": True,
        "value_adapter": "zero_gated_residual",
        "policy_strategy_adapter": "readout_only_zero_gated_residual",
    }
    if (
        rl_payload.get("schema_version") != RL_SCHEMA
        or metadata.get("project_id") != PROJECT_ID
        or metadata.get("version") != checkpoint.parent.parent.name
        or adaptation != expected_adaptation
        or not flags.get("enable_action_boundary")
        or not flags.get("enable_forced_shortcut")
        or not flags.get("enable_dragapult_macro")
    ):
        raise ValueError("checkpoint is not an audited 0042 compound policy")
    checkpoint_sha256 = _checkpoint_sidecar(checkpoint)
    frozen = (
        _frozen_selection(checkpoint, checkpoint_update)
        if require_frozen_selection
        else None
    )
    if _sha256(BASE_CHECKPOINT) != BASE_SHA256:
        raise ValueError("audited 0809 base checkpoint SHA-256 mismatch")
    base_payload = _load_payload(BASE_CHECKPOINT)

    source_state = source_payload.get("state_dict")
    rl_state = rl_payload.get("state_dict")
    if not isinstance(source_state, dict) or not isinstance(rl_state, dict):
        raise ValueError("checkpoint has no state_dict mapping")
    merged_state = _canonical_base_state(base_payload)
    if set(merged_state) != set(source_state):
        raise ValueError("audited fp16 source package tensor inventory differs from 0809 base")

    decoder_names = {name for name in source_state if name.startswith("action_decoder.")}
    rl_decoder_names = {f"actor.{name}" for name in decoder_names}
    deployed_prefixes = (
        "allocation_head.", "value_adapter.", "policy_strategy_adapter."
    )
    value_names = {name for name in rl_state if name.startswith("value_head.")}
    ignored_prefixes = ("prize_aux.",)
    deployed_names = {name for name in rl_state if name.startswith(deployed_prefixes)}
    ignored_names = {name for name in rl_state if name.startswith(ignored_prefixes)}
    if (
        set(rl_state) != (
            rl_decoder_names | value_names | deployed_names | ignored_names
        )
        or not value_names
        or not any(name.startswith("allocation_head.") for name in deployed_names)
        or not any(name.startswith("value_adapter.") for name in deployed_names)
        or not any(name.startswith("policy_strategy_adapter.") for name in deployed_names)
    ):
        raise ValueError("0042 trainable tensor inventory is incomplete or unexpected")

    for name in decoder_names:
        value = rl_state[f"actor.{name}"]
        if value.shape != merged_state[name].shape:
            raise ValueError(f"decoder shape mismatch: {name}")
        merged_state[name] = value.clone()

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

    focal_cards = [int(value) for value in FOCAL_DECK_PATH.read_text().splitlines() if value]
    own_archetype_id = OwnArchetypeVocabulary.load().classify_own_deck(focal_cards).value
    portable_metadata = {
        "project_id": PROJECT_ID,
        "version": metadata.get("version"),
        "checkpoint_update": checkpoint_update,
        "checkpoint_sha256": checkpoint_sha256,
        "source_policy_update": metadata.get("source_policy_update"),
        "actor_metadata": source_payload["metadata"],
        "opponent_meta_class_count": 15,
        "own_archetype_id": own_archetype_id,
        "own_archetype_vocabulary_version": metadata["own_archetype_vocabulary_version"],
        "own_archetype_taxonomy_sha256": metadata["own_archetype_taxonomy_sha256"],
        "no_option_lora": True,
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
        "value_adapter_state_dict": portable_substate("value_adapter."),
        "policy_strategy_adapter_state_dict": portable_substate(
            "policy_strategy_adapter."
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
        # Deploy the project-local semantic runtime rather than inheriting
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
            "adapter_deployment": "q0 Value residual + decoder readout-only Strategy residual",
            "action_boundary_deployment": {
                "decision_gate": True,
                "forced_observe_only": True,
                "phantom_allocation_head": True,
                "official_primitive_transaction": True,
                "pretrained_q1_meta_conditioning": True,
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
            "critic_runtime_role": (
                "q0/q1, frozen Meta logits, and adapted Value feed every strategic "
                "policy decision through detached strategy context"
            ),
            "frozen_evaluation": frozen,
            "selection": (
                f"{metadata.get('version')} update{checkpoint_update} canonical Frozen-0806 greedy "
                f"{frozen['wins']}-{frozen['losses']} "
                f"({frozen['win_rate'] * 100:.8f}%), 0 error"
                if frozen is not None
                else "pre-evaluation deployment materialization; no strength result attached"
            ),
        }
        manifest["deployment_effective_sha256"] = deployment_effective_sha256(
            portable, manifest
        )
        if (
            frozen is not None
            and frozen["candidate_deployment_identity_audit"][
                "effective_candidate_sha256"
            ] != manifest["deployment_effective_sha256"]
        ):
            raise ValueError(
                "final package deployment identity differs from evaluated candidate"
            )
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
