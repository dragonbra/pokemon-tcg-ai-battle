"""Human-authorized U110 -> immutable Champion-G3 admission workflow."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import torch

from ..assets import AssetRegistry, canonical_json_sha256, sha256_file
from ..evaluation import benchmark_v2_g2_vs_g3_u110_full67 as full67
from ..evaluation.candidate import FIELDS, materialize
from ..own_archetype import OwnArchetypeVocabulary
from ..policy_identity import COMPLETE_PORTABLE_FIELDS, _tensor_hash
from .workflow import PromotionGateError, freeze_candidate_record, validate_manual_decision


POLICY_ID = "Champion-G3"
CANDIDATE_ID = "Candidate-0044-G3-U110"
SOURCE_UPDATE = 110
SOURCE_SHA256 = full67.G3_CHECKPOINT_SHA256
MINIMUM_EVALUATED_DECKS = 18


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PromotionGateError(f"JSON evidence must be an object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def collect_evidence(repository_root: Path) -> dict[str, Any]:
    source = repository_root / full67.G3_CHECKPOINT.relative_to(full67.ROOT)
    if not source.is_file() or sha256_file(source) != SOURCE_SHA256:
        raise PromotionGateError("G3 U110 source checkpoint identity mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != "0044_focal_v1_model_only_v1"
        or payload.get("update") != SOURCE_UPDATE
        or payload.get("metadata", {}).get("version") != full67.G3_VERSION
    ):
        raise PromotionGateError("G3 U110 source metadata mismatch")
    if (source.parent / "update-000111.pt").exists():
        raise PromotionGateError("G3 source escaped the exact U110 terminal boundary")

    completed: list[dict[str, Any]] = []
    for deck_id in full67.EXECUTION_DECK_IDS:
        g2_path = full67.REPORT_ROOT / deck_id / "g2/report.json"
        g3_path = full67.REPORT_ROOT / deck_id / "g3_u110/report.json"
        if not g2_path.is_file() or not g3_path.is_file():
            continue
        g2, g3 = _json(g2_path), _json(g3_path)
        full67.validate_pair(g2, g3, deck_id=deck_id)
        completed.append({
            "deck_id": deck_id,
            "g2_report_sha256": sha256_file(g2_path),
            "g3_report_sha256": sha256_file(g3_path),
            "g2_win_rate": g2["summary"]["win_rate"],
            "g3_win_rate": g3["summary"]["win_rate"],
            "games_per_arm": 2048,
        })
    required = {"007", "003", "002", "066"}
    if len(completed) < MINIMUM_EVALUATED_DECKS or not required <= {
        row["deck_id"] for row in completed
    }:
        raise PromotionGateError(
            "Champion-G3 requires the authorized partial Full67 evidence floor"
        )
    return {
        "source": source,
        "source_checkpoint_sha256": SOURCE_SHA256,
        "evaluated_decks": completed,
        "evaluation_status_at_decision": "PARTIAL_USER_ACCEPTED",
        "evaluation_continues_after_decision": True,
    }


def promote_champion_g3(
    repository_root: Path, *, decided_by: str, decision_note: str,
) -> dict[str, Any]:
    if not decided_by or decided_by in {"automation", "codex"}:
        raise PromotionGateError("Champion-G3 requires an identified human decision")
    repository_root = repository_root.resolve()
    project_root = repository_root / "train/0044_g2_dragapult_policy_option_lora"
    evidence = collect_evidence(repository_root)
    registry = AssetRegistry.load(project_root)
    registry.validate_all()
    if registry.latest_champion_policy_id != "Champion-G2":
        raise PromotionGateError("Champion-G3 parent must be immutable Champion-G2")

    definition = project_root / "assets/policies/definitions/champion_g003"
    manifest_path = project_root / "assets/policies/manifests/champion_g003.json"
    promotion_path = (
        repository_root / "experiments/0044_g2_dragapult_policy_option_lora/promotion/"
        "champion_g003_u110.json"
    )
    if definition.exists() or manifest_path.exists() or promotion_path.exists():
        raise PromotionGateError("Champion-G3 immutable target already exists")
    staging = definition.with_name("champion_g003.tmp")
    if staging.exists():
        raise PromotionGateError("stale Champion-G3 promotion staging exists")
    staging.mkdir(parents=True)
    source_out = staging / "source_update_000110.pt"
    shutil.copyfile(evidence["source"], source_out)

    deck_asset = next(row for row in registry.decks if row.deck_id == "001")
    cards = tuple(map(int, (project_root / deck_asset.deck_path).read_text().splitlines()))
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=project_root
    )
    own_id = next(row.archetype_id for row in vocabulary.mappings if row.deck_id == "001")
    portable_out = staging / "model.bin"
    _, audit = materialize(
        checkpoint=source_out,
        base_portable=project_root / "assets/policies/definitions/champion_g002/model.bin",
        deck=cards,
        deck_id="001",
        own_archetype_id=own_id,
        output=portable_out,
        device=torch.device("cpu"),
    )
    strict = torch.load(portable_out, map_location="cpu", weights_only=True)
    if tuple(field for field in COMPLETE_PORTABLE_FIELDS if field in strict) != COMPLETE_PORTABLE_FIELDS:
        raise PromotionGateError("Champion-G3 portable policy is incomplete")
    tensor_rows = {
        f"{field}.{name}": value
        for field in COMPLETE_PORTABLE_FIELDS
        for name, value in strict[field].items()
    }
    effective_policy_sha256 = _tensor_hash(tensor_rows)
    candidate = freeze_candidate_record(
        candidate_id=CANDIDATE_ID,
        source_checkpoint=source_out,
        source_update=SOURCE_UPDATE,
        deployment_audit=audit.to_manifest(),
        parent_policy_id="Champion-G2",
    )
    evidence_identity = {
        "source_checkpoint_sha256": SOURCE_SHA256,
        "evaluated_decks": evidence["evaluated_decks"],
        "portable_checkpoint_sha256": sha256_file(portable_out),
        "portable_tensor_sha256": effective_policy_sha256,
    }
    evidence_sha256 = canonical_json_sha256(evidence_identity)
    decision = {
        "decision": "PROMOTE",
        "candidate_id": candidate["candidate_id"],
        "candidate_record_sha256": candidate["candidate_record_sha256"],
        "evidence_sha256": evidence_sha256,
        "decided_by": decided_by,
        "decision_note": decision_note,
    }
    if validate_manual_decision(
        decision, candidate_record=candidate, evidence_sha256=evidence_sha256
    ) != "PROMOTE":
        raise PromotionGateError("Champion-G3 registry mutation requires PROMOTE")

    source_sha = sha256_file(source_out)
    portable_sha = sha256_file(portable_out)
    artifacts = [
        {
            "path": "assets/policies/definitions/champion_g003/source_update_000110.pt",
            "purpose": "model_only_checkpoint", "sha256": source_sha,
        },
        {
            "path": "assets/policies/definitions/champion_g003/model.bin",
            "purpose": "portable_fp16_artifact", "sha256": portable_sha,
        },
    ]
    promoted_at = datetime.now(UTC).isoformat()
    manifest = {
        "schema_version": "0044_policy_manifest_v2",
        "policy_id": POLICY_ID, "role": "latest_champion", "generation": 3,
        "frozen": True, "parent_policy_id": "Champion-G2",
        "architecture_version": "0044_focal_v1_plus_policy_option_lora_v1",
        "source_checkpoint_schema": "0044_focal_v1_model_only_v1",
        "source_checkpoint_artifact_purpose": "model_only_checkpoint",
        "portable_checkpoint_schema": strict["schema_version"],
        "source_base_artifact_purpose": None,
        "portable_fields": list(COMPLETE_PORTABLE_FIELDS),
        "effective_policy_sha256": effective_policy_sha256,
        "portable_tensor_sha256": effective_policy_sha256,
        "artifacts": artifacts,
        "components": {
            field: {"effective_sha256": _tensor_hash({
                f"{field}.{name}": value for name, value in strict[field].items()
            })}
            for field in COMPLETE_PORTABLE_FIELDS
        },
        "inference_semantics": {
            "deploy_mode": "kaggle_fp16_storage_fp32_runtime_v1",
            "storage_dtype": "fp16", "runtime_dtype": "fp32",
            "own_deck_conditioning": "own_archetypes_v2_exact_deck_29_way",
            "opponent_meta_head": "frozen_15_way_unchanged",
            "policy_option_lora": "final_option_block_qv_rank4",
            "meta_actor_residual": "zero_initialized_not_part_of_g3_training",
        },
        "promotion_record": (
            "experiments/0044_g2_dragapult_policy_option_lora/promotion/"
            "champion_g003_u110.json"
        ),
        "source": {
            "project_version": full67.G3_VERSION, "update": SOURCE_UPDATE,
            "promoted_at": promoted_at,
        },
    }
    _atomic_json(manifest_path, manifest)
    os.replace(staging, definition)
    manifest_sha = sha256_file(manifest_path)

    registry_path = project_root / "assets/policies/registry.json"
    registry_payload = _json(registry_path)
    for row in registry_payload["policies"]:
        if row["policy_id"] == "Champion-G2":
            row["role"] = "champion"
    registry_payload["policies"].append({
        "policy_id": POLICY_ID, "role": "latest_champion", "generation": 3,
        "frozen": True,
        "manifest_path": "assets/policies/manifests/champion_g003.json",
        "manifest_sha256": manifest_sha,
        "effective_policy_sha256": effective_policy_sha256,
        "artifacts": artifacts,
    })
    registry_payload["active_policy_pool"] = [POLICY_ID]
    registry_payload["latest_champion_policy_id"] = POLICY_ID
    promotion = {
        "schema_version": "0044_champion_promotion_record_v2",
        "status": "PROMOTED", "policy_id": POLICY_ID, "generation": 3,
        "parent_policy_id": "Champion-G2", "source_update": SOURCE_UPDATE,
        "source_checkpoint_sha256": SOURCE_SHA256,
        "portable_checkpoint_sha256": portable_sha,
        "effective_policy_sha256": effective_policy_sha256,
        "candidate_record": candidate, "human_decision": decision,
        "evidence_sha256": evidence_sha256,
        "evidence": evidence_identity,
        "promoted_at": promoted_at,
    }
    _atomic_json(promotion_path, promotion)
    _atomic_json(registry_path, registry_payload)
    admitted = AssetRegistry.load(project_root)
    admitted.validate_all()
    return {
        "status": "PROMOTED", "policy_id": POLICY_ID,
        "source_checkpoint_sha256": SOURCE_SHA256,
        "portable_checkpoint_sha256": portable_sha,
        "effective_policy_sha256": effective_policy_sha256,
        "evidence_sha256": evidence_sha256,
        "evaluated_decks_at_decision": len(evidence["evaluated_decks"]),
        "active_policy_pool": list(admitted.active_policy_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--decided-by", default="project_owner")
    parser.add_argument(
        "--decision-note",
        default=(
            "User explicitly authorized Champion-G3 promotion on 2026-08-14 "
            "and accepted the available partial Full67 CUDA-2048 evidence."
        ),
    )
    args = parser.parse_args()
    if not args.promote:
        parser.error("Champion-G3 registry mutation requires --promote")
    root = Path(__file__).resolve().parents[3]
    print(json.dumps(promote_champion_g3(
        root, decided_by=args.decided_by, decision_note=args.decision_note,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
