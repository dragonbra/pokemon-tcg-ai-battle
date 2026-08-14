"""Fail-closed U407 -> immutable Champion-G2 admission workflow."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import torch

from ..assets import AssetRegistry, canonical_json_sha256, sha256_file
from ..evaluation.candidate import FIELDS, _tensor_hash, materialize
from ..evaluation import g2_candidate_full67 as full67
from ..evaluation import g2_candidate_gate as gate
from ..own_archetype import OwnArchetypeVocabulary
from ..policy_identity import PORTABLE_FIELDS, _tensor_hash as portable_tensor_hash
from .workflow import PromotionGateError, freeze_candidate_record, validate_manual_decision


POLICY_ID = "Champion-G2"
CANDIDATE_ID = "Candidate-0044-U407"
SOURCE_SHA256 = "828c1791b62152667f0b87b2b668c2b16c6ab0bce9b7996558d3012628b106bb"
REPORT_RELATIVE = Path(
    "docs/evaluation/combat_mat/policy_0809/"
    "u407_g2_candidate_vs_champion_g1_cuda_seeded_2048_agent_choice_v3_"
    "kaggle_fp16_storage_fp32_runtime_v1"
)


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


def collect_full67_evidence(repository_root: Path) -> dict[str, Any]:
    project_root = repository_root / "train/0044_g2_dragapult_policy_option_lora"
    registry = AssetRegistry.load(project_root)
    registry.validate_all()
    if registry.latest_champion_policy_id != "Champion-G1" or POLICY_ID in {
        row.policy_id for row in registry.policies
    }:
        raise PromotionGateError("Champion-G2 generation is not available for first admission")
    checkpoint = repository_root / gate.U407_CHECKPOINT.relative_to(gate.ROOT)
    if sha256_file(checkpoint) != SOURCE_SHA256:
        raise PromotionGateError("U407 source checkpoint identity mismatch")
    report_root = repository_root / REPORT_RELATIVE
    manifest_path = report_root / "manifest.json"
    index_path = report_root / "index.html"
    manifest = _json(manifest_path)
    expected_ids = tuple(f"{value:03d}" for value in range(1, 68))
    if (
        manifest.get("status") != "HUMAN_DECISION_REQUIRED"
        or manifest.get("completed_decks") != 67
        or manifest.get("target_decks") != 67
        or tuple(manifest.get("fixed_deck_ids", ())) != expected_ids
    ):
        raise PromotionGateError("authoritative Full67 report is incomplete")
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=project_root
    )
    deck_assets = {row.deck_id: row for row in registry.decks}
    evaluations: list[dict[str, Any]] = []
    portable_tensor_identity: str | None = None
    for deck_id in expected_ids:
        report_path = full67.source_report(deck_id, "g2_candidate")
        if report_path is None:
            raise PromotionGateError(f"missing Full67 G2 evidence: {deck_id}")
        report_path = repository_root / report_path.relative_to(gate.ROOT)
        report = _json(report_path)
        gate.validate_arm_report(report, arm="g2_candidate", deck_id=deck_id)
        audit = report["candidate_deployment_identity_audit"]
        asset = deck_assets[deck_id]
        if (
            report.get("focal_deck_id") != deck_id
            or report.get("focal_exact_deck_sha256") != asset.content_sha256
            or audit.get("focal_exact_deck_sha256") != asset.content_sha256
            or audit.get("source_checkpoint_sha256") != SOURCE_SHA256
        ):
            raise PromotionGateError(f"Full67 exact-deck/source identity mismatch: {deck_id}")
        portable_path = report_path.parent / "materialization/model.bin"
        strict = torch.load(portable_path, map_location="cpu", weights_only=True)
        tensor_rows = {
            f"{field}.{name}": value
            for field in PORTABLE_FIELDS
            for name, value in strict[field].items()
        }
        identity = portable_tensor_hash(tensor_rows)
        portable_tensor_identity = portable_tensor_identity or identity
        if identity != portable_tensor_identity:
            raise PromotionGateError(f"candidate tensor content differs by deck: {deck_id}")
        if _tensor_hash(strict, asset.content_sha256) != audit["effective_candidate_sha256"]:
            raise PromotionGateError(f"deployment-effective reconstruction mismatch: {deck_id}")
        evaluations.append({
            "deck_id": deck_id,
            "own_archetype_id": next(
                row.archetype_id for row in vocabulary.mappings if row.deck_id == deck_id
            ),
            "exact_deck_sha256": asset.content_sha256,
            "effective_candidate_sha256": audit["effective_candidate_sha256"],
            "portable_checkpoint_sha256": audit["portable_checkpoint_sha256"],
            "report_sha256": sha256_file(report_path),
        })
    return {
        "checkpoint": checkpoint,
        "source_checkpoint_sha256": SOURCE_SHA256,
        "portable_tensor_sha256": portable_tensor_identity,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "index_path": index_path,
        "index_sha256": sha256_file(index_path),
        "evaluations": evaluations,
    }


def prepare_decision_binding(repository_root: Path) -> dict[str, Any]:
    evidence = collect_full67_evidence(repository_root)
    first = evidence["evaluations"][0]
    audit = {
        "status": "PASS",
        "contract_id": "kaggle_fp16_storage_fp32_runtime_v1",
        "storage_dtype": "fp16",
        "runtime_dtype": "fp32",
        "source_checkpoint_sha256": SOURCE_SHA256,
        "portable_checkpoint_sha256": first["portable_checkpoint_sha256"],
        "effective_candidate_sha256": first["effective_candidate_sha256"],
    }
    candidate = freeze_candidate_record(
        candidate_id=CANDIDATE_ID,
        source_checkpoint=evidence["checkpoint"],
        source_update=407,
        deployment_audit=audit,
        parent_policy_id="Champion-G1",
    )
    evidence_identity = {
        "report_manifest_sha256": evidence["manifest_sha256"],
        "report_index_sha256": evidence["index_sha256"],
        "source_checkpoint_sha256": SOURCE_SHA256,
        "per_deck_effective_sha256": {
            row["deck_id"]: row["effective_candidate_sha256"]
            for row in evidence["evaluations"]
        },
    }
    return {
        "candidate_record": candidate,
        "evidence_sha256": canonical_json_sha256(evidence_identity),
        "evidence": evidence,
    }


def promote_champion_g2(
    repository_root: Path, decision: Mapping[str, Any]
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    project_root = repository_root / "train/0044_g2_dragapult_policy_option_lora"
    binding = prepare_decision_binding(repository_root)
    if validate_manual_decision(
        decision,
        candidate_record=binding["candidate_record"],
        evidence_sha256=binding["evidence_sha256"],
    ) != "PROMOTE":
        raise PromotionGateError("Champion-G2 registry mutation requires PROMOTE")

    definition = project_root / "assets/policies/definitions/champion_g002"
    manifest_path = project_root / "assets/policies/manifests/champion_g002.json"
    promotion_path = (
        repository_root / "experiments/0044_g2_dragapult_policy_option_lora/promotion/"
        "champion_g002_u407.json"
    )
    if definition.exists() or manifest_path.exists() or promotion_path.exists():
        raise PromotionGateError("Champion-G2 immutable target already exists")
    staging = definition.with_name("champion_g002.tmp")
    if staging.exists():
        raise PromotionGateError("stale Champion-G2 promotion staging exists")
    staging.mkdir(parents=True)
    source_out = staging / "source_update_000407.pt"
    shutil.copyfile(binding["evidence"]["checkpoint"], source_out)

    registry = AssetRegistry.load(project_root)
    deck = next(row for row in registry.decks if row.deck_id == "001")
    cards = tuple(map(int, (project_root / deck.deck_path).read_text().splitlines()))
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=project_root
    )
    own_id = next(row.archetype_id for row in vocabulary.mappings if row.deck_id == "001")
    portable_out = staging / "model.bin"
    _, canonical_audit = materialize(
        checkpoint=source_out,
        base_portable=repository_root / gate.BASE_PORTABLE.relative_to(gate.ROOT),
        deck=cards,
        deck_id="001",
        own_archetype_id=own_id,
        output=portable_out,
        device=torch.device("cpu"),
    )
    strict = torch.load(portable_out, map_location="cpu", weights_only=True)
    tensors = {
        f"{field}.{name}": value
        for field in PORTABLE_FIELDS
        for name, value in strict[field].items()
    }
    effective_policy_sha256 = portable_tensor_hash(tensors)
    if effective_policy_sha256 != binding["evidence"]["portable_tensor_sha256"]:
        raise PromotionGateError("frozen G2 tensor identity differs from evaluated candidate")
    source_sha = sha256_file(source_out)
    portable_sha = sha256_file(portable_out)
    artifacts = [
        {
            "path": "assets/policies/definitions/champion_g002/source_update_000407.pt",
            "purpose": "model_only_checkpoint",
            "sha256": source_sha,
        },
        {
            "path": "assets/policies/definitions/champion_g002/model.bin",
            "purpose": "portable_fp16_artifact",
            "sha256": portable_sha,
        },
    ]
    promoted_at = datetime.now(UTC).isoformat()
    manifest = {
        "schema_version": "0044_policy_manifest_v1",
        "policy_id": POLICY_ID,
        "role": "latest_champion",
        "generation": 2,
        "frozen": True,
        "parent_policy_id": "Champion-G1",
        "architecture_version": "0044_focal_v1_model_only_v1",
        "source_checkpoint_schema": "0044_focal_v1_model_only_v1",
        "source_checkpoint_artifact_purpose": "model_only_checkpoint",
        "portable_checkpoint_schema": "0044_focal_v1_kaggle_candidate_v1",
        "source_base_artifact_purpose": None,
        "effective_policy_sha256": effective_policy_sha256,
        "portable_tensor_sha256": effective_policy_sha256,
        "artifacts": artifacts,
        "components": {
            field: {"effective_sha256": portable_tensor_hash({
                f"{field}.{name}": value for name, value in strict[field].items()
            })} for field in FIELDS
        },
        "inference_semantics": {
            "deploy_mode": "kaggle_fp16_storage_fp32_runtime_v1",
            "storage_dtype": "fp16",
            "runtime_dtype": "fp32",
            "own_deck_conditioning": "own_archetypes_v2_exact_deck_29_way",
            "opponent_meta_head": "frozen_15_way_unchanged",
        },
        "promotion_record": "experiments/0044_g2_dragapult_policy_option_lora/promotion/champion_g002_u407.json",
        "source": {
            "project_version": "V6_generalist_focal_001_067_u233_restart",
            "update": 407,
            "promoted_at": promoted_at,
        },
    }
    _atomic_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)
    os.replace(staging, definition)

    registry_path = project_root / "assets/policies/registry.json"
    registry_payload = _json(registry_path)
    for row in registry_payload["policies"]:
        if row["policy_id"] == "Champion-G1":
            row["role"] = "champion"
    registry_payload["policies"].append({
        "policy_id": POLICY_ID,
        "role": "latest_champion",
        "generation": 2,
        "frozen": True,
        "manifest_path": "assets/policies/manifests/champion_g002.json",
        "manifest_sha256": manifest_sha,
        "effective_policy_sha256": effective_policy_sha256,
        "artifacts": artifacts,
    })
    registry_payload["active_policy_pool"] = ["Policy-0809", "Champion-G1", POLICY_ID]
    registry_payload["latest_champion_policy_id"] = POLICY_ID

    promotion = {
        "schema_version": "0044_champion_promotion_record_v1",
        "status": "PROMOTED",
        "policy_id": POLICY_ID,
        "generation": 2,
        "parent_policy_id": "Champion-G1",
        "source_update": 407,
        "source_checkpoint_sha256": SOURCE_SHA256,
        "portable_checkpoint_sha256": portable_sha,
        "effective_policy_sha256": effective_policy_sha256,
        "candidate_record": binding["candidate_record"],
        "human_decision": dict(decision),
        "evidence_sha256": binding["evidence_sha256"],
        "evidence": {
            "authoritative_report": str(REPORT_RELATIVE / "index.html"),
            "index_sha256": binding["evidence"]["index_sha256"],
            "manifest_sha256": binding["evidence"]["manifest_sha256"],
            "evaluated_decks": binding["evidence"]["evaluations"],
        },
        "promoted_at": promoted_at,
    }
    _atomic_json(promotion_path, promotion)
    _atomic_json(registry_path, registry_payload)
    audit = AssetRegistry.load(project_root).validate_all()
    return {
        "status": "PROMOTED",
        "policy_id": POLICY_ID,
        "effective_policy_sha256": effective_policy_sha256,
        "portable_checkpoint_sha256": portable_sha,
        "evidence_sha256": binding["evidence_sha256"],
        "latest_champion_policy_id": audit.latest_champion_policy_id,
        "active_policy_pool": list(AssetRegistry.load(project_root).active_policy_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-decision", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--decided-by", default="human_user")
    parser.add_argument("--decision-note", default="User explicitly approved formal Champion-G2 admission.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    binding = prepare_decision_binding(root)
    if args.prepare_decision and not args.promote:
        print(json.dumps({
            "candidate_record": binding["candidate_record"],
            "evidence_sha256": binding["evidence_sha256"],
        }, indent=2, sort_keys=True))
        return 0
    if not args.promote:
        parser.error("choose --prepare-decision or --promote")
    decision = {
        "decision": "PROMOTE",
        "candidate_id": binding["candidate_record"]["candidate_id"],
        "candidate_record_sha256": binding["candidate_record"]["candidate_record_sha256"],
        "evidence_sha256": binding["evidence_sha256"],
        "decided_by": args.decided_by,
        "decision_note": args.decision_note,
    }
    print(json.dumps(promote_champion_g2(root, decision), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
