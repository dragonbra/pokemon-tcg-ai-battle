"""Candidate freeze and explicit human decision records.

Registry mutation is intentionally not exposed until formal Frozen evaluation is
implemented and the user supplies a decision for an exact candidate/evidence hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..assets import sha256_file


class PromotionGateError(RuntimeError):
    pass


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def freeze_candidate_record(
    *, candidate_id: str, source_checkpoint: Path, source_update: int,
    deployment_audit: Mapping[str, Any], parent_policy_id: str,
) -> dict[str, Any]:
    if not candidate_id.startswith("Candidate-0044-"):
        raise PromotionGateError("candidate_id must use the Candidate-0044 namespace")
    if deployment_audit.get("status") != "PASS":
        raise PromotionGateError("candidate deployment audit must PASS before freeze")
    if (
        deployment_audit.get("contract_id") != "kaggle_fp16_storage_fp32_runtime_v1"
        or deployment_audit.get("storage_dtype") != "fp16"
        or deployment_audit.get("runtime_dtype") != "fp32"
    ):
        raise PromotionGateError("candidate deployment precision contract mismatch")
    digest = sha256_file(source_checkpoint)
    if deployment_audit.get("source_checkpoint_sha256") != digest:
        raise PromotionGateError("candidate source checkpoint identity mismatch")
    record = {
        "schema_version": "0044_frozen_candidate_record_v1",
        "candidate_id": candidate_id,
        "parent_policy_id": parent_policy_id,
        "source_update": source_update,
        "source_checkpoint_sha256": digest,
        "portable_checkpoint_sha256": deployment_audit["portable_checkpoint_sha256"],
        "deployment_effective_sha256": deployment_audit["effective_candidate_sha256"],
        "deployment_contract": deployment_audit["contract_id"],
        "immutable": True,
        "promotion_status": "AWAITING_HUMAN_DECISION",
    }
    record["candidate_record_sha256"] = _canonical_hash(record)
    return record


def validate_manual_decision(
    decision: Mapping[str, Any], *, candidate_record: Mapping[str, Any],
    evidence_sha256: str,
) -> str:
    value = decision.get("decision")
    if value not in {"PROMOTE", "HOLD", "REJECT"}:
        raise PromotionGateError("human decision must be PROMOTE, HOLD, or REJECT")
    if (
        decision.get("candidate_id") != candidate_record.get("candidate_id")
        or decision.get("candidate_record_sha256")
        != candidate_record.get("candidate_record_sha256")
        or decision.get("evidence_sha256") != evidence_sha256
        or decision.get("decided_by") in {None, "", "automation", "codex"}
    ):
        raise PromotionGateError("human decision identity/evidence binding mismatch")
    return str(value)


__all__ = ["PromotionGateError", "freeze_candidate_record", "validate_manual_decision"]
