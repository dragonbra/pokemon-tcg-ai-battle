from __future__ import annotations

import importlib

import pytest


module = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.promote.workflow")


def test_candidate_freeze_and_explicit_decision_binding(tmp_path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"immutable candidate")
    import hashlib
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    audit = {
        "status": "PASS", "contract_id": "kaggle_fp16_storage_fp32_runtime_v1",
        "storage_dtype": "fp16", "runtime_dtype": "fp32",
        "source_checkpoint_sha256": digest,
        "portable_checkpoint_sha256": "b" * 64,
        "effective_candidate_sha256": "c" * 64,
    }
    record = module.freeze_candidate_record(
        candidate_id="Candidate-0044-update000100", source_checkpoint=checkpoint,
        source_update=100, deployment_audit=audit, parent_policy_id="Champion-G1",
    )
    decision = {
        "decision": "HOLD", "candidate_id": record["candidate_id"],
        "candidate_record_sha256": record["candidate_record_sha256"],
        "evidence_sha256": "e" * 64, "decided_by": "human_operator",
    }
    assert module.validate_manual_decision(
        decision, candidate_record=record, evidence_sha256="e" * 64
    ) == "HOLD"


def test_automation_or_mismatched_evidence_cannot_promote() -> None:
    record = {"candidate_id": "Candidate-0044-x", "candidate_record_sha256": "a" * 64}
    decision = {
        "decision": "PROMOTE", "candidate_id": record["candidate_id"],
        "candidate_record_sha256": record["candidate_record_sha256"],
        "evidence_sha256": "b" * 64, "decided_by": "codex",
    }
    with pytest.raises(module.PromotionGateError):
        module.validate_manual_decision(
            decision, candidate_record=record, evidence_sha256="b" * 64
        )
