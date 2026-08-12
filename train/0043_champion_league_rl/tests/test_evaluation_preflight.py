from __future__ import annotations

import importlib

import pytest


module = importlib.import_module("train.0043_champion_league_rl.evaluation.preflight")


def _manifest():
    return {
        "candidate_deployment_identity_audit": {
            "status": "PASS", "contract_id": module.CONTRACT_ID,
            "storage_dtype": "fp16", "runtime_dtype": "fp32",
            "source_checkpoint_sha256": "a" * 64,
            "portable_checkpoint_sha256": "b" * 64,
            "effective_candidate_sha256": "c" * 64,
        },
        "opponent_policy_identity_audit": {
            "status": "PASS", "requested_policy_id": "Policy-0809",
            "effective_policy_sha256": "d" * 64,
        },
        "schedule": {
            "evaluation_id": "FrozenMeta256-V1", "master_seed": 341512806,
            "games": 256, "replicas": [0], "schedule_sha256": "e" * 64,
            "seat_semantics": "seeded_toss_winner_agent_context_41_choice",
            "exact_deck_pool_hash": "f" * 64,
        },
        "selection_mode": "greedy", "official_engine": True,
    }


def test_complete_formal_preflight_passes() -> None:
    module.require_formal_evaluation_manifest(_manifest())


def test_raw_fp32_candidate_and_missing_opponent_fail() -> None:
    manifest = _manifest()
    manifest["candidate_deployment_identity_audit"]["storage_dtype"] = "fp32"
    with pytest.raises(module.EvaluationPreflightError):
        module.require_formal_evaluation_manifest(manifest)
    manifest = _manifest()
    manifest.pop("opponent_policy_identity_audit")
    with pytest.raises(module.EvaluationPreflightError):
        module.require_formal_evaluation_manifest(manifest)


def test_cuda_2048_requires_complete_cuda_engine_2_identity() -> None:
    manifest = _manifest()
    manifest["schedule"].update({
        "evaluation_id": "FrozenMeta2048-V1", "games": 2048,
        "replicas": list(range(8)),
    })
    with pytest.raises(module.EvaluationPreflightError, match="CUDA Engine 2.0 identity"):
        module.require_formal_evaluation_manifest(manifest)
    manifest["cuda_engine_identity_audit"] = {
        "status": "PASS", "engine_label": "cuda_engine_2_0",
        "engine_source_sha256": "1" * 64,
        "official_state_abi": 7, "official_rule_abi": 1,
        "official_rule_pack_sha256": "2" * 64,
        "compute_capability": [12, 0], "runtime_dtype": "fp32",
        "semantic_codec": "semantic0031_v2",
        "binary_sha256": "3" * 64, "extension_sha256": "4" * 64,
    }
    module.require_formal_evaluation_manifest(manifest)
