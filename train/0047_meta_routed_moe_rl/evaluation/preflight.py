"""Shared fail-closed boundary before any formal official-engine game starts."""

from __future__ import annotations

from typing import Any, Mapping


CONTRACT_ID = "kaggle_fp16_storage_fp32_runtime_v1"
CUDA_BACKEND = "cuda_engine_2_0"


class EvaluationPreflightError(RuntimeError):
    pass


def require_formal_evaluation_manifest(manifest: Mapping[str, Any]) -> None:
    candidate = manifest.get("candidate_deployment_identity_audit")
    opponent = manifest.get("opponent_policy_identity_audit")
    schedule = manifest.get("schedule")
    if not isinstance(candidate, Mapping) or candidate.get("status") != "PASS":
        raise EvaluationPreflightError("candidate deployment identity audit must PASS")
    if (
        candidate.get("contract_id") != CONTRACT_ID
        or candidate.get("storage_dtype") != "fp16"
        or candidate.get("runtime_dtype") != "fp32"
        or not isinstance(candidate.get("source_checkpoint_sha256"), str)
        or not isinstance(candidate.get("portable_checkpoint_sha256"), str)
        or not isinstance(candidate.get("effective_candidate_sha256"), str)
    ):
        raise EvaluationPreflightError("candidate deployment identity contract mismatch")
    if not isinstance(opponent, Mapping) or opponent.get("status") != "PASS":
        raise EvaluationPreflightError("opponent full-policy identity audit must PASS")
    if opponent.get("requested_policy_id") not in {"Policy-0809", "Champion-G1"}:
        raise EvaluationPreflightError("opponent policy is not registered in 0044")
    if not isinstance(schedule, Mapping):
        raise EvaluationPreflightError("formal evaluation schedule is missing")
    required = {
        "evaluation_id", "master_seed", "games", "replicas", "schedule_sha256",
        "seat_semantics", "exact_deck_pool_hash",
    }
    if required - set(schedule):
        raise EvaluationPreflightError("formal evaluation schedule identity is incomplete")
    expected_games = {"FrozenMeta256-V1": 256, "FrozenMeta2048-V1": 2048}
    if schedule["evaluation_id"] not in expected_games:
        raise EvaluationPreflightError("unregistered FrozenMeta benchmark")
    if schedule["games"] != expected_games[schedule["evaluation_id"]]:
        raise EvaluationPreflightError("formal FrozenMeta game count mismatch")
    if schedule["seat_semantics"] != "seeded_toss_winner_agent_context_41_choice":
        raise EvaluationPreflightError("formal seat-selection semantics mismatch")
    if manifest.get("selection_mode") != "greedy":
        raise EvaluationPreflightError("formal Frozen evaluation must be greedy")
    if manifest.get("official_engine") is not True:
        raise EvaluationPreflightError("formal evidence requires official engine runtime")
    if schedule["games"] == 2048:
        cuda = manifest.get("cuda_engine_identity_audit")
        if not isinstance(cuda, Mapping) or cuda.get("status") != "PASS":
            raise EvaluationPreflightError("CUDA-2048 requires CUDA Engine 2.0 identity PASS")
        required_cuda = {
            "engine_label", "engine_source_sha256", "official_state_abi",
            "official_rule_abi", "official_rule_pack_sha256",
            "compute_capability", "runtime_dtype", "semantic_codec",
            "binary_sha256", "extension_sha256",
        }
        if required_cuda - set(cuda):
            raise EvaluationPreflightError("CUDA Engine 2.0 identity is incomplete")
        if (
            cuda["engine_label"] != CUDA_BACKEND
            or cuda["runtime_dtype"] != "fp32"
            or cuda["semantic_codec"] != "semantic0031_v2"
            or cuda["official_state_abi"] != 7
            or cuda["official_rule_abi"] != 1
        ):
            raise EvaluationPreflightError("CUDA Engine 2.0 contract mismatch")
        hashes = (
            cuda["engine_source_sha256"], cuda["official_rule_pack_sha256"],
            cuda["binary_sha256"], cuda["extension_sha256"],
        )
        if any(not isinstance(value, str) or len(value) != 64 for value in hashes):
            raise EvaluationPreflightError("CUDA Engine 2.0 hashes are incomplete")
        if list(cuda["compute_capability"]) != [12, 0]:
            raise EvaluationPreflightError("CUDA-2048 is not bound to the admitted SM120 device")


__all__ = [
    "CONTRACT_ID", "CUDA_BACKEND", "EvaluationPreflightError",
    "require_formal_evaluation_manifest",
]
