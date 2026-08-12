"""Shared fail-closed boundary before any formal official-engine game starts."""

from __future__ import annotations

from typing import Any, Mapping


CONTRACT_ID = "kaggle_fp16_storage_fp32_runtime_v1"


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
        raise EvaluationPreflightError("opponent policy is not registered in 0043")
    if not isinstance(schedule, Mapping):
        raise EvaluationPreflightError("formal evaluation schedule is missing")
    required = {
        "evaluation_id", "master_seed", "games", "replicas", "schedule_sha256",
        "seat_semantics", "exact_deck_pool_hash",
    }
    if required - set(schedule):
        raise EvaluationPreflightError("formal evaluation schedule identity is incomplete")
    if schedule["evaluation_id"] not in {"FrozenMeta256-V1", "FrozenMeta2048-V1"}:
        raise EvaluationPreflightError("unregistered FrozenMeta benchmark")
    if schedule["games"] not in {256, 2048}:
        raise EvaluationPreflightError("formal FrozenMeta game count mismatch")
    if schedule["seat_semantics"] != "seeded_toss_winner_agent_context_41_choice":
        raise EvaluationPreflightError("formal seat-selection semantics mismatch")
    if manifest.get("selection_mode") != "greedy":
        raise EvaluationPreflightError("formal Frozen evaluation must be greedy")
    if manifest.get("official_engine") is not True:
        raise EvaluationPreflightError("formal evidence requires official engine runtime")


__all__ = ["CONTRACT_ID", "EvaluationPreflightError", "require_formal_evaluation_manifest"]
