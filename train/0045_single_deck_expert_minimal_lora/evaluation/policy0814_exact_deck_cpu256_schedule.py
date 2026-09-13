"""Half-scale CPU-engine schedule derived from the V18 Policy-0814 Eval512."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from . import policy0814_exact_deck_schedule as source_schedule


CONTRACT_ID = "0045_v18_distribution_cpu256_gpu_inference_diagnostic_v1"
SCHEMA_VERSION = "0045_policy0814_exact_deck_cpu256_schedule_v1"
GAMES = 256
MASTER_SEED = source_schedule.MASTER_SEED
OPPONENT_POLICY_ID = source_schedule.OPPONENT_POLICY_ID
FIRST_PLAYER_CONTRACT = source_schedule.FIRST_PLAYER_CONTRACT


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _halved_counts(realized: dict[str, int]) -> dict[str, int]:
    target = {deck_id: count // 2 for deck_id, count in sorted(realized.items())}
    remaining = GAMES - sum(target.values())
    odd_decks = sorted(deck_id for deck_id, count in realized.items() if count % 2)
    if not 0 <= remaining <= len(odd_decks):
        raise RuntimeError("V18 realized counts cannot be halved to 256 games")
    for deck_id in odd_decks[:remaining]:
        target[deck_id] += 1
    return target


def materialize(
    project_root: Path,
    *,
    focal_deployment_identity: str,
) -> dict[str, Any]:
    if len(focal_deployment_identity) != 64:
        raise ValueError("focal deployment identity must be a SHA-256")
    source = source_schedule.materialize(
        project_root,
        focal_deck_id="003",
        focal_deployment_identity=focal_deployment_identity,
    )
    source_counts = dict(source["realized_deck_counts"])
    target_counts = _halved_counts(source_counts)
    selected_counts: Counter[str] = Counter()
    jobs: list[dict[str, Any]] = []
    for source_job in source["jobs"]:
        deck_id = str(source_job["opponent_deck_id"])
        if selected_counts[deck_id] >= target_counts[deck_id]:
            continue
        selected_counts[deck_id] += 1
        jobs.append({
            **source_job,
            "game_id": f"v18-cpu256-{len(jobs) + 1:04d}-{deck_id}",
            "schedule_slot": len(jobs),
            "source_schedule_slot": source_job["schedule_slot"],
        })
    realized = dict(sorted(selected_counts.items()))
    source_jobs_sha256 = _hash(source["jobs"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "master_seed": MASTER_SEED,
        "games": GAMES,
        "first_player_contract": FIRST_PLAYER_CONTRACT,
        "common_random_numbers": True,
        "seed_derivation_excludes": ["focal_deployment_identity"],
        "focal_deck_id": "003",
        "focal_deployment_identity": focal_deployment_identity,
        "opponent_policy_id": OPPONENT_POLICY_ID,
        "opponent_effective_policy_sha256": source[
            "opponent_effective_policy_sha256"
        ],
        "halving_method": "floor_then_deck_id_tie_break_on_odd_realized_counts",
        "target_deck_counts": target_counts,
        "realized_deck_counts": realized,
        "source_schedule": {
            "schema_version": source["schema_version"],
            "contract_id": source["contract_id"],
            "master_seed": source["master_seed"],
            "games": source["games"],
            "realized_deck_counts": source_counts,
            "jobs_sha256": source_jobs_sha256,
        },
        "jobs": jobs,
    }
    hash_payload = {
        key: value
        for key, value in payload.items()
        if key != "focal_deployment_identity"
    }
    payload["schedule_sha256"] = _hash(hash_payload)
    if len(jobs) != GAMES or realized != target_counts:
        raise RuntimeError("CPU256 schedule inventory changed")
    if len({row["source_schedule_slot"] for row in jobs}) != GAMES:
        raise RuntimeError("CPU256 source slots are not unique")
    return payload


__all__ = [
    "CONTRACT_ID",
    "FIRST_PLAYER_CONTRACT",
    "GAMES",
    "MASTER_SEED",
    "OPPONENT_POLICY_ID",
    "SCHEMA_VERSION",
    "materialize",
]
