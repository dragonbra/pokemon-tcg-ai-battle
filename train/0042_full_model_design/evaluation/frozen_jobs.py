"""Canonical Policy-0809-only Frozen schedules for 0042."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evaluation.runtime.seeded import build_seeded_runtime

from ..league import load_frozen_catalog
from ..rollout.deck_routing import exact_deck_sha256
from ..rollout.protocol import DEFAULT_FULL_ROUND_DRAW_LIMIT, RolloutJob


CANONICAL_CONTRACT_ID = "0042_frozen_0809_seeded_agent_first_player_v1"
EVALUATION_SEED = 341_512_806
EVALUATION_UNITS = 8
UNIT_GAMES = 256
FIRST_PLAYER_CONTRACT = "seeded_coin_winner_then_winning_agent_selects_context_41"
OPPONENT_POLICY_ID = "Policy-0809"

# Pinned after generating the Policy-0809 identity-bound canonical payload.
EXPECTED_BASE_SCHEDULE_SHA256 = (
    "74ee1527686c8b2344b15a1fd2c49a7dc2386da5c8164fb92fbd969cba8900ec"
)
EXPECTED_007_SCHEDULE_SHA256 = (
    "c8d48d03d70e40332598ea14f49e1847b7e09c0e0f8f2a6130d4cd7c00a05712"
)
EXPECTED_007_U0_DEPLOYMENT_SHA256 = (
    "b9b4b162915ba64d1156e6724df9bfa29633c41ae5e6caf5afbc923202add995"
)


def _seed(
    *,
    focal_deployment_identity: str,
    opponent_effective_policy_sha256: str,
    opponent_exact_deck_sha256: str,
    slot: int,
    replica: int,
    namespace: str,
) -> int:
    if not focal_deployment_identity:
        raise ValueError("Frozen-0809 focal deployment identity must be nonempty")
    for name, value in (
        ("opponent effective policy", opponent_effective_policy_sha256),
        ("opponent exact deck", opponent_exact_deck_sha256),
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"Frozen-0809 {name} identity must be a SHA-256")
    if slot < 0 or not 0 <= replica < EVALUATION_UNITS:
        raise ValueError("Frozen-0809 slot/replica is outside the contract")
    payload = (
        EVALUATION_SEED,
        CANONICAL_CONTRACT_ID,
        namespace,
        focal_deployment_identity,
        opponent_effective_policy_sha256,
        opponent_exact_deck_sha256,
        slot,
        replica,
    )
    value = int.from_bytes(
        hashlib.sha256(":".join(map(str, payload)).encode("utf-8")).digest()[:8],
        "big",
    )
    return (value & 0x7FFFFFFF) or 1


def _canonical_payload(
    focal_deck_id: str,
    *,
    focal_deployment_identity: str,
    opponent_effective_policy_sha256: str,
    evaluation_units: int = EVALUATION_UNITS,
) -> dict:
    if evaluation_units not in {1, EVALUATION_UNITS}:
        raise ValueError("Frozen-0809 evaluation units must be CPU=1 or CUDA=8")
    jobs = []
    base_rows = []
    for opponent in load_frozen_catalog():
        deck_sha = exact_deck_sha256(opponent.deck)
        if deck_sha != opponent.deck_sha256:
            raise RuntimeError(f"exact deck hash drift: {opponent.deck_id}")
        for slot in range(opponent.games):
            base_rows.append({
                "opponent_id": opponent.deck_id,
                "opponent_exact_deck_sha256": deck_sha,
                "slot": slot,
            })
    base_payload = {
        "contract_id": CANONICAL_CONTRACT_ID,
        "pool_id": "0042_policy_0809_neutral_55_v1",
        "unit_games": UNIT_GAMES,
        "rows": base_rows,
    }
    base_sha = hashlib.sha256(
        json.dumps(base_payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    for replica in range(evaluation_units):
        for base_index, row in enumerate(base_rows):
            common = {
                "focal_deployment_identity": focal_deployment_identity,
                "opponent_effective_policy_sha256": opponent_effective_policy_sha256,
                "opponent_exact_deck_sha256": row["opponent_exact_deck_sha256"],
                "slot": base_index,
                "replica": replica,
            }
            jobs.append({
                **row,
                "replica": replica,
                "game_id": f"r{replica + 1:02d}-{row['opponent_id']}-{int(row['slot']) + 1:03d}",
                "engine_seed": _seed(**common, namespace="engine"),
                "search_seed": _seed(**common, namespace="search"),
                "policy_seed": _seed(**common, namespace="policy"),
                "focal_won_toss": bool(_seed(**common, namespace="coin-winner") & 1),
            })
    payload = {
        "schema": "0042_policy_0809_frozen_schedule_v1",
        "contract_id": CANONICAL_CONTRACT_ID,
        "first_player_contract": FIRST_PLAYER_CONTRACT,
        "evaluation_seed": EVALUATION_SEED,
        "evaluation_units": evaluation_units,
        "focal_deck_id": focal_deck_id,
        "focal_deployment_identity": focal_deployment_identity,
        "opponent_policy_id": OPPONENT_POLICY_ID,
        "opponent_effective_policy_sha256": opponent_effective_policy_sha256,
        "base_schedule_sha256": base_sha,
        "jobs": jobs,
    }
    payload["schedule_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def build_frozen_jobs(
    *,
    focal_deck_id: str,
    focal_deck: tuple[int, ...],
    runtime_root: Path,
    source_policy_update: int,
    focal_deployment_identity: str,
    opponent_effective_policy_sha256: str,
    evaluation_units: int = EVALUATION_UNITS,
) -> tuple[list[RolloutJob], str]:
    payload = _canonical_payload(
        focal_deck_id,
        focal_deployment_identity=focal_deployment_identity,
        opponent_effective_policy_sha256=opponent_effective_policy_sha256,
        evaluation_units=evaluation_units,
    )
    rows = payload["jobs"]
    catalog = {item.deck_id: item for item in load_frozen_catalog()}
    runtime = build_seeded_runtime()
    jobs = [
        RolloutJob(
            game_id=str(row["game_id"]),
            opponent_id=str(row["opponent_id"]),
            focal_first=bool(row["focal_won_toss"]),
            focal_won_toss=bool(row["focal_won_toss"]),
            seed=int(row["engine_seed"]),
            source_policy_update=source_policy_update,
            focal_deck=focal_deck,
            opponent_deck=catalog[str(row["opponent_id"])].deck,
            runtime_root=runtime_root,
            opponent_policy_id=OPPONENT_POLICY_ID,
            policy_seed=int(row["policy_seed"]),
            search_seed=int(row["search_seed"]),
            engine_library=runtime.library_path,
            full_round_draw_limit=DEFAULT_FULL_ROUND_DRAW_LIMIT,
            action_boundary_mode="enabled",
            trace_policy="errors_and_sample",
        )
        for row in rows
    ]
    expected_games = UNIT_GAMES * evaluation_units
    if len(jobs) != expected_games or len({job.seed for job in jobs}) != expected_games:
        raise RuntimeError("canonical Frozen-0809 schedule lost unique jobs")
    if EXPECTED_BASE_SCHEDULE_SHA256 and (
        payload["base_schedule_sha256"] != EXPECTED_BASE_SCHEDULE_SHA256
    ):
        raise RuntimeError("0042 Frozen-0809 base schedule hash drift")
    schedule_sha = str(payload["schedule_sha256"])
    if (
        focal_deck_id == "dragapult_ex_07bedfffbfad"
        and evaluation_units == EVALUATION_UNITS
        and focal_deployment_identity == EXPECTED_007_U0_DEPLOYMENT_SHA256
        and EXPECTED_007_SCHEDULE_SHA256
        and schedule_sha != EXPECTED_007_SCHEDULE_SHA256
    ):
        raise RuntimeError("0042 Frozen-0809 schedule hash drift")
    return jobs, schedule_sha


__all__ = [
    "CANONICAL_CONTRACT_ID",
    "EVALUATION_SEED",
    "EVALUATION_UNITS",
    "EXPECTED_007_SCHEDULE_SHA256",
    "EXPECTED_007_U0_DEPLOYMENT_SHA256",
    "EXPECTED_BASE_SCHEDULE_SHA256",
    "FIRST_PLAYER_CONTRACT",
    "OPPONENT_POLICY_ID",
    "UNIT_GAMES",
    "build_frozen_jobs",
]
