"""Build the canonical Frozen-0806 seeded-2048 evaluation schedule."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evaluation.frozen_0806_contract import (
    FROZEN_0806_CONTRACT_ID,
    FROZEN_0806_EVALUATION_SEED,
    FROZEN_0806_EVALUATION_UNITS,
    FROZEN_0806_FIRST_PLAYER_CONTRACT,
    evaluation_coin_winner,
    evaluation_game_seed,
)
from evaluation.runtime.seeded import build_seeded_runtime

from ..league import load_frozen_catalog
from ..rollout.protocol import DEFAULT_FULL_ROUND_DRAW_LIMIT, RolloutJob


CANONICAL_CONTRACT_ID = FROZEN_0806_CONTRACT_ID
EXPECTED_007_SCHEDULE_SHA256 = (
    "7e1c79c7e9265bbc1dd5c084aa482e381366a0cd64541cd26305886ad218a77c"
)


def _canonical_payload(focal_deck_id: str) -> dict:
    jobs = []
    for replica in range(FROZEN_0806_EVALUATION_UNITS):
        for opponent in load_frozen_catalog():
            for slot in range(opponent.games):
                jobs.append({
                    "game_id": f"r{replica + 1:02d}-{opponent.deck_id}-{slot + 1:03d}",
                    "opponent_id": opponent.deck_id,
                    "replica": replica,
                    "slot": slot,
                    "engine_seed": evaluation_game_seed(
                        evaluation_seed=FROZEN_0806_EVALUATION_SEED,
                        focal_identity=focal_deck_id,
                        opponent_identity=opponent.deck_id,
                        slot=slot,
                        replica=replica,
                    ),
                    "search_seed": evaluation_game_seed(
                        evaluation_seed=FROZEN_0806_EVALUATION_SEED,
                        focal_identity=focal_deck_id,
                        opponent_identity=opponent.deck_id,
                        slot=slot,
                        replica=replica,
                        namespace="search",
                    ),
                    "focal_won_toss": evaluation_coin_winner(
                        evaluation_seed=FROZEN_0806_EVALUATION_SEED,
                        focal_identity=focal_deck_id,
                        opponent_identity=opponent.deck_id,
                        slot=slot,
                        replica=replica,
                    ),
                })
    payload = {
        "schema": "policy_0806_cuda_seeded2048_agent_choice_v3",
        "contract_id": CANONICAL_CONTRACT_ID,
        "first_player_contract": FROZEN_0806_FIRST_PLAYER_CONTRACT,
        "evaluation_seed": FROZEN_0806_EVALUATION_SEED,
        "focal_deck_id": focal_deck_id,
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
) -> tuple[list[RolloutJob], str]:
    payload = _canonical_payload(focal_deck_id)
    rows = payload["jobs"]
    catalog = {item.deck_id: item for item in load_frozen_catalog()}
    runtime = build_seeded_runtime()
    jobs = [
        RolloutJob(
            game_id=str(row["game_id"]),
            opponent_id=str(row["opponent_id"]),
            # Retained only for legacy non-Frozen callers.  Frozen collectors
            # must use focal_won_toss and let context 41 determine the seat.
            focal_first=bool(row["focal_won_toss"]),
            focal_won_toss=bool(row["focal_won_toss"]),
            seed=int(row["engine_seed"]),
            source_policy_update=source_policy_update,
            focal_deck=focal_deck,
            opponent_deck=catalog[str(row["opponent_id"])].deck,
            runtime_root=runtime_root,
            policy_seed=evaluation_game_seed(
                evaluation_seed=FROZEN_0806_EVALUATION_SEED,
                focal_identity=focal_deck_id,
                opponent_identity=str(row["opponent_id"]),
                slot=int(row["slot"]),
                replica=int(row["replica"]),
                namespace="policy",
            ),
            search_seed=int(row["search_seed"]),
            engine_library=runtime.library_path,
            full_round_draw_limit=DEFAULT_FULL_ROUND_DRAW_LIMIT,
            action_boundary_mode="enabled",
            trace_policy="errors_and_sample",
        )
        for row in rows
    ]
    if len(jobs) != 2048 or len({job.seed for job in jobs}) != 2048:
        raise RuntimeError("canonical Frozen schedule lost its 2,048 unique jobs")
    schedule_sha = str(payload["schedule_sha256"])
    if focal_deck_id == "dragapult_ex_07bedfffbfad" \
            and schedule_sha != EXPECTED_007_SCHEDULE_SHA256:
        raise RuntimeError("0038 Frozen schedule drifted from the standard 007 report")
    return jobs, schedule_sha


__all__ = [
    "CANONICAL_CONTRACT_ID",
    "EXPECTED_007_SCHEDULE_SHA256",
    "build_frozen_jobs",
]
