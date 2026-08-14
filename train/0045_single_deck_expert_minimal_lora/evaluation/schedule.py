"""Identity-bound FrozenMeta256 / FrozenMeta2048 schedule materialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..assets import AssetRegistry


CONTRACT_ID = "0044_frozen_meta_seeded_agent_first_player_v1"
MASTER_SEED = 341_512_806
UNIT_GAMES = 256
CUDA_REPLICAS = 8
FIRST_PLAYER_CONTRACT = "seeded_coin_winner_then_winning_agent_selects_context_41"


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _seed(*, focal: str, opponent: str, deck: str, slot: int, replica: int, namespace: str) -> int:
    if not all(len(value) == 64 and set(value) <= set("0123456789abcdef") for value in (focal, opponent, deck)):
        raise ValueError("Frozen schedule identities must be lowercase SHA-256 values")
    payload = (MASTER_SEED, CONTRACT_ID, namespace, focal, opponent, deck, slot, replica)
    value = int.from_bytes(hashlib.sha256(":".join(map(str, payload)).encode()).digest()[:8], "big")
    return (value & 0x7FFFFFFF) or 1


def materialize(
    project_root: Path, *, focal_deck_id: str, focal_deployment_identity: str,
    opponent_policy_id: str = "Policy-0809", replicas: int = 1,
) -> dict[str, Any]:
    if replicas not in {1, CUDA_REPLICAS}:
        raise ValueError("Frozen evaluation replicas must be CPU=1 or CUDA=8")
    assets = AssetRegistry.load(project_root)
    assets.validate_all()
    evaluation = next(item for item in assets.evaluations if item.evaluation_id == "FrozenMeta256-V1")
    opponent = next(item for item in assets.policies if item.policy_id == opponent_policy_id)
    if len(focal_deployment_identity) != 64 or set(focal_deployment_identity) > set("0123456789abcdef"):
        raise ValueError("focal deployment identity must be a lowercase SHA-256")
    rows: list[dict[str, Any]] = []
    for entry in evaluation.entries:
        rows.extend(
            {"opponent_deck_id": entry.deck_id, "opponent_exact_deck_sha256": entry.exact_deck_sha256, "deck_slot": slot}
            for slot in range(entry.games)
        )
    if len(rows) != UNIT_GAMES:
        raise RuntimeError("FrozenMeta256 composition did not materialize exactly 256 rows")
    base = {
        "contract_id": CONTRACT_ID,
        "pool_id": "0044_policy_0809_numeric_55_v1",
        "unit_games": UNIT_GAMES,
        "rows": rows,
    }
    jobs = []
    for replica in range(replicas):
        for absolute_slot, row in enumerate(rows):
            common = dict(
                focal=focal_deployment_identity,
                opponent=opponent.effective_policy_sha256,
                deck=row["opponent_exact_deck_sha256"],
                slot=absolute_slot,
                replica=replica,
            )
            jobs.append({
                **row,
                "replica": replica,
                "game_id": f"r{replica + 1:02d}-{row['opponent_deck_id']}-{row['deck_slot'] + 1:03d}",
                "engine_seed": _seed(**common, namespace="engine"),
                "search_seed": _seed(**common, namespace="search"),
                "policy_seed": _seed(**common, namespace="policy"),
                "focal_won_toss": bool(_seed(**common, namespace="coin-winner") & 1),
            })
    payload = {
        "schema_version": "0044_frozen_schedule_v1",
        "contract_id": CONTRACT_ID,
        "first_player_contract": FIRST_PLAYER_CONTRACT,
        "master_seed": MASTER_SEED,
        "replicas": replicas,
        "focal_deck_id": focal_deck_id,
        "focal_deployment_identity": focal_deployment_identity,
        "opponent_policy_id": opponent_policy_id,
        "opponent_effective_policy_sha256": opponent.effective_policy_sha256,
        "base_schedule_sha256": _sha(base),
        "jobs": jobs,
    }
    payload["schedule_sha256"] = _sha(payload)
    if len(jobs) != UNIT_GAMES * replicas:
        raise RuntimeError("Frozen schedule game count mismatch")
    for key in ("engine_seed", "search_seed", "policy_seed"):
        if len({job[key] for job in jobs}) != len(jobs):
            raise RuntimeError(f"Frozen schedule has duplicate {key}")
    return payload


__all__ = ["CONTRACT_ID", "CUDA_REPLICAS", "FIRST_PLAYER_CONTRACT", "MASTER_SEED", "UNIT_GAMES", "materialize"]
