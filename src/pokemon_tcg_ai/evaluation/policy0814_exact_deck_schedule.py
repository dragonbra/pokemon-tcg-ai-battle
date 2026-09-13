"""Frozen common-random-number schedule for the V13 Policy-0814 eval512."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from ..assets import AssetRegistry
from ..league.exact_deck_quota import exact_deck_quota_schedule
from ..own_archetype import OwnArchetypeVocabulary


CONTRACT_ID = "0045_v13_policy0814_exact_deck_common_seeds_cuda512_v1"
MASTER_SEED = 341_512_814
GAMES = 512
OPPONENT_POLICY_ID = "Policy-0814"
FIRST_PLAYER_CONTRACT = "seeded_coin_winner_then_winning_agent_selects_context_41"
FIXED_DECK_QUOTAS = {
    "001": 143, "002": 68, "003": 48, "007": 64,
    "071": 30, "008": 14, "009": 16, "011": 7,
}
RANDOM_DECK_IDS = ("071", "008", "009", "011")


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _seed(opponent_hash: str, deck_hash: str, slot: int, namespace: str) -> int:
    payload = f"{MASTER_SEED}:{CONTRACT_ID}:{namespace}:{opponent_hash}:{deck_hash}:{slot}"
    value = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")
    return (value & 0x7FFFFFFF) or 1


def materialize(
    project_root: Path, *, focal_deck_id: str, focal_deployment_identity: str,
) -> dict[str, Any]:
    if len(focal_deployment_identity) != 64:
        raise ValueError("focal deployment identity must be a SHA-256")
    registry = AssetRegistry.load(project_root)
    registry.validate_all()
    opponent = next(row for row in registry.policies if row.policy_id == OPPONENT_POLICY_ID)
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=project_root
    )
    assets = {row.deck_id: row for row in registry.decks}
    slots = exact_deck_quota_schedule(
        random_seed=MASTER_SEED,
        shuffle_seed=MASTER_SEED + 1,
        mappings=vocabulary.mappings,
        lanes=GAMES,
        fixed_deck_quotas=FIXED_DECK_QUOTAS,
        random_deck_ids=RANDOM_DECK_IDS,
    )
    jobs = []
    for slot in slots:
        asset = assets[slot.deck_id]
        common = (opponent.effective_policy_sha256, asset.content_sha256, slot.lane_id)
        coin_seed = _seed(*common, "coin-winner")
        jobs.append({
            "game_id": f"v13-eval-{slot.lane_id + 1:04d}-{slot.deck_id}",
            "schedule_slot": slot.lane_id,
            "opponent_deck_id": slot.deck_id,
            "opponent_meta_archetype_id": slot.archetype_id,
            "opponent_exact_deck_sha256": asset.content_sha256,
            "engine_seed": _seed(*common, "engine"),
            "search_seed": _seed(*common, "search"),
            "policy_seed": _seed(*common, "policy"),
            "coin_winner_seed": coin_seed,
            "focal_won_toss": bool(coin_seed & 1),
        })
    counts = dict(sorted(Counter(row["opponent_deck_id"] for row in jobs).items()))
    payload = {
        "schema_version": "0045_v13_policy0814_exact_deck_schedule_v1",
        "contract_id": CONTRACT_ID,
        "master_seed": MASTER_SEED,
        "games": GAMES,
        "first_player_contract": FIRST_PLAYER_CONTRACT,
        "common_random_numbers": True,
        "seed_derivation_excludes": ["focal_deployment_identity"],
        "focal_deck_id": focal_deck_id,
        "focal_deployment_identity": focal_deployment_identity,
        "opponent_policy_id": OPPONENT_POLICY_ID,
        "opponent_effective_policy_sha256": opponent.effective_policy_sha256,
        "fixed_deck_quotas": FIXED_DECK_QUOTAS,
        "random_remainder_deck_ids": list(RANDOM_DECK_IDS),
        "realized_deck_counts": counts,
        "jobs": jobs,
    }
    payload["schedule_sha256"] = _hash(payload)
    if len(jobs) != GAMES or set(counts) != set(FIXED_DECK_QUOTAS):
        raise RuntimeError("V13 eval schedule inventory changed")
    if any(counts[key] < value for key, value in FIXED_DECK_QUOTAS.items()):
        raise RuntimeError("V13 eval fixed deck quota changed")
    return payload


__all__ = [
    "CONTRACT_ID", "FIXED_DECK_QUOTAS", "FIRST_PLAYER_CONTRACT", "GAMES",
    "MASTER_SEED", "OPPONENT_POLICY_ID", "RANDOM_DECK_IDS", "materialize",
]
