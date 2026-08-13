"""Identity-bound Meta-balanced Champion-G2 CUDA-2048 schedule."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

from ..assets import AssetRegistry, sha256_file
from ..own_archetype import OwnArchetypeVocabulary


CONTRACT_ID = "0044_benchmark_v1_meta_balanced_g2_cuda2048_v1"
MASTER_SEED = 341_512_902
GAMES = 2048
OPPONENT_POLICY_ID = "Champion-G2"
FIRST_PLAYER_CONTRACT = "seeded_coin_winner_then_winning_agent_selects_context_41"


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _seed(*, focal: str, opponent: str, deck: str, slot: int, namespace: str) -> int:
    values = (focal, opponent, deck)
    if any(len(value) != 64 or set(value) > set("0123456789abcdef") for value in values):
        raise ValueError("Benchmark V1 identities must be lowercase SHA-256 values")
    payload = (MASTER_SEED, CONTRACT_ID, namespace, focal, opponent, deck, slot)
    value = int.from_bytes(hashlib.sha256(":".join(map(str, payload)).encode()).digest()[:8], "big")
    return (value & 0x7FFFFFFF) or 1


def _balanced_counts(ids: list[int | str], total: int, *, seed: int) -> dict[int | str, int]:
    if not ids or total < len(ids):
        raise ValueError("balanced allocation requires at least one row per identity")
    base, remainder = divmod(total, len(ids))
    rotated = list(ids)
    random.Random(seed).shuffle(rotated)
    selected = set(rotated[:remainder])
    return {identity: base + int(identity in selected) for identity in ids}


def materialize(
    project_root: Path, *, focal_deck_id: str, focal_deployment_identity: str,
) -> dict[str, Any]:
    registry = AssetRegistry.load(project_root)
    registry.validate_all()
    opponent = next(row for row in registry.policies if row.policy_id == OPPONENT_POLICY_ID)
    if registry.latest_champion_policy_id != OPPONENT_POLICY_ID:
        raise RuntimeError("Benchmark V1 fixed opponent is not the registered latest Champion-G2")
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=project_root
    )
    by_class: dict[int, list[str]] = {index: [] for index in range(vocabulary.class_count)}
    for row in vocabulary.mappings:
        by_class[row.archetype_id].append(row.deck_id)
    nonempty = [class_id for class_id, deck_ids in by_class.items() if deck_ids]
    class_counts = _balanced_counts(nonempty, GAMES, seed=MASTER_SEED)
    deck_assets = {row.deck_id: row for row in registry.decks}
    rows: list[dict[str, Any]] = []
    for class_id in nonempty:
        deck_ids = sorted(by_class[class_id])
        deck_counts = _balanced_counts(
            deck_ids, int(class_counts[class_id]), seed=MASTER_SEED + class_id + 1
        )
        for deck_id in deck_ids:
            rows.extend({
                "opponent_meta_archetype_id": class_id,
                "opponent_deck_id": deck_id,
                "opponent_exact_deck_sha256": deck_assets[deck_id].content_sha256,
                "class_deck_slot": slot,
            } for slot in range(int(deck_counts[deck_id])))
    random.Random(MASTER_SEED).shuffle(rows)
    if len(rows) != GAMES:
        raise RuntimeError("Benchmark V1 schedule is not exact CUDA-2048")
    jobs = []
    for slot, row in enumerate(rows):
        common = {
            "focal": focal_deployment_identity,
            "opponent": opponent.effective_policy_sha256,
            "deck": row["opponent_exact_deck_sha256"],
            "slot": slot,
        }
        jobs.append({
            **row, "schedule_slot": slot,
            "game_id": f"b1-{slot + 1:04d}-{row['opponent_deck_id']}",
            "engine_seed": _seed(**common, namespace="engine"),
            "search_seed": _seed(**common, namespace="search"),
            "policy_seed": _seed(**common, namespace="policy"),
            "focal_won_toss": bool(_seed(**common, namespace="coin-winner") & 1),
        })
    payload = {
        "schema_version": "0044_benchmark_v1_schedule_v1",
        "contract_id": CONTRACT_ID,
        "master_seed": MASTER_SEED,
        "games": GAMES,
        "first_player_contract": FIRST_PLAYER_CONTRACT,
        "focal_deck_id": focal_deck_id,
        "focal_deployment_identity": focal_deployment_identity,
        "opponent_policy_id": OPPONENT_POLICY_ID,
        "opponent_effective_policy_sha256": opponent.effective_policy_sha256,
        "taxonomy_sha256": vocabulary.taxonomy_sha256,
        "mapping_sha256": vocabulary.mapping_sha256,
        "class_counts": {str(key): value for key, value in class_counts.items()},
        "empty_class_ids": [key for key, values in by_class.items() if not values],
        "jobs": jobs,
    }
    payload["schedule_sha256"] = _hash(payload)
    for name in ("engine_seed", "search_seed", "policy_seed"):
        if len({row[name] for row in jobs}) != GAMES:
            raise RuntimeError(f"Benchmark V1 contains duplicate {name}")
    return payload


__all__ = [
    "CONTRACT_ID", "FIRST_PLAYER_CONTRACT", "GAMES", "MASTER_SEED",
    "OPPONENT_POLICY_ID", "materialize",
]
