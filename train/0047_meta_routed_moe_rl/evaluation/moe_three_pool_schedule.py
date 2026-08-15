"""Two fixed Policy-0814 CUDA-512 pools with a split focus sentinel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

from ..assets import AssetRegistry
from ..own_archetype import OwnArchetypeVocabulary


CONTRACT_ID = "0047_policy0814_focus256x2_remain512_v3"
MASTER_SEED = 341_512_947
GAMES_PER_POOL = 512
FOCUS_GROUP_GAMES = 256
OPPONENT_POLICY_ID = "Policy-0814"
FIRST_PLAYER_CONTRACT = "seeded_coin_winner_then_winning_agent_selects_context_41"
FROZEN_TAXONOMY_SHA256 = "9d9bc5cb95d253731270b064d27279448e1a099e0f548dfeb6bc8215bd891f60"
FROZEN_MAPPING_SHA256 = "6b7d97027d465031344d6851f8826fcf3955e2a9cc9fbd8ba49ff32fd6cb5120"
POOL_DECK_IDS = {
    "old_three": ("001", "002", "011"),
    "new_four": ("007", "003", "009", "023"),
}


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _seed(
    *, opponent: str, deck: str, pool: str, slot: int, namespace: str,
) -> int:
    payload = (MASTER_SEED, CONTRACT_ID, namespace, pool, opponent, deck, slot)
    value = int.from_bytes(
        hashlib.sha256(":".join(map(str, payload)).encode()).digest()[:8], "big"
    )
    return (value & 0x7FFFFFFF) or 1


def _balanced_counts(
    identities: list[int | str], total: int, *, seed: int,
) -> dict[int | str, int]:
    if not identities or total < len(identities):
        raise ValueError("balanced allocation requires one row per identity")
    base, remainder = divmod(total, len(identities))
    order = list(identities)
    random.Random(seed).shuffle(order)
    extras = set(order[:remainder])
    return {identity: base + int(identity in extras) for identity in identities}


def materialize(
    project_root: Path, *, focal_deck_id: str, focal_deployment_identity: str,
) -> dict[str, Any]:
    if (
        len(focal_deployment_identity) != 64
        or set(focal_deployment_identity) > set("0123456789abcdef")
    ):
        raise ValueError("focal deployment identity must be lowercase SHA-256")
    registry = AssetRegistry.load(project_root)
    registry.validate_all()
    opponent = next(
        row for row in registry.policies if row.policy_id == OPPONENT_POLICY_ID
    )
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=project_root
    )
    by_class: dict[int, list[str]] = {
        class_id: [] for class_id in range(vocabulary.class_count)
    }
    for row in vocabulary.mappings:
        by_class[row.archetype_id].append(row.deck_id)
    meta_by_deck = {
        row.deck_id: row.archetype_id for row in vocabulary.mappings
    }
    focused_decks = set(POOL_DECK_IDS["old_three"]) | set(POOL_DECK_IDS["new_four"])
    remain_by_class = {
        class_id: [deck_id for deck_id in decks if deck_id not in focused_decks]
        for class_id, decks in by_class.items()
    }
    remain_by_class = {
        class_id: decks for class_id, decks in remain_by_class.items() if decks
    }

    deck_assets = {row.deck_id: row for row in registry.decks}
    pools: dict[str, dict[str, Any]] = {}
    all_jobs: list[dict[str, Any]] = []
    for pool_index, pool_name in enumerate(("focus_seven", "remain_meta")):
        pool_seed = MASTER_SEED + 10_000 * pool_index
        rows: list[dict[str, Any]] = []
        focus_group_counts: dict[str, int] = {}
        if pool_name == "focus_seven":
            class_counts: dict[int, int] = {}
            for group_index, (group_name, configured) in enumerate(
                POOL_DECK_IDS.items()
            ):
                deck_ids = list(configured)
                deck_counts = _balanced_counts(
                    deck_ids, FOCUS_GROUP_GAMES,
                    seed=pool_seed + 1_000 * group_index,
                )
                focus_group_counts[group_name] = sum(deck_counts.values())
                for deck_id in deck_ids:
                    count = int(deck_counts[deck_id])
                    class_id = meta_by_deck[deck_id]
                    class_counts[class_id] = class_counts.get(class_id, 0) + count
                    rows.extend({
                        "pool": pool_name,
                        "focus_group": group_name,
                        "opponent_meta_archetype_id": class_id,
                        "opponent_deck_id": deck_id,
                        "opponent_exact_deck_sha256": deck_assets[deck_id].content_sha256,
                        "class_deck_slot": class_slot,
                    } for class_slot in range(count))
            class_ids = tuple(sorted(class_counts))
        else:
            class_ids = tuple(sorted(remain_by_class))
            class_counts = _balanced_counts(
                list(class_ids), GAMES_PER_POOL, seed=pool_seed
            )
            for class_id in class_ids:
                deck_ids = sorted(remain_by_class[class_id])
                deck_counts = _balanced_counts(
                    deck_ids, int(class_counts[class_id]),
                    seed=pool_seed + class_id + 1,
                )
                for deck_id in deck_ids:
                    rows.extend({
                        "pool": pool_name,
                        "focus_group": None,
                        "opponent_meta_archetype_id": class_id,
                        "opponent_deck_id": deck_id,
                        "opponent_exact_deck_sha256": deck_assets[deck_id].content_sha256,
                        "class_deck_slot": class_slot,
                    } for class_slot in range(int(deck_counts[deck_id])))
        random.Random(pool_seed).shuffle(rows)
        jobs: list[dict[str, Any]] = []
        for slot, row in enumerate(rows):
            common = {
                "opponent": opponent.effective_policy_sha256,
                "deck": row["opponent_exact_deck_sha256"],
                "pool": pool_name,
                "slot": slot,
            }
            coin_seed = _seed(**common, namespace="coin-winner")
            jobs.append({
                **row,
                "schedule_slot": slot,
                "game_id": f"tp-{pool_name}-{slot + 1:04d}-{row['opponent_deck_id']}",
                "engine_seed": _seed(**common, namespace="engine"),
                "search_seed": _seed(**common, namespace="search"),
                "policy_seed": _seed(**common, namespace="policy"),
                "coin_winner_seed": coin_seed,
                "focal_won_toss": bool(coin_seed & 1),
            })
        common_payload = {
            "contract_id": CONTRACT_ID,
            "pool": pool_name,
            "master_seed": MASTER_SEED,
            "opponent_effective_policy_sha256": opponent.effective_policy_sha256,
            "selected_class_ids": list(class_ids),
            "jobs": jobs,
        }
        pool = {
            "pool_id": pool_name,
            "games": GAMES_PER_POOL,
            "selected_class_ids": list(class_ids),
            "selected_exact_deck_ids": (
                sorted(focused_decks)
                if pool_name == "focus_seven" else sorted(
                    deck_id for decks in remain_by_class.values() for deck_id in decks
                )
            ),
            "focus_group_counts": focus_group_counts,
            "class_counts": {str(key): value for key, value in class_counts.items()},
            "common_random_schedule_sha256": _hash(common_payload),
            "jobs": jobs,
        }
        pools[pool_name] = pool
        all_jobs.extend(jobs)

    payload = {
        "schema_version": "0047_policy0814_two_pool_schedule_v1",
        "contract_id": CONTRACT_ID,
        "master_seed": MASTER_SEED,
        "games_per_pool": GAMES_PER_POOL,
        "games": len(all_jobs),
        "first_player_contract": FIRST_PLAYER_CONTRACT,
        "common_random_numbers": True,
        "seed_derivation_excludes": ["focal_deployment_identity"],
        "focal_deck_id": focal_deck_id,
        "focal_deployment_identity": focal_deployment_identity,
        "opponent_policy_id": OPPONENT_POLICY_ID,
        "opponent_effective_policy_sha256": opponent.effective_policy_sha256,
        "taxonomy_sha256": FROZEN_TAXONOMY_SHA256,
        "mapping_sha256": FROZEN_MAPPING_SHA256,
        "focus_exact_deck_ids": sorted(focused_decks),
        "pools": pools,
    }
    payload["schedule_sha256"] = _hash(payload)
    if len(all_jobs) != 2 * GAMES_PER_POOL:
        raise RuntimeError("two-pool schedule is not exactly 2 x CUDA-512")
    return payload


__all__ = [
    "CONTRACT_ID", "FIRST_PLAYER_CONTRACT", "FOCUS_GROUP_GAMES",
    "GAMES_PER_POOL", "MASTER_SEED",
    "OPPONENT_POLICY_ID", "POOL_DECK_IDS", "materialize",
]
