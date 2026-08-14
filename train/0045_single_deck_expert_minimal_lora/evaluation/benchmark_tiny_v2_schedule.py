"""0045 meta-balanced CUDA-512 Benchmark Tiny V2 schedule."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

from ..assets import AssetRegistry
from ..own_archetype import OwnArchetypeVocabulary


CONTRACT_ID = "0045_benchmark_tiny_v2_core16_meta_balanced_common_seeds_cuda512_v1"
MASTER_SEED = 341_512_945
GAMES = 512
# The exact immutable opponent is injected only after the G2-vs-active-G3
# governance ambiguity has been resolved.  A placeholder may never materialize.
OPPONENT_POLICY_ID = "UNRESOLVED_GENERALIST"
SELECTED_CLASS_IDS = tuple(range(14)) + (17, 27)
FIRST_PLAYER_CONTRACT = "seeded_coin_winner_then_winning_agent_selects_context_41"
FROZEN_TAXONOMY_SHA256 = "9d9bc5cb95d253731270b064d27279448e1a099e0f548dfeb6bc8215bd891f60"
FROZEN_MAPPING_SHA256 = "6b7d97027d465031344d6851f8826fcf3955e2a9cc9fbd8ba49ff32fd6cb5120"


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _seed(*, opponent: str, deck: str, slot: int, namespace: str) -> int:
    if any(
        len(value) != 64 or set(value) > set("0123456789abcdef")
        for value in (opponent, deck)
    ):
        raise ValueError("Benchmark V2 identities must be lowercase SHA-256 values")
    payload = (MASTER_SEED, CONTRACT_ID, namespace, opponent, deck, slot)
    value = int.from_bytes(
        hashlib.sha256(":".join(map(str, payload)).encode()).digest()[:8], "big"
    )
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
    opponent_policy_id: str,
) -> dict[str, Any]:
    if len(focal_deployment_identity) != 64 or set(focal_deployment_identity) > set("0123456789abcdef"):
        raise ValueError("focal deployment identity must be lowercase SHA-256")
    if opponent_policy_id not in {"Champion-G2", "Champion-G3"}:
        raise ValueError("0045 Tiny V2 opponent must be explicitly resolved to Champion-G2 or Champion-G3")
    registry = AssetRegistry.load(project_root)
    registry.validate_all()
    opponent = next(row for row in registry.policies if row.policy_id == opponent_policy_id)
    vocabulary = OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=project_root
    )
    # Benchmark V2 is an immutable historical comparison contract. The live
    # training pool may append decks, but its opponent identities stay 001-067.
    benchmark_deck_ids = {f"{value:03d}" for value in range(1, 68)}
    by_class: dict[int, list[str]] = {index: [] for index in range(vocabulary.class_count)}
    for row in vocabulary.mappings:
        if row.deck_id in benchmark_deck_ids:
            by_class[row.archetype_id].append(row.deck_id)
    if any(not by_class[class_id] for class_id in SELECTED_CLASS_IDS):
        raise RuntimeError("Benchmark V2 selected an empty Meta Archetype")
    class_counts = _balanced_counts(list(SELECTED_CLASS_IDS), GAMES, seed=MASTER_SEED)
    deck_assets = {row.deck_id: row for row in registry.decks}
    rows: list[dict[str, Any]] = []
    for class_id in SELECTED_CLASS_IDS:
        deck_ids = sorted(by_class[class_id])
        deck_counts = _balanced_counts(
            deck_ids, int(class_counts[class_id]), seed=MASTER_SEED + class_id + 1
        )
        for deck_id in deck_ids:
            rows.extend({
                "opponent_meta_archetype_id": class_id,
                "opponent_deck_id": deck_id,
                "opponent_exact_deck_sha256": deck_assets[deck_id].content_sha256,
                "class_deck_slot": class_slot,
            } for class_slot in range(int(deck_counts[deck_id])))
    random.Random(MASTER_SEED).shuffle(rows)
    jobs = []
    for slot, row in enumerate(rows):
        common = {
            "opponent": opponent.effective_policy_sha256,
            "deck": row["opponent_exact_deck_sha256"],
            "slot": slot,
        }
        coin_seed = _seed(**common, namespace="coin-winner")
        jobs.append({
            **row,
            "schedule_slot": slot,
            "game_id": f"b2-{slot + 1:04d}-{row['opponent_deck_id']}",
            "engine_seed": _seed(**common, namespace="engine"),
            "search_seed": _seed(**common, namespace="search"),
            "policy_seed": _seed(**common, namespace="policy"),
            "coin_winner_seed": coin_seed,
            "focal_won_toss": bool(coin_seed & 1),
        })
    common_payload = {
        "contract_id": CONTRACT_ID,
        "master_seed": MASTER_SEED,
        "opponent_effective_policy_sha256": opponent.effective_policy_sha256,
        "taxonomy_sha256": FROZEN_TAXONOMY_SHA256,
        "mapping_sha256": FROZEN_MAPPING_SHA256,
        "selected_class_ids": list(SELECTED_CLASS_IDS),
        "jobs": jobs,
    }
    payload = {
        "schema_version": "0045_benchmark_tiny_v2_schedule_v1",
        "contract_id": CONTRACT_ID,
        "master_seed": MASTER_SEED,
        "games": GAMES,
        "first_player_contract": FIRST_PLAYER_CONTRACT,
        "common_random_numbers": True,
        "seed_derivation_excludes": ["focal_deployment_identity"],
        "focal_deck_id": focal_deck_id,
        "focal_deployment_identity": focal_deployment_identity,
        "opponent_policy_id": opponent_policy_id,
        "opponent_effective_policy_sha256": opponent.effective_policy_sha256,
        "taxonomy_sha256": FROZEN_TAXONOMY_SHA256,
        "mapping_sha256": FROZEN_MAPPING_SHA256,
        "selected_class_ids": list(SELECTED_CLASS_IDS),
        "excluded_class_ids": [
            class_id for class_id in range(vocabulary.class_count)
            if class_id not in SELECTED_CLASS_IDS
        ],
        "class_counts": {str(key): value for key, value in class_counts.items()},
        "common_random_schedule_sha256": _hash(common_payload),
        "jobs": jobs,
    }
    payload["schedule_sha256"] = _hash(payload)
    if len(jobs) != GAMES:
        raise RuntimeError("Benchmark Tiny V2 schedule is not exact CUDA-512")
    for name in ("engine_seed", "search_seed", "policy_seed", "coin_winner_seed"):
        if len({row[name] for row in jobs}) != GAMES:
            raise RuntimeError(f"Benchmark V2 contains duplicate {name}")
    return payload


__all__ = [
    "CONTRACT_ID", "FIRST_PLAYER_CONTRACT", "FROZEN_MAPPING_SHA256",
    "FROZEN_TAXONOMY_SHA256", "GAMES", "MASTER_SEED",
    "OPPONENT_POLICY_ID", "SELECTED_CLASS_IDS", "materialize",
]
