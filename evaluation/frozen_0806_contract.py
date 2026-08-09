"""Canonical seeded evaluation contract for the immutable Frozen-0806 pool."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable


FROZEN_0806_CONTRACT_ID = "frozen_0806_seeded_agent_first_player_v3"
FROZEN_0806_LEGACY_BALANCED_SEAT_CONTRACT_ID = "frozen_0806_seeded_2048_v2"
FROZEN_0806_FIRST_PLAYER_CONTRACT = (
    "seeded_coin_winner_then_winning_agent_selects_context_41"
)
FROZEN_0806_UNIT_GAMES = 256
FROZEN_0806_EVALUATION_UNITS = 8
FROZEN_0806_CUDA_GAMES = (
    FROZEN_0806_UNIT_GAMES * FROZEN_0806_EVALUATION_UNITS
)
FROZEN_0806_CPU_GAMES = FROZEN_0806_UNIT_GAMES
# Backward-compatible name for callers whose backend is necessarily CUDA.
FROZEN_0806_EVALUATION_GAMES = FROZEN_0806_CUDA_GAMES
FROZEN_0806_EVALUATION_SEED = 341_512_806


def evaluation_counts(schedule: Iterable[object]) -> tuple[int, ...]:
    """Expand one committed 256-game pool schedule into eight fixed units."""

    counts = tuple(getattr(entry, "games", None) for entry in schedule)
    if not counts or any(type(count) is not int or count < 1 for count in counts):
        raise ValueError("Frozen-0806 schedule must contain positive integer game counts")
    if sum(counts) != FROZEN_0806_UNIT_GAMES:
        raise ValueError("Frozen-0806 base schedule must contain exactly 256 games")
    expanded = tuple(FROZEN_0806_EVALUATION_UNITS * count for count in counts)
    if sum(expanded) != FROZEN_0806_CUDA_GAMES:
        raise AssertionError("Frozen-0806 evaluation schedule expansion is inconsistent")
    return expanded


def evaluation_schedule_id(
    base_schedule_sha256: str,
    *,
    evaluation_units: int = FROZEN_0806_EVALUATION_UNITS,
) -> str:
    """Commit the distribution, seed, units, and Agent-owned seat choice."""

    if (
        len(base_schedule_sha256) != 64
        or any(character not in "0123456789abcdef" for character in base_schedule_sha256)
    ):
        raise ValueError("base schedule SHA-256 must be 64 lowercase hexadecimal characters")
    if evaluation_units not in {1, FROZEN_0806_EVALUATION_UNITS}:
        raise ValueError("Frozen-0806 evaluation units must be CPU=1 or CUDA=8")
    payload = {
        "base_schedule_sha256": base_schedule_sha256,
        "contract_id": FROZEN_0806_CONTRACT_ID,
        "evaluation_seed": FROZEN_0806_EVALUATION_SEED,
        "evaluation_units": evaluation_units,
        "first_player_contract": FROZEN_0806_FIRST_PLAYER_CONTRACT,
        "games": FROZEN_0806_UNIT_GAMES * evaluation_units,
        "unit_games": FROZEN_0806_UNIT_GAMES,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def evaluation_game_seed(
    *,
    focal_identity: str,
    opponent_identity: str,
    slot: int,
    replica: int,
    namespace: str = "engine",
    evaluation_seed: int = FROZEN_0806_EVALUATION_SEED,
) -> int:
    """Derive one backend-independent seed for a committed slot replica."""

    if not focal_identity or not opponent_identity:
        raise ValueError("Frozen-0806 seed identities must be nonempty")
    if slot < 0 or not 0 <= replica < FROZEN_0806_EVALUATION_UNITS:
        raise ValueError("Frozen-0806 slot/replica is outside the contract")
    parts: tuple[object, ...] = (
        evaluation_seed,
        focal_identity,
        opponent_identity,
        slot,
        replica,
    )
    if namespace != "engine":
        parts = (evaluation_seed, namespace, focal_identity, opponent_identity, slot, replica)
    encoded = ":".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    return (value & 0x7FFFFFFF) or 1


def evaluation_coin_winner(
    *,
    focal_identity: str,
    opponent_identity: str,
    slot: int,
    replica: int,
    evaluation_seed: int = FROZEN_0806_EVALUATION_SEED,
) -> bool:
    """Return True when the focal player wins the fixed seeded opening toss."""

    value = evaluation_game_seed(
        evaluation_seed=evaluation_seed,
        focal_identity=focal_identity,
        opponent_identity=opponent_identity,
        slot=slot,
        replica=replica,
        namespace="coin-winner",
    )
    return bool(value & 1)


__all__ = [
    "FROZEN_0806_CONTRACT_ID",
    "FROZEN_0806_CPU_GAMES",
    "FROZEN_0806_CUDA_GAMES",
    "FROZEN_0806_EVALUATION_GAMES",
    "FROZEN_0806_EVALUATION_SEED",
    "FROZEN_0806_EVALUATION_UNITS",
    "FROZEN_0806_FIRST_PLAYER_CONTRACT",
    "FROZEN_0806_LEGACY_BALANCED_SEAT_CONTRACT_ID",
    "FROZEN_0806_UNIT_GAMES",
    "evaluation_counts",
    "evaluation_coin_winner",
    "evaluation_game_seed",
    "evaluation_schedule_id",
]
