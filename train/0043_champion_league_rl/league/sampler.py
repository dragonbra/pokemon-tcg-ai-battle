"""Exact 128/64/64 V2 opponent schedule construction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Mapping, Sequence

from .pfsp import Curriculum


BRANCH_COUNTS = {"pfsp": 128, "uniform": 64, "latest_champion": 64}


@dataclass(frozen=True, slots=True)
class LeagueLane:
    lane_id: int
    branch: str
    opponent_deck_id: str
    opponent_policy_id: str
    seat_slot: int
    focal_goes_first: bool
    engine_seed: int
    search_seed: int
    policy_seed: int
    curriculum_version: str


def _draw(rng: random.Random, values: Sequence[str], weights: Mapping[str, float] | None = None) -> str:
    if not values:
        raise ValueError("opponent pool cannot be empty")
    if weights is None:
        return values[rng.randrange(len(values))]
    probabilities = [weights[value] for value in values]
    return rng.choices(values, weights=probabilities, k=1)[0]


def _seed(master: int, logical_slot: int, namespace: str) -> int:
    value = f"0043:{master}:{logical_slot}:{namespace}".encode("ascii")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") & 0x7FFFFFFF


def build_schedule(
    *, update: int, seed: int, deck_ids: Sequence[str], policy_ids: Sequence[str],
    latest_champion_policy_id: str, curriculum: Curriculum,
) -> tuple[LeagueLane, ...]:
    deck_ids, policy_ids = tuple(deck_ids), tuple(policy_ids)
    if latest_champion_policy_id not in policy_ids:
        raise ValueError("latest champion must belong to the active policy pool")
    if set(curriculum.deck_weights) != set(deck_ids):
        raise ValueError("curriculum deck pool identity mismatch")
    if set(curriculum.policy_weights) != set(policy_ids):
        raise ValueError("curriculum policy pool identity mismatch")
    rng = random.Random(seed)
    pending: list[tuple[str, str, str, int]] = []
    logical_slot = 0
    for branch, count in BRANCH_COUNTS.items():
        for _ in range(count):
            if branch == "pfsp":
                deck = _draw(rng, deck_ids, curriculum.deck_weights)
                policy = _draw(rng, policy_ids, curriculum.policy_weights)
            elif branch == "uniform":
                deck, policy = _draw(rng, deck_ids), _draw(rng, policy_ids)
            else:
                deck, policy = _draw(rng, deck_ids), latest_champion_policy_id
            pending.append((branch, deck, policy, logical_slot))
            logical_slot += 1
    rng.shuffle(pending)
    lanes = tuple(
        LeagueLane(
            lane_id=lane_id, branch=branch, opponent_deck_id=deck,
            opponent_policy_id=policy, seat_slot=slot,
            focal_goes_first=slot % 2 == 0,
            engine_seed=_seed(seed, slot, "engine"),
            search_seed=_seed(seed, slot, "search"),
            policy_seed=_seed(seed, slot, "policy"),
            curriculum_version=curriculum.curriculum_version,
        )
        for lane_id, (branch, deck, policy, slot) in enumerate(pending)
    )
    if len(lanes) != 256 or sum(lane.focal_goes_first for lane in lanes) != 128:
        raise RuntimeError("0043 schedule violated 256-game/seat contract")
    return lanes


__all__ = ["BRANCH_COUNTS", "LeagueLane", "build_schedule"]
