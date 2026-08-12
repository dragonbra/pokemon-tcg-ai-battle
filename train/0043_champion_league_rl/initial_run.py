"""Fail-closed first-run focal and opponent scheduling contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from .assets import AssetRegistry
from .league.pfsp import PFSPConfig, PFSPState
from .league.sampler import build_schedule
from .cuda_engine_2.routing import materialize_lane_requests


PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
CONTRACT_PATH = REPOSITORY_ROOT / "experiments/0043_champion_league_rl/initial_run_contract.json"


@dataclass(frozen=True, slots=True)
class FocalLane:
    lane_id: int
    deck_id: str


def focal_schedule(seed: int) -> tuple[FocalLane, ...]:
    values = ["002"] * 128 + ["007"] * 128
    random.Random(seed).shuffle(values)
    return tuple(FocalLane(index, deck_id) for index, deck_id in enumerate(values))


def acceptance(seed: int = 430043001, *, schedules: int = 64) -> dict[str, Any]:
    registry = AssetRegistry.load(PROJECT_ROOT)
    registry.validate_all()
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    decks = tuple(deck.deck_id for deck in registry.decks if "training" in deck.roles)
    policies = registry.active_policy_ids
    if decks != tuple(f"{index:03d}" for index in range(1, 68)):
        raise RuntimeError("initial opponent deck pool is not exact 001-067")
    if policies != ("Policy-0809", "Champion-G1"):
        raise RuntimeError("initial opponent policy pool identity changed")
    seen_pairs: set[tuple[str, str]] = set()
    schedule_hashes = []
    first_requests = None
    for offset in range(schedules):
        curriculum = PFSPState().curriculum_for(
            0, deck_ids=decks, policy_ids=policies, config=PFSPConfig(),
            training_deck_pool_hash="d" * 64, active_policy_pool_hash="p" * 64,
        )
        lanes = build_schedule(
            update=offset, seed=seed + offset, deck_ids=decks, policy_ids=policies,
            latest_champion_policy_id=registry.latest_champion_policy_id,
            curriculum=curriculum,
        )
        requests, digest = materialize_lane_requests(PROJECT_ROOT, lanes)
        schedule_hashes.append(digest)
        first_requests = requests if first_requests is None else first_requests
        seen_pairs.update((row.opponent_deck_id, row.requested_policy_id) for row in requests)
    expected_pairs = {(deck, policy) for deck in decks for policy in policies}
    if seen_pairs != expected_pairs:
        raise RuntimeError(f"PFSP acceptance missed deck-policy pairs: {sorted(expected_pairs-seen_pairs)}")
    focal = focal_schedule(seed)
    if {row.deck_id for row in focal} != {"002", "007"}:
        raise RuntimeError("focal deck schedule escaped 002/007")
    policy_hashes = {
        row.requested_policy_id: row.opponent_effective_policy_sha256
        for row in first_requests
    }
    return {
        "schema_version": "0043_initial_run_acceptance_v1",
        "status": "PASS",
        "formal_launch_authorized": False,
        "focal": {"deck_counts": {"002": 128, "007": 128}, "parent": "Champion-G1"},
        "opponents": {
            "deck_count": len(decks), "policy_ids": list(policies),
            "deck_policy_pairs_covered": len(seen_pairs),
            "expected_deck_policy_pairs": len(expected_pairs),
            "effective_policy_sha256": policy_hashes,
        },
        "sampling": {"schedules": schedules, "lanes": schedules * 256,
                     "unique_schedule_hashes": len(set(schedule_hashes)),
                     "first_schedule_sha256": schedule_hashes[0]},
        "contract_sha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schedules", type=int, default=64)
    args = parser.parse_args()
    payload = acceptance(schedules=args.schedules)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
