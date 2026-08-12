"""Resolve complete 0043 identities before constructing CUDA lane requests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

from ..assets import AssetRegistry
from ..league.sampler import LeagueLane
from ..policy_identity import materialize_policy_bundle


@dataclass(frozen=True, slots=True)
class CudaLaneRequest:
    lane_id: int
    branch: str
    opponent_deck_id: str
    opponent_exact_deck_sha256: str
    requested_policy_id: str
    materialized_policy_id: str
    opponent_effective_policy_sha256: str
    seat_slot: int
    focal_goes_first: bool
    engine_seed: int
    search_seed: int
    policy_seed: int
    curriculum_version: str


def materialize_lane_requests(
    project_root: Path, lanes: Sequence[LeagueLane],
) -> tuple[tuple[CudaLaneRequest, ...], str]:
    registry = AssetRegistry.load(project_root)
    registry.validate_all()
    decks = {deck.deck_id: deck for deck in registry.decks}
    admitted_policies = {
        policy.policy_id for policy in registry.policies
        if policy.frozen and policy.role in {"historical_anchor", "champion", "latest_champion"}
    }
    requested_policies = {lane.opponent_policy_id for lane in lanes}
    if not requested_policies <= admitted_policies:
        rejected = sorted(requested_policies - admitted_policies)
        raise ValueError(f"CUDA lanes use non-admitted opponent policies: {rejected}")
    bundles = {
        policy_id: materialize_policy_bundle(project_root, policy_id, purpose="cuda_engine_2_route")
        for policy_id in sorted(requested_policies)
    }
    requests = []
    for lane in lanes:
        deck = decks.get(lane.opponent_deck_id)
        if deck is None or not "training" in deck.roles:
            raise ValueError(f"CUDA lane uses an unregistered training deck: {lane.opponent_deck_id}")
        bundle = bundles[lane.opponent_policy_id]
        if bundle.policy_id != lane.opponent_policy_id or bundle.audit.requested_policy_id != lane.opponent_policy_id:
            raise RuntimeError("requested/materialized CUDA opponent identity mismatch")
        requests.append(CudaLaneRequest(
            lane_id=lane.lane_id, branch=lane.branch,
            opponent_deck_id=deck.deck_id,
            opponent_exact_deck_sha256=deck.content_sha256,
            requested_policy_id=lane.opponent_policy_id,
            materialized_policy_id=bundle.policy_id,
            opponent_effective_policy_sha256=bundle.audit.effective_policy_sha256,
            seat_slot=lane.seat_slot, focal_goes_first=lane.focal_goes_first,
            engine_seed=lane.engine_seed, search_seed=lane.search_seed,
            policy_seed=lane.policy_seed, curriculum_version=lane.curriculum_version,
        ))
    encoded = json.dumps([asdict(row) for row in requests], sort_keys=True, separators=(",", ":")).encode()
    return tuple(requests), hashlib.sha256(encoded).hexdigest()


__all__ = ["CudaLaneRequest", "materialize_lane_requests"]
