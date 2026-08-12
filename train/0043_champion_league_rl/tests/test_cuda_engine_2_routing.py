from __future__ import annotations

import importlib
from collections import Counter
from pathlib import Path


assets_module = importlib.import_module("train.0043_champion_league_rl.assets")
pfsp = importlib.import_module("train.0043_champion_league_rl.league.pfsp")
sampler = importlib.import_module("train.0043_champion_league_rl.league.sampler")
routing = importlib.import_module("train.0043_champion_league_rl.cuda_engine_2.routing")
ROOT = Path(__file__).resolve().parents[1]


def test_all_256_lanes_resolve_numeric_decks_and_complete_policies() -> None:
    assets = assets_module.AssetRegistry.load(ROOT)
    decks = [deck.deck_id for deck in assets.decks]
    policies = [policy.policy_id for policy in assets.policies if policy.frozen]
    curriculum = pfsp.PFSPState().curriculum_for(
        0, deck_ids=decks, policy_ids=policies, config=pfsp.PFSPConfig(),
        training_deck_pool_hash="d" * 64, active_policy_pool_hash="p" * 64,
    )
    lanes = sampler.build_schedule(
        update=0, seed=430043001, deck_ids=decks, policy_ids=policies,
        latest_champion_policy_id=assets.latest_champion_policy_id,
        curriculum=curriculum,
    )
    requests, schedule_hash = routing.materialize_lane_requests(ROOT, lanes)
    assert len(requests) == 256
    assert len(schedule_hash) == 64
    assert Counter(row.branch for row in requests) == sampler.BRANCH_COUNTS
    assert all(len(row.opponent_deck_id) == 3 and row.opponent_deck_id.isdigit() for row in requests)
    assert all(row.requested_policy_id == row.materialized_policy_id for row in requests)
    assert all(len(row.opponent_effective_policy_sha256) == 64 for row in requests)
