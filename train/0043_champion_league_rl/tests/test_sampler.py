from __future__ import annotations

from collections import Counter
import importlib

import pytest


pfsp = importlib.import_module("train.0043_champion_league_rl.league.pfsp")
sampler = importlib.import_module("train.0043_champion_league_rl.league.sampler")


def _curriculum():
    state = pfsp.PFSPState()
    return state.curriculum_for(
        0, deck_ids=("001", "002", "003"),
        policy_ids=("Policy-0809", "Champion-G1"),
        config=pfsp.PFSPConfig(maximum_probability=0.8),
        training_deck_pool_hash="a" * 64, active_policy_pool_hash="b" * 64,
    )


def test_exact_quota_seed_reproducibility_and_seat_preservation() -> None:
    kwargs = dict(
        update=0, seed=430043001, deck_ids=("001", "002", "003"),
        policy_ids=("Policy-0809", "Champion-G1"),
        latest_champion_policy_id="Champion-G1", curriculum=_curriculum(),
    )
    first = sampler.build_schedule(**kwargs)
    second = sampler.build_schedule(**kwargs)
    assert first == second
    assert Counter(lane.branch for lane in first) == {
        "pfsp": 128, "uniform": 64, "latest_champion": 64,
    }
    assert len(first) == 256
    assert sum(lane.focal_goes_first for lane in first) == 128
    assert len({lane.engine_seed for lane in first}) == 256
    assert any(first[index].branch != first[index + 1].branch for index in range(255))
    assert all(
        lane.opponent_policy_id == "Champion-G1"
        for lane in first if lane.branch == "latest_champion"
    )


def test_deck_and_policy_are_independently_sampled() -> None:
    lanes = sampler.build_schedule(
        update=0, seed=9, deck_ids=("001", "002", "003"),
        policy_ids=("Policy-0809", "Champion-G1"),
        latest_champion_policy_id="Champion-G1", curriculum=_curriculum(),
    )
    pfsp_pairs = {
        (lane.opponent_deck_id, lane.opponent_policy_id)
        for lane in lanes if lane.branch == "pfsp"
    }
    assert len({deck for deck, _ in pfsp_pairs}) > 1
    assert len({policy for _, policy in pfsp_pairs}) > 1
    assert len(pfsp_pairs) > 3


def test_empty_pool_and_unregistered_latest_fail() -> None:
    with pytest.raises(ValueError, match="latest champion"):
        sampler.build_schedule(
            update=0, seed=1, deck_ids=("001",), policy_ids=("Policy-0809",),
            latest_champion_policy_id="Champion-G1", curriculum=pfsp.PFSPState().curriculum_for(
                0, deck_ids=("001",), policy_ids=("Policy-0809",),
                config=pfsp.PFSPConfig(maximum_probability=1.0),
                training_deck_pool_hash="a", active_policy_pool_hash="b",
            ),
        )
