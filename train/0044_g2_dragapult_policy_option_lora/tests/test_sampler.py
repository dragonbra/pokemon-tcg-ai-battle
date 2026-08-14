from __future__ import annotations

from collections import Counter
import importlib

import pytest


pfsp = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.league.pfsp")
sampler = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.league.sampler")


def _curriculum():
    state = pfsp.PFSPState()
    return state.curriculum_for(
        0, deck_ids=("001", "002", "003"),
        policy_ids=("Champion-G2",),
        config=pfsp.PFSPConfig(maximum_probability=1.0),
        training_deck_pool_hash="a" * 64, active_policy_pool_hash="b" * 64,
    )


def test_exact_quota_seed_reproducibility_and_agent_owned_seat_contract() -> None:
    kwargs = dict(
        update=0, seed=430044001, deck_ids=("001", "002", "003"),
        policy_ids=("Champion-G2",),
        latest_champion_policy_id="Champion-G2", curriculum=_curriculum(),
    )
    first = sampler.build_schedule(**kwargs)
    second = sampler.build_schedule(**kwargs)
    assert first == second
    assert Counter(lane.branch for lane in first) == {
        "pfsp": 128, "uniform": 64, "latest_champion": 64,
    }
    assert len(first) == 256
    assert len({lane.engine_seed for lane in first}) == 256
    assert len({lane.coin_winner_seed for lane in first}) == 256
    assert all(
        lane.focal_won_toss is bool(lane.coin_winner_seed & 1)
        for lane in first
    )
    assert all(not hasattr(lane, "focal_goes_first") for lane in first)
    assert any(first[index].branch != first[index + 1].branch for index in range(255))
    assert all(
        lane.opponent_policy_id == "Champion-G2" for lane in first
    )


def test_deck_pfsp_survives_singleton_policy_pool() -> None:
    lanes = sampler.build_schedule(
        update=0, seed=9, deck_ids=("001", "002", "003"),
        policy_ids=("Champion-G2",),
        latest_champion_policy_id="Champion-G2", curriculum=_curriculum(),
    )
    pfsp_pairs = {
        (lane.opponent_deck_id, lane.opponent_policy_id)
        for lane in lanes if lane.branch == "pfsp"
    }
    assert len({deck for deck, _ in pfsp_pairs}) > 1
    assert {policy for _, policy in pfsp_pairs} == {"Champion-G2"}
    assert len(pfsp_pairs) > 1


def test_empty_pool_and_unregistered_latest_fail() -> None:
    with pytest.raises(ValueError, match="singleton requested Champion"):
        sampler.build_schedule(
            update=0, seed=1, deck_ids=("001",), policy_ids=("Policy-0809",),
            latest_champion_policy_id="Champion-G1", curriculum=pfsp.PFSPState().curriculum_for(
                0, deck_ids=("001",), policy_ids=("Policy-0809",),
                config=pfsp.PFSPConfig(maximum_probability=1.0),
                training_deck_pool_hash="a", active_policy_pool_hash="b",
            ),
        )


def test_uniform_schedule_is_exactly_256_g2_lanes_without_pfsp() -> None:
    deck_ids = tuple(f"{index:03d}" for index in range(1, 68))
    curriculum = pfsp.PFSPState().curriculum_for(
        0, deck_ids=deck_ids, policy_ids=("Champion-G2",),
        config=pfsp.PFSPConfig(maximum_probability=1.0),
        training_deck_pool_hash="c" * 64, active_policy_pool_hash="d" * 64,
    )
    kwargs = dict(
        update=0, seed=430044001, deck_ids=deck_ids,
        policy_ids=("Champion-G2",), latest_champion_policy_id="Champion-G2",
        curriculum=curriculum,
    )
    first = sampler.build_uniform_schedule(**kwargs)
    assert first == sampler.build_uniform_schedule(**kwargs)
    assert len(first) == 256
    assert Counter(lane.branch for lane in first) == {"uniform": 256}
    assert {lane.opponent_policy_id for lane in first} == {"Champion-G2"}
    assert len({lane.engine_seed for lane in first}) == 256
    assert len({lane.coin_winner_seed for lane in first}) == 256
    assert all(
        lane.focal_won_toss is bool(lane.coin_winner_seed & 1)
        for lane in first
    )
    assert all(not hasattr(lane, "focal_goes_first") for lane in first)
