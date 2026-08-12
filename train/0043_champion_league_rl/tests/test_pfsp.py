from __future__ import annotations

import importlib

import pytest


module = importlib.import_module("train.0043_champion_league_rl.league.pfsp")
PFSPConfig, PFSPState, Record = module.PFSPConfig, module.PFSPState, module.Record


def test_beta_smoothing_and_record_updates() -> None:
    record = Record()
    assert record.smoothed_win_rate == 0.5
    assert record.weakness == 0.5
    record.observe("loss", 3)
    assert record.games == 1 and record.losses == 1
    assert record.smoothed_win_rate == pytest.approx(1 / 3)


def test_curriculum_is_frozen_for_ten_updates_and_refreshes_at_boundary() -> None:
    state = PFSPState()
    config = PFSPConfig(maximum_probability=0.75)
    kwargs = dict(
        deck_ids=("001", "002"), policy_ids=("Policy-0809", "Champion-G1"),
        config=config, training_deck_pool_hash="d" * 64,
        active_policy_pool_hash="p" * 64,
    )
    c0 = state.curriculum_for(0, **kwargs)
    state.observe(deck_id="001", policy_id="Policy-0809", result="loss", update=0)
    assert state.curriculum_for(9, **kwargs) is c0
    c1 = state.curriculum_for(10, **kwargs)
    assert c0.curriculum_version == "C000"
    assert c1.curriculum_version == "C001"
    assert c1.deck_weights["001"] > c1.deck_weights["002"]


def test_state_round_trip_preserves_curriculum(tmp_path) -> None:
    state = PFSPState()
    state.observe(deck_id="001", policy_id="Policy-0809", result="win", update=0)
    kwargs = dict(
        deck_ids=("001",), policy_ids=("Policy-0809",),
        config=PFSPConfig(maximum_probability=1.0), training_deck_pool_hash="a" * 64,
        active_policy_pool_hash="b" * 64,
    )
    expected = state.curriculum_for(0, **kwargs)
    path = tmp_path / "pfsp.json"
    state.save(path)
    actual = PFSPState.load(path).curriculum_for(9, **kwargs)
    assert actual == expected
    assert actual.content_hash == expected.content_hash


def test_infeasible_floor_or_cap_fails() -> None:
    state = PFSPState()
    with pytest.raises(ValueError, match="infeasible"):
        state.curriculum_for(
            0, deck_ids=("001", "002"), policy_ids=("P",),
            config=PFSPConfig(minimum_probability=0.6, maximum_probability=1.0),
            training_deck_pool_hash="a", active_policy_pool_hash="b",
        )


def test_default_config_is_feasible_for_initial_two_policy_pool() -> None:
    curriculum = PFSPState().curriculum_for(
        0, deck_ids=("001",), policy_ids=("Policy-0809", "Champion-G1"),
        config=PFSPConfig(), training_deck_pool_hash="a",
        active_policy_pool_hash="b",
    )
    assert curriculum.policy_weights == {"Policy-0809": 0.5, "Champion-G1": 0.5}
