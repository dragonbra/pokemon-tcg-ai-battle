from __future__ import annotations

import importlib


PKG = "train.0045_single_deck_expert_minimal_lora"


def test_policy0814_candidate_benchmark_v2_identity_contract():
    benchmark = importlib.import_module(
        f"{PKG}.evaluation.run_policy0814_candidate_benchmark_v2"
    )
    actor_critic = importlib.import_module(f"{PKG}.policy.actor_critic")
    schedule = importlib.import_module(f"{PKG}.evaluation.benchmark_v2_schedule")

    assert benchmark.FOCAL_BASE_POLICY_ID == "Policy-0814"
    assert benchmark.OPPONENT_POLICY_ID == "Policy-0809"
    assert benchmark.CONTRACT_ID == schedule.CONTRACT_ID
    assert benchmark.GAMES == 2048
    assert benchmark.SUPPORTED_LANE_COUNTS == (256, 512)
    assert benchmark.DEFAULT_LANE_COUNT == 512
    assert benchmark.DEFAULT_FOCAL_ACTOR_CHECKPOINT == (
        actor_critic.DEFAULT_0814_ACTOR_CHECKPOINT
    )
    assert benchmark.DEFAULT_FOCAL_VALUE_CHECKPOINT == (
        actor_critic.DEFAULT_0814_VALUE_CHECKPOINT
    )
    assert benchmark.candidate_policy_id("V14", "067", 5) == (
        "0045-policy0814-v14-deck067-update-000005"
    )
    assert benchmark.candidate_policy_id("V19", "067", 5) != (
        benchmark.candidate_policy_id("V14", "067", 5)
    )


def test_candidate_source_version_is_restricted_to_ascii_identity_component():
    benchmark = importlib.import_module(
        f"{PKG}.evaluation.run_policy0814_candidate_benchmark_v2"
    )
    for invalid in ("", "V19/U125", "../V19", "v 19", "V19_扩容"):
        try:
            benchmark.candidate_policy_id(invalid, "067", 125)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid source version was accepted: {invalid!r}")
