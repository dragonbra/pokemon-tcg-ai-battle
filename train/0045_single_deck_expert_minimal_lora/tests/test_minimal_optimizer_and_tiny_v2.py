from __future__ import annotations

import importlib
from pathlib import Path
import sys

import torch


PKG = "train.0045_single_deck_expert_minimal_lora"
PROJECT = Path(__file__).resolve().parents[1]


def _model():
    assets = importlib.import_module(f"{PKG}.assets")
    actor_critic = importlib.import_module(f"{PKG}.policy.actor_critic")
    row = next(item for item in assets.AssetRegistry.load(PROJECT).decks if item.deck_id == "007")
    deck = tuple(map(int, (PROJECT / row.deck_path).read_text().splitlines()))
    return actor_critic.load_actor_critic(deck=deck, deck_id="007")[0]


def test_optimizer_has_only_minimal_actor_and_critic_groups():
    ppo = importlib.import_module(f"{PKG}.training.ppo_full_semantic")
    trainer = ppo.PPOTrainer(
        _model(), device=torch.device("cpu"),
        config=ppo.PPOConfig(
            batch_size=4096, forward_microbatch_size=256,
            behavior_probe_batch_size=256, value_learning_rate=2e-5,
            prize_learning_rate=2e-5, offload_reference_after_cache=True,
        ),
    )
    manifest = trainer.optimizer_group_manifest()
    assert [row["name"] for row in manifest] == [
        "action_decoder", "value_win", "value_adapter", "allocation_head",
        "policy_option_lora", "value_prize",
    ]
    assert trainer.parameter_partition_manifest() == {
        "actor_only": {
            "parameters": 1_659_203,
            "learning_rates": {
                "action_decoder": 1e-5,
                "allocation_head": 1e-5,
                "policy_option_lora": 2e-5,
            },
        },
        "value_only": {"parameters": 3_712_467, "learning_rate": 2e-5},
        "shared_trainable": {"parameters": 0, "learning_rate": 1e-5},
    }


def test_tiny_v2_is_exact_deterministic_cuda512():
    schedule = importlib.import_module(f"{PKG}.evaluation.benchmark_tiny_v2_schedule")
    first = schedule.materialize(
        PROJECT, focal_deck_id="007", focal_deployment_identity="a" * 64,
        opponent_policy_id="Champion-G2",
    )
    second = schedule.materialize(
        PROJECT, focal_deck_id="007", focal_deployment_identity="b" * 64,
        opponent_policy_id="Champion-G2",
    )
    assert first["games"] == len(first["jobs"]) == 512
    assert set(first["class_counts"].values()) == {32}
    assert len({row["engine_seed"] for row in first["jobs"]}) == 512
    assert first["common_random_schedule_sha256"] == second["common_random_schedule_sha256"]
    assert first["schedule_sha256"] != second["schedule_sha256"]


def test_periodic_cadence_is_five_and_ten_is_not_duplicated():
    periodic = importlib.import_module(f"{PKG}.training.periodic_evaluation")
    assert [update for update in range(1, 21) if periodic.is_due(update)] == [5, 10, 15, 20]
    assert periodic.FORMAL_EVALUATE_INTERVAL_UPDATES == 10


def test_formal_cli_logs_the_frozen_u0_evaluation_baseline(monkeypatch, tmp_path):
    module = importlib.import_module(f"{PKG}.training.run_v1")
    captured = {}
    monkeypatch.setattr(module, "run", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_v1",
            "--launch-formal",
            "--u0-checkpoint",
            str(tmp_path / "update-000000.pt"),
        ],
    )
    assert module.main() == 0
    assert captured["baseline_evaluation_checkpoint"] == 0


def test_new_expert_default_and_limit_profiles_are_explicit():
    profiles = importlib.import_module(f"{PKG}.training.lr_profiles")
    assert profiles.STANDARD_LR_PROFILE is profiles.EXPERT_COLD_START_LR_PROFILE
    assert profiles.EXPERT_COLD_START_LR_PROFILE.rates() == {
        "decoder_learning_rate": 1e-5,
        "allocation_learning_rate": 1e-5,
        "option_lora_learning_rate": 2e-5,
        "value_learning_rate": 2e-5,
        "prize_learning_rate": 2e-5,
    }
    assert profiles.LIMIT_FINETUNE_LR_PROFILE.rates() == {
        "decoder_learning_rate": 5e-6,
        "allocation_learning_rate": 5e-6,
        "option_lora_learning_rate": 1e-5,
        "value_learning_rate": 2e-5,
        "prize_learning_rate": 2e-5,
    }
