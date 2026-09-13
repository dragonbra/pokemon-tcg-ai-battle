from __future__ import annotations

import importlib
import json
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


def test_tiny_v2_renderer_uses_report_deck_and_update_identity(tmp_path):
    renderer = importlib.import_module(f"{PKG}.evaluation.render_benchmark_tiny_v2")
    report = {
        "status": "PASS",
        "benchmark_id": "Benchmark-Tiny-V2",
        "focal_deck_id": "003",
        "focal_deck_display_name": "Mega Lopunny ex / Mega Froslass ex",
        "focal_checkpoint_update": 385,
        "summary": {
            "win_rate": 0.5, "wins": 1, "losses": 1, "draws": 0,
            "focal_first_win_rate": 1.0, "focal_first_games": 1,
            "focal_second_win_rate": 0.0, "focal_second_games": 1,
        },
        "focal_policy_identity_audit": {
            "contract_id": "kaggle_fp16_storage_fp32_runtime_v1",
            "source_checkpoint_sha256": "a" * 64,
            "portable_checkpoint_sha256": "b" * 64,
            "effective_candidate_sha256": "c" * 64,
        },
        "entries": [
            {"opponent_meta_archetype_id": 0, "opponent_id": "001", "outcome": 1},
            {"opponent_meta_archetype_id": 1, "opponent_id": "002", "outcome": -1},
        ],
    }
    source = tmp_path / "report.json"
    output = tmp_path / "report.html"
    source.write_text(json.dumps(report), encoding="utf-8")
    renderer.render(source, output)
    html = output.read_text(encoding="utf-8")
    assert "Deck 003 · U385 · Benchmark Tiny V2" in html
    assert "Mega Lopunny ex / Mega Froslass ex" in html
    assert "001 · Marnie&#x27;s Grimmsnarl ex / Froslass" in html
    assert "U0 baseline" not in html


def test_three_pool_tiny_v2_is_disjoint_balanced_policy0809_cuda512():
    schedule = importlib.import_module(
        f"{PKG}.evaluation.benchmark_tiny_v2_three_pool_schedule"
    )
    first = schedule.materialize(
        PROJECT, focal_deck_id="007", focal_deployment_identity="a" * 64,
    )
    second = schedule.materialize(
        PROJECT, focal_deck_id="007", focal_deployment_identity="b" * 64,
    )
    assert first["opponent_policy_id"] == "Policy-0809"
    assert first["games_per_pool"] == 512
    assert first["games"] == 1536
    assert set(first["pools"]) == {"low_score", "priority", "remaining"}
    expected = {
        "low_score": {2, 3, 5},
        "priority": {0, 1, 4, 27},
    }
    all_pool_ids = []
    for pool_name, pool in first["pools"].items():
        assert len(pool["jobs"]) == 512
        ids = set(pool["selected_class_ids"])
        all_pool_ids.extend(ids)
        if pool_name in expected:
            assert ids == expected[pool_name]
        counts = list(pool["class_counts"].values())
        assert max(counts) - min(counts) <= 1
        for seed_name in (
            "engine_seed", "search_seed", "policy_seed", "coin_winner_seed",
        ):
            assert len({row[seed_name] for row in pool["jobs"]}) == 512
        assert (
            pool["common_random_schedule_sha256"]
            == second["pools"][pool_name]["common_random_schedule_sha256"]
        )
    assert len(all_pool_ids) == len(set(all_pool_ids)) == 28
    assert 14 not in all_pool_ids  # Empty taxonomy class has no exact deck.
    assert first["schedule_sha256"] != second["schedule_sha256"]


def test_benchmark_v2_candidate_identity_is_checkpoint_specific():
    benchmark = importlib.import_module(f"{PKG}.evaluation.run_benchmark_v2")
    assert benchmark.candidate_policy_id("007", 190) == (
        "0045-single-deck-expert-007-update-000190"
    )
    assert benchmark.candidate_policy_role(190) == "specialist_checkpoint_candidate"
    assert benchmark.candidate_policy_role(0) == "frozen_specialist_initialization"


def test_periodic_cadence_is_five_and_ten_is_not_duplicated():
    periodic = importlib.import_module(f"{PKG}.training.periodic_evaluation")
    assert [update for update in range(1, 21) if periodic.is_due(update)] == [5, 10, 15, 20]
    assert periodic.FORMAL_EVALUATE_INTERVAL_UPDATES == 10


def test_three_pool_metrics_keep_each_cuda512_curve_separate():
    periodic = importlib.import_module(f"{PKG}.training.periodic_evaluation")
    summaries = {
        "low_score": {"games": 512, "wins": 256, "losses": 255, "draws": 1,
                      "win_rate": 0.5, "focal_first_win_rate": 0.51,
                      "focal_second_win_rate": 0.49, "games_per_second": 8.0},
        "priority": {"games": 512, "wins": 300, "losses": 212, "draws": 0,
                     "win_rate": 300 / 512, "focal_first_win_rate": 0.6,
                     "focal_second_win_rate": 0.57, "games_per_second": 8.0},
        "remaining": {"games": 512, "wins": 350, "losses": 162, "draws": 0,
                      "win_rate": 350 / 512, "focal_first_win_rate": 0.7,
                      "focal_second_win_rate": 0.67, "games_per_second": 8.0},
    }
    report = {
        "status": "PASS",
        "benchmark_id": "Benchmark-Tiny-V2-Policy0809-Three-Pool",
        "focal_checkpoint_update": 205,
        "pool_summaries": summaries,
        "summary": {"games": 1536, "wins": 906, "losses": 629, "draws": 1,
                    "win_rate": 906 / 1536, "games_per_second": 8.0},
    }
    metrics = periodic.wandb_metrics_three_pool(report)
    assert metrics["eval/checkpoint_update"] == 205
    assert metrics["eval/low_score/games"] == 512
    assert metrics["eval/low_score/win_rate"] == 0.5
    assert metrics["eval/priority/win_rate"] == 300 / 512
    assert metrics["eval/remaining/win_rate"] == 350 / 512
    assert metrics["eval/three_pool_aggregate/games"] == 1536
    assert "eval/core/win_rate" not in metrics


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
        "shared_encoder_learning_rate": 2e-5,
        "value_learning_rate": 2e-5,
        "prize_learning_rate": 2e-5,
    }
    assert profiles.LIMIT_FINETUNE_LR_PROFILE.rates() == {
        "decoder_learning_rate": 5e-6,
        "allocation_learning_rate": 5e-6,
        "option_lora_learning_rate": 1e-5,
        "shared_encoder_learning_rate": 1e-5,
        "value_learning_rate": 2e-5,
        "prize_learning_rate": 2e-5,
    }


def test_v4_u200_targeted_meta_contract_is_exact():
    module = importlib.import_module(
        f"{PKG}.training.run_v4_u200_policy0809_three_pool"
    )
    assert module.START_UPDATE == 200
    assert module.ROLLOUT_OPPONENT_POLICY_IDS == ("Champion-G2",)
    assert module.PERIODIC_EVALUATION_PROFILE == "policy0809_three_pool_cuda512"
    assert module.OPPONENT_META_WEIGHTS == {
        0: 3.0, 1: 3.0, 2: 3.0, 3: 6.0,
        4: 3.0, 5: 6.0, 27: 1.0,
    }
