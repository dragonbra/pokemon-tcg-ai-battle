from __future__ import annotations

from collections import Counter
import importlib
from pathlib import Path

import pytest


PKG = "pokemon_tcg_ai"
PROJECT = Path(__file__).resolve().parents[1]


def _mappings():
    own = importlib.import_module(f"{PKG}.own_archetype")
    return own.OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT
    ).mappings


def test_v13_exact_deck_rollout_schedule_and_policy_identity():
    assets = importlib.import_module(f"{PKG}.assets")
    runner = importlib.import_module(f"{PKG}.training.run_v1")
    v13 = importlib.import_module(f"{PKG}.training.run_v13_policy0814_shared_encoder")
    registry = assets.AssetRegistry.load(PROJECT)
    jobs, _, policy_weights, curriculum = runner._jobs(
        3, registry, focal_deck_ids=("007",), focal_deck_id="007",
        opponent_sampling_mode="exact_deck_quota_training_pool",
        opponent_policy_ids=("Policy-0814",),
        latest_champion_policy_id="Policy-0814", rollout_games=512,
        opponent_deck_quotas=v13.FIXED_DECK_QUOTAS,
        opponent_random_deck_ids=v13.RANDOM_DECK_IDS,
    )
    counts = Counter(job.opponent_id for job in jobs)
    assert len(jobs) == 512
    assert set(counts) == set(v13.FIXED_DECK_QUOTAS)
    assert all(counts[key] >= value for key, value in v13.FIXED_DECK_QUOTAS.items())
    assert sum(counts[key] - v13.FIXED_DECK_QUOTAS[key] for key in counts) == 122
    assert {key for key in counts if counts[key] > v13.FIXED_DECK_QUOTAS[key]} <= set(
        v13.RANDOM_DECK_IDS
    )
    assert {job.opponent_policy_id for job in jobs} == {"Policy-0814"}
    assert policy_weights == {"Policy-0814": 1.0}
    assert curriculum == "exact_deck_quota-u000003"


def test_v13_frozen_eval_schedule_is_reproducible_and_matches_frequency_model():
    schedule = importlib.import_module(f"{PKG}.evaluation.policy0814_exact_deck_schedule")
    first = schedule.materialize(PROJECT, focal_deck_id="007", focal_deployment_identity="a" * 64)
    second = schedule.materialize(PROJECT, focal_deck_id="007", focal_deployment_identity="b" * 64)
    assert first["jobs"] == second["jobs"]
    assert first["games"] == 512
    assert first["opponent_policy_id"] == "Policy-0814"
    counts = first["realized_deck_counts"]
    assert set(counts) == set(schedule.FIXED_DECK_QUOTAS)
    assert all(counts[key] >= value for key, value in schedule.FIXED_DECK_QUOTAS.items())
    assert sum(counts[key] - schedule.FIXED_DECK_QUOTAS[key] for key in counts) == 122


def test_v13_eval_portable_candidate_requires_all_bound_identities(tmp_path, monkeypatch):
    runner = importlib.import_module(
        f"{PKG}.evaluation.run_policy0814_exact_deck_cuda512"
    )
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="all expected hashes"):
        runner.run(
            deck_id="003",
            output_root=tmp_path / "report",
            checkpoint_update=385,
            portable_candidate=tmp_path / "model.bin",
        )


def test_v13_trainable_tensor_boundary_is_exact():
    v13 = importlib.import_module(f"{PKG}.training.run_v13_policy0814_shared_encoder")
    model, _, audit = v13._model_and_audit()
    assert audit["tensor_count"] == 115
    assert audit["total_trainable_params"] == 5_515_030
    assert audit["module_totals"] == v13.EXPECTED_MODULE_TOTALS
    names = {row["name"] for row in audit["tensors"]}
    assert not any("prototype_encoder" in name for name in names)
    assert not any("board_encoder.layers.0." in name for name in names)
    assert not any("board_encoder.layers.1." in name for name in names)
    assert not any("board_encoder.layers.2." in name for name in names)
    assert all(
        row["lora_rank"] == 16
        for row in audit["tensors"]
        if row["module"].startswith(("StateEncoder", "OptionEncoder"))
    )
    model.assert_trainable_contract()


def test_v13_optimizer_owns_shared_encoder_once_at_option_lora_lr():
    import torch

    v13 = importlib.import_module(f"{PKG}.training.run_v13_policy0814_shared_encoder")
    ppo = importlib.import_module(f"{PKG}.training.ppo_full_semantic")
    model, _, _ = v13._model_and_audit()
    trainer = ppo.PPOTrainer(
        model, device=torch.device("cpu"),
        config=ppo.PPOConfig(
            batch_size=4096, forward_microbatch_size=256,
            behavior_probe_batch_size=256, offload_reference_after_cache=True,
        ),
    )
    groups = {row["name"]: row for row in trainer.optimizer_group_manifest()}
    assert groups["shared_encoder_lora"]["trainable_parameters"] == 92_160
    assert groups["shared_encoder_lora"]["tensor_count"] == 16
    assert groups["shared_encoder_lora"]["base_learning_rate"] == 2e-5
    assert sum(row["trainable_parameters"] for row in groups.values()) == 5_515_030


def test_policy0814_materializes_as_complete_independent_identity():
    identity = importlib.import_module(f"{PKG}.policy_identity")
    bundle = identity.materialize_policy_bundle(
        PROJECT, "Policy-0814", purpose="0045_v13_test"
    )
    assert bundle.audit.status == "PASS"
    assert bundle.audit.requested_policy_id == "Policy-0814"
    assert bundle.audit.effective_policy_sha256 == (
        "476d57d55eb9c040fa4e75ce74ac5205af5d094a296db71cba4ecbf792902580"
    )


def test_ppo_identity_gate_admits_registered_policy0814():
    batch = importlib.import_module(f"{PKG}.training.batch_full_semantic")
    assert batch.is_admitted_frozen_opponent_policy_id("Policy-0809")
    assert batch.is_admitted_frozen_opponent_policy_id("Policy-0814")
    assert batch.is_admitted_frozen_opponent_policy_id("Champion-G2")
    assert not batch.is_admitted_frozen_opponent_policy_id("Policy-unknown")
