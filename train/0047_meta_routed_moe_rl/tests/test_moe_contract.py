from __future__ import annotations

import importlib

import pytest
import torch


assets = importlib.import_module("train.0047_meta_routed_moe_rl.assets")
model_module = importlib.import_module(
    "train.0047_meta_routed_moe_rl.policy.moe_actor_critic"
)
distribution = importlib.import_module(
    "train.0047_meta_routed_moe_rl.policy.moe_distribution"
)
meta_schedule = importlib.import_module(
    "train.0047_meta_routed_moe_rl.league.meta_balanced"
)
own_archetype = importlib.import_module(
    "train.0047_meta_routed_moe_rl.own_archetype"
)
eval_schedule = importlib.import_module(
    "train.0047_meta_routed_moe_rl.evaluation.moe_three_pool_schedule"
)
moe_evaluation = importlib.import_module(
    "train.0047_meta_routed_moe_rl.evaluation.moe_three_pool"
)
moe_run = importlib.import_module(
    "train.0047_meta_routed_moe_rl.training.run_v1_moe"
)
moe_ppo = importlib.import_module(
    "train.0047_meta_routed_moe_rl.training.ppo_moe"
)
moe_checkpoint = importlib.import_module(
    "train.0047_meta_routed_moe_rl.training.moe_checkpoint"
)
runtime = importlib.import_module("train.0047_meta_routed_moe_rl.runtime")
run_v1 = importlib.import_module("train.0047_meta_routed_moe_rl.training.run_v1")


def _model():
    registry = assets.AssetRegistry.load(run_v1.PROJECT_ROOT)
    return model_module.load_moe_actor_critic(
        deck=run_v1._cards(registry, "070"), device="cpu"
    )[0]


def test_u0_parity_and_expert_storage_independence():
    model = _model()
    batch = runtime.synthetic_batch(batch_size=2)
    base = model.actor.deterministic_action_tensors(batch)
    validated, state, prefix, value_options = model.encode_shared(dict(batch.items()))
    decoded, _options, _posterior = distribution.decode_moe_device(
        model, validated, state, prefix, value_options, torch.tensor((-1, 2)),
        max_select=64, greedy=True, compute_stats=True,
        sampling_seeds=None, sampling_counters=None,
    )
    assert torch.equal(base.lengths, decoded["lengths"])
    assert torch.equal(
        base.sequences, decoded["actions"][:, : base.sequences.shape[1]]
    )
    assert len(model.experts) == 7
    for index in range(1, model.expert_count):
        for (left_name, left), (right_name, right) in zip(
            model.experts[0].named_parameters(),
            model.experts[index].named_parameters(), strict=True,
        ):
            assert left_name == right_name
            assert torch.equal(left, right)
            assert left.untyped_storage().data_ptr() != right.untyped_storage().data_ptr()


def test_hard_routes_and_soft_router_gradient():
    model = _model()
    assert model.router_logits.shape == (29, 7)
    routing_ids = torch.tensor((-1, 0, 1, 2, 3, 5, 27, 4))
    expected = torch.tensor((0, 1, 2, 3, 4, 5, 6, 0))
    actual = model.gate_probabilities(routing_ids).argmax(1)
    assert torch.equal(actual, expected)
    batch = runtime.synthetic_batch(batch_size=6)
    action = model.actor.deterministic_action_tensors(batch)
    meta = torch.tensor((0, 1, 2, 3, 5, 27))
    with torch.no_grad():
        for expert in model.experts[1:]:
            next(expert.action_decoder.parameters()).add_(1e-3)
    model.set_soft_routing(True)
    gate = model.gate_probabilities(meta)
    assert torch.allclose(gate.sum(dim=1), torch.ones(6), atol=1e-6)
    assert torch.allclose(gate[:, 0], torch.full((6,), 0.1995), atol=1e-6)
    assert torch.allclose(gate.max(dim=1).values, torch.full((6,), 0.8), atol=1e-6)
    assert torch.equal(gate.gt(0).sum(dim=1), torch.full((6,), 7))
    fallback = model.gate_probabilities(torch.tensor((-1, 4, 28)))
    assert torch.equal(fallback[0], torch.tensor((1.0, 0, 0, 0, 0, 0, 0)))
    assert torch.allclose(fallback[1:].sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.allclose(fallback[1:, 0], torch.full((2,), 0.9994), atol=1e-6)
    evaluated = distribution.evaluate_moe_actions(
        model, dict(batch.items()), action.sequences, action.lengths,
        torch.zeros(6, dtype=torch.bool), meta, (None,) * 6,
    )
    (-evaluated.joint_log_prob.mean()).backward()
    assert torch.isfinite(evaluated.joint_log_prob).all()
    assert model.router_logits.grad is not None
    assert torch.isfinite(model.router_logits.grad).all()
    assert float(model.router_logits.grad.norm()) > 0
    assert evaluated.auxiliary is not None
    # This is the inherited Policy-0814 critic auxiliary head. Public Meta29
    # routing is the separate audited lookup table above, not this old head.
    assert evaluated.auxiliary["meta_logits"].shape == (6, 15)
    assert evaluated.auxiliary["v_prize"].shape == (6,)


def test_sparse_execution_calls_only_gate_support():
    model = _model()
    batch = runtime.synthetic_batch(batch_size=3)
    action = model.actor.deterministic_action_tensors(batch)
    calls = [0] * model.expert_count
    hooks = [
        expert.policy_option_lora.register_forward_hook(
            lambda _module, _inputs, _output, index=index: calls.__setitem__(
                index, calls[index] + 1
            )
        )
        for index, expert in enumerate(model.experts)
    ]
    distribution.evaluate_moe_actions(
        model, dict(batch.items()), action.sequences, action.lengths,
        torch.zeros(3, dtype=torch.bool), torch.tensor((-1, 4, 28)),
        (None,) * 3,
    )
    assert calls == [1, 0, 0, 0, 0, 0, 0]
    calls[:] = [0] * model.expert_count
    model.set_soft_routing(True)
    distribution.evaluate_moe_actions(
        model, dict(batch.items()), action.sequences, action.lengths,
        torch.zeros(3, dtype=torch.bool), torch.tensor((5, 5, 5)),
        (None,) * 3,
    )
    assert calls == [1, 1, 1, 1, 1, 1, 1]
    for hook in hooks:
        hook.remove()


def test_core_deck_half_schedule_is_exact_and_reproducible():
    vocabulary = own_archetype.OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=run_v1.PROJECT_ROOT
    )
    core = ("007", "003", "001", "002", "009", "011", "023")
    first, audit = meta_schedule.core_deck_half_meta_balanced_schedule(
        quota_seed=440_120_000, shuffle_seed=440_120_200,
        mappings=vocabulary.mappings, core_deck_ids=core, lanes=512,
    )
    second, _ = meta_schedule.core_deck_half_meta_balanced_schedule(
        quota_seed=440_120_000, shuffle_seed=440_120_200,
        mappings=vocabulary.mappings, core_deck_ids=core, lanes=512,
    )
    assert first == second and len(first) == 512
    assert audit["core_lanes"] == 256
    assert audit["general_meta_balanced_lanes"] == 256
    assert sum(audit["core_deck_counts"].values()) == 256
    assert set(audit["core_deck_counts"].values()) == {36, 37}
    assert sum(row.branch == "core_deck_half" for row in first) == 256
    assert sum(row.branch == "meta_balanced_half" for row in first) == 256


def test_eval_two_physical_pools_have_exact_focus_subgroup_quotas():
    schedule = eval_schedule.materialize(
        run_v1.PROJECT_ROOT, focal_deck_id="070",
        focal_deployment_identity="0" * 64,
    )
    assert tuple(schedule["pools"]) == ("focus_seven", "remain_meta")
    expected = {
        "old_three": {"001", "002", "011"},
        "new_four": {"007", "003", "009", "023"},
    }
    focus = schedule["pools"]["focus_seven"]
    assert focus["games"] == 512
    assert focus["focus_group_counts"] == {"old_three": 256, "new_four": 256}
    for group, deck_ids in expected.items():
        rows = [row for row in focus["jobs"] if row["focus_group"] == group]
        assert {row["opponent_deck_id"] for row in rows} == deck_ids
        assert len(rows) == 256
        counts = {
            deck_id: sum(row["opponent_deck_id"] == deck_id for row in rows)
            for deck_id in deck_ids
        }
        assert max(counts.values()) - min(counts.values()) <= 1
    remaining = {
        row["opponent_deck_id"]
        for row in schedule["pools"]["remain_meta"]["jobs"]
    }
    assert not remaining & set().union(*expected.values())
    assert schedule["pools"]["remain_meta"]["games"] == 512
    assert schedule["games"] == 1024


def test_v16_uses_win_only_actor_advantage_and_parent_u5_then_every_five_eval():
    config = moe_ppo.PPOConfig()
    assert config.prize_aux_actor_weight == 0.0
    assert moe_run.PRIZE_AUX_ACTOR_WEIGHT == 0.0
    assert moe_run.EVAL_AT_U0 is False
    assert moe_run.EVAL_INTERVAL == 5
    assert moe_run.PARENT_UPDATE == 5


def test_formal_training_keeps_optimized_cuda512_and_microbatch256_contract():
    assert moe_run.CUDA_LANES == 512
    assert moe_run.ROLLOUT_CHUNK_GAMES == 512
    assert moe_run.PPO_FORWARD_MICROBATCH == 256
    assert moe_ppo.PPOConfig().offload_reference_after_cache is True
    assert moe_run.PARENT_UPDATE == 5
    assert moe_run.PARENT_CHECKPOINT_SHA256 == (
        "e033c2cbc63f9f33e1eae582d8407493688694bcda6ea3c40097492cff930d78"
    )


def test_candidate_metadata_names_the_meta29_identifier_and_29x7_router():
    assert moe_evaluation.ROUTER_IDENTIFIER == "0047_public_meta29_priority_rules_v1"
    assert moe_evaluation.ROUTER_TOPOLOGY == "public_meta29_lookup_softmax_29x7"


def test_router_heatmap_generates_png(tmp_path):
    output = moe_run._router_heatmap(_model(), 0, tmp_path / "router.png")
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_router_audit_json_serializes_the_complete_29x7_table(tmp_path):
    model = _model()
    path = tmp_path / "router.json"
    model_module.save_router_table(model, path)
    payload = __import__("json").loads(path.read_text())
    assert payload["schema_version"] == "0047_meta29x7_lookup_softmax_router_v1"
    assert len(payload["logits"]) == 29
    assert all(len(row) == 7 for row in payload["logits"])
    assert len(payload["rows"]) == 29
    assert "alphas" not in payload


def test_compact_checkpoint_round_trip_is_exact_and_omits_frozen_actor():
    model = _model()
    with torch.no_grad():
        next(model.experts[3].action_decoder.parameters()).add_(0.125)
        next(model.value_adapter.parameters()).sub_(0.25)
    model.set_soft_routing(True)
    payload = moe_checkpoint.build_compact_checkpoint(
        model, 5, metadata={"version": "test", "focal_deck_id": "007"}
    )
    assert payload["schema_version"] == moe_checkpoint.SCHEMA_VERSION
    assert payload["update"] == 5
    assert payload["delta_state_dict"]
    assert not any(name.startswith("actor.") for name in payload["delta_state_dict"])
    assert all(
        not value.is_floating_point() or value.dtype == torch.float32
        for value in payload["delta_state_dict"].values()
    )
    restored = _model()
    moe_checkpoint.load_compact_checkpoint(restored, payload)
    assert restored.soft_routing is True
    assert restored.router_logits.requires_grad is True
    for name, expected in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], expected), name


def test_compact_checkpoint_rejects_base_or_delta_contract_changes():
    model = _model()
    payload = moe_checkpoint.build_compact_checkpoint(
        model, 0, metadata={"version": "test"}
    )
    wrong_base = dict(payload)
    wrong_base["base"] = {**payload["base"], "actor_checkpoint_sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="immutable Policy-0814 base mismatch"):
        moe_checkpoint.load_compact_checkpoint(_model(), wrong_base)
    missing_delta = dict(payload)
    missing_delta["delta_state_dict"] = dict(payload["delta_state_dict"])
    missing_delta["delta_state_dict"].pop(next(iter(missing_delta["delta_state_dict"])))
    with pytest.raises(RuntimeError, match="delta inventory mismatch"):
        moe_checkpoint.load_compact_checkpoint(_model(), missing_delta)
    with pytest.raises(ValueError, match="forbidden fields"):
        moe_checkpoint.build_compact_checkpoint(
            model, 0, metadata={"optimizer_state": {}}
        )


def test_checkpoint_pruning_keeps_only_successfully_evaluated_nodes(tmp_path):
    for update in range(6):
        (tmp_path / f"update-{update:06d}.pt").write_bytes(b"checkpoint")
        (tmp_path / f"router-{update:06d}.json").write_text("{}\n")
    with pytest.raises(RuntimeError, match="successful evaluation"):
        moe_checkpoint.prune_non_eval_checkpoints(
            tmp_path, through_update=5, eval_interval=5,
            evaluation_status="FAIL",
        )
    removed = moe_checkpoint.prune_non_eval_checkpoints(
        tmp_path, through_update=5, eval_interval=5,
        evaluation_status="PASS",
    )
    assert [path.name for path in removed] == [
        f"update-{update:06d}.pt" for update in range(1, 5)
    ]
    assert sorted(path.name for path in tmp_path.glob("update-*.pt")) == [
        "update-000000.pt", "update-000005.pt"
    ]
    assert len(tuple(tmp_path.glob("router-*.json"))) == 6
