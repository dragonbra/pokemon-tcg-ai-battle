from __future__ import annotations

import importlib

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
moe_run = importlib.import_module(
    "train.0047_meta_routed_moe_rl.training.run_v1_moe"
)
moe_ppo = importlib.import_module(
    "train.0047_meta_routed_moe_rl.training.ppo_moe"
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
    assert torch.allclose(gate[:, 0], torch.full((6,), 0.2), atol=1e-6)
    assert torch.allclose(gate.max(dim=1).values, torch.full((6,), 0.8), atol=1e-6)
    assert torch.equal(gate.gt(0).sum(dim=1), torch.full((6,), 2))
    fallback = model.gate_probabilities(torch.tensor((-1, 4, 28)))
    assert torch.equal(fallback[:, 0], torch.ones(3))
    assert torch.equal(fallback[:, 1:], torch.zeros((3, 6)))
    evaluated = distribution.evaluate_moe_actions(
        model, dict(batch.items()), action.sequences, action.lengths,
        torch.zeros(6, dtype=torch.bool), meta, (None,) * 6,
    )
    (-evaluated.joint_log_prob.mean()).backward()
    assert torch.isfinite(evaluated.joint_log_prob).all()
    assert model.router_logits.grad is not None
    assert torch.isfinite(model.router_logits.grad).all()
    assert float(model.router_logits.grad.norm()) > 0


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
    assert calls == [1, 0, 0, 0, 0, 1, 0]
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


def test_v9_uses_win_only_actor_advantage_and_u0_then_every_five_eval():
    config = moe_ppo.PPOConfig()
    assert config.prize_aux_actor_weight == 0.0
    assert moe_run.PRIZE_AUX_ACTOR_WEIGHT == 0.0
    assert moe_run.EVAL_AT_U0 is True
    assert moe_run.EVAL_INTERVAL == 5


def test_router_heatmap_generates_png(tmp_path):
    output = moe_run._router_heatmap(_model(), 0, tmp_path / "router.png")
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
