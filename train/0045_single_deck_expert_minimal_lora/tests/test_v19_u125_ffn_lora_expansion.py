from __future__ import annotations

import importlib
from pathlib import Path

import torch


PKG = "train.0045_single_deck_expert_minimal_lora"
ROOT = Path(__file__).resolve().parents[3]


def _actor_logits(model, batch):
    validated, state, options = model.encode_policy(batch)
    decoder_state = model.actor.action_decoder.initialize(
        validated, model.actor_summary(state)
    )
    return model.head.logits(validated, options, decoder_state)


def _models():
    assets = importlib.import_module(f"{PKG}.assets")
    actor_critic = importlib.import_module(f"{PKG}.policy.actor_critic")
    checkpointing = importlib.import_module(f"{PKG}.training.checkpointing")
    v13 = importlib.import_module(f"{PKG}.training.run_v13_policy0814_shared_encoder")
    v19 = importlib.import_module(f"{PKG}.training.run_v19_u125_ffn_lora_expansion")
    registry = assets.AssetRegistry.load(actor_critic.PROJECT_ROOT)
    deck = tuple(
        int(value) for value in (
            actor_critic.PROJECT_ROOT
            / next(row.deck_path for row in registry.decks if row.deck_id == "007")
        ).read_text().splitlines()
    )
    old, _ = actor_critic.load_actor_critic(
        checkpoint=actor_critic.DEFAULT_0814_ACTOR_CHECKPOINT,
        value_checkpoint=actor_critic.DEFAULT_0814_VALUE_CHECKPOINT,
        deck=deck, deck_id="007", adaptation=v13.ADAPTATION,
    )
    payload = torch.load(v19.PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    checkpointing.load_complete_delta(old, payload)
    expanded, _ = actor_critic.load_actor_critic(
        checkpoint=actor_critic.DEFAULT_0814_ACTOR_CHECKPOINT,
        value_checkpoint=actor_critic.DEFAULT_0814_VALUE_CHECKPOINT,
        deck=deck, deck_id="007", adaptation=v19.EXPANDED_ADAPTATION,
    )
    incompatible = expanded.load_state_dict(payload["state_dict"], strict=False)
    assert not incompatible.unexpected_keys
    return old.eval(), expanded.eval()


def test_u125_expansion_is_zero_delta_for_logits_and_greedy_actions():
    runtime = importlib.import_module(f"{PKG}.runtime")
    old, expanded = _models()
    batch = runtime.synthetic_batch(batch_size=2)
    with torch.no_grad():
        old_logits = _actor_logits(old, batch)
        new_logits = _actor_logits(expanded, batch)
        old_validated, old_state, old_options = old.encode_policy(batch)
        new_validated, new_state, new_options = expanded.encode_policy(batch)
        old_actions = old.actor.action_decoder.greedy(
            old_validated, old_options, old.actor_summary(old_state)
        )
        new_actions = expanded.actor.action_decoder.greedy(
            new_validated, new_options, expanded.actor_summary(new_state)
        )
    assert torch.equal(old_logits, new_logits)
    assert torch.equal(old_actions.sequences, new_actions.sequences)
    assert torch.equal(old_actions.lengths, new_actions.lengths)
    assert torch.equal(old_actions.legal, new_actions.legal)


def test_expansion_inventory_and_zero_initialized_effective_deltas():
    _, model = _models()
    names = {
        name: parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    option_ffn = {
        name: value for name, value in names.items()
        if name.startswith("policy_option_lora.ffn_")
    }
    state_ffn = {
        name: value for name, value in names.items()
        if name.startswith("actor.state_encoder.board_encoder.layers.3")
        and ".linear" in name and ".parametrizations." in name
        and not name.endswith(".original")
    }
    norm = {
        name: value for name, value in names.items()
        if name.startswith("actor.option_encoder.cross_attention_transformer.norm.")
    }
    assert len(option_ffn) == 4
    assert len(state_ffn) == 4
    assert set(norm) == {
        "actor.option_encoder.cross_attention_transformer.norm.weight",
        "actor.option_encoder.cross_attention_transformer.norm.bias",
    }
    assert sum(value.numel() for value in option_ffn.values()) == 40_960
    assert sum(value.numel() for value in state_ffn.values()) == 40_960
    assert sum(value.numel() for value in norm.values()) == 640
    assert all(
        torch.count_nonzero(value).item() == 0
        for name, value in {**option_ffn, **state_ffn}.items()
        if name.endswith((".b", "_b"))
    )


def test_expansion_optimizer_groups_are_disjoint_and_have_exact_rates():
    ppo = importlib.import_module(f"{PKG}.training.ppo_full_semantic")
    v19 = importlib.import_module(f"{PKG}.training.run_v19_u125_ffn_lora_expansion")
    _, model = _models()
    trainer = ppo.PPOTrainer(
        model, device=torch.device("cpu"),
        config=ppo.PPOConfig(
            **v19.LEARNING_RATE_PROFILE.rates(),
            batch_size=4096, forward_microbatch_size=256,
            behavior_probe_batch_size=256, offload_reference_after_cache=True,
        ),
    )
    groups = {row["name"]: row for row in trainer.optimizer_group_manifest()}
    assert groups["policy_option_lora"]["base_learning_rate"] == 2e-5
    assert groups["shared_encoder_lora"]["base_learning_rate"] == 2e-5
    assert groups["policy_option_ffn_lora"]["base_learning_rate"] == 3e-5
    assert groups["shared_state_ffn_lora"]["base_learning_rate"] == 1.5e-5
    assert groups["option_final_norm"]["base_learning_rate"] == 5e-6
    assert all(group["weight_decay"] == 0.0 for group in groups.values())
    ids = [
        id(parameter)
        for group in trainer.optimizer.param_groups for parameter in group["params"]
    ]
    assert len(ids) == len(set(ids))
    assert set(ids) == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


def test_expanded_checkpoint_inventory_round_trips_all_new_tensors(tmp_path: Path):
    checkpointing = importlib.import_module(f"{PKG}.training.checkpointing")
    runner = importlib.import_module(f"{PKG}.training.run_v19_u125_ffn_lora_expansion")
    result = runner.materialize_sources(tmp_path)
    payload = torch.load(result["parent_checkpoint"], map_location="cpu", weights_only=True)
    names = set(payload["checkpoint_integrity"]["trainable_parameter_names"])
    assert len([name for name in names if name.startswith("policy_option_lora.ffn_")]) == 4
    assert len([
        name for name in names
        if name.startswith("actor.state_encoder.board_encoder.layers.3")
        and ".linear" in name and ".parametrizations." in name
        and not name.endswith(".original")
    ]) == 4
    assert {
        "actor.option_encoder.cross_attention_transformer.norm.weight",
        "actor.option_encoder.cross_attention_transformer.norm.bias",
    } <= names
    assert result["audit"]["status"] == "PASS"
    assert result["audit"]["logits_exact_equal"] is True
    assert result["audit"]["greedy_actions_exact_equal"] is True


def test_deployment_option_lora_strictly_matches_expanded_training_inventory():
    training = importlib.import_module(f"{PKG}.policy.option_policy_lora")
    deployment = importlib.import_module(
        f"{PKG}.semantic_runtime.deployment.compound_inference"
    )
    source = training.PolicyOnlyOptionLoRA(
        320, rank=16, alpha=16.0, output_projection=True,
        ffn_expansion=True,
    )
    deployed = deployment.PolicyOnlyOptionLoRA(
        320, rank=16, alpha=16.0, output_projection=True,
        ffn_expansion=True,
    )
    deployed.load_state_dict(source.state_dict(), strict=True)
    assert set(deployed.state_dict()) == set(source.state_dict())
    assert len(deployed.state_dict()) == 16
