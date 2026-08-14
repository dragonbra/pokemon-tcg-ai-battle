from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch


PKG = "train.0045_single_deck_expert_minimal_lora"
PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def loaded():
    assets = importlib.import_module(f"{PKG}.assets")
    actor_critic = importlib.import_module(f"{PKG}.policy.actor_critic")
    runtime = importlib.import_module(f"{PKG}.runtime")
    registry = assets.AssetRegistry.load(PROJECT)
    row = next(item for item in registry.decks if item.deck_id == "007")
    deck = tuple(
        int(value)
        for value in (PROJECT / row.deck_path).read_text(encoding="utf-8").splitlines()
    )
    model, _ = actor_critic.load_actor_critic(deck=deck, deck_id="007")
    return model, runtime.synthetic_batch(batch_size=2)


def _actor_logits(model, batch):
    validated, state, options = model.encode_policy(batch)
    decoder_state = model.actor.action_decoder.initialize(
        validated, model.actor_summary(state)
    )
    return model.head.logits(validated, options, decoder_state)


def _critic_outputs(model, batch):
    validated, state, dual = model.encode_dual_options(batch)
    value, auxiliary = model.value_and_aux_from_encoded(
        validated, state, dual.value_options
    )
    return value, auxiliary["v_prize"], auxiliary["meta_logits"]


def _randomize(module):
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn_like(parameter).mul_(7.0))


def test_critic_mutation_cannot_change_actor_logits(loaded):
    model, batch = loaded
    before = _actor_logits(model, batch).detach().clone()
    saved = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if name.startswith(("value_head.", "value_adapter.", "prize_aux."))
    }
    for module in (model.value_head, model.value_adapter, model.prize_aux):
        _randomize(module)
    after = _actor_logits(model, batch).detach()
    assert torch.equal(before, after)
    model.load_state_dict(saved, strict=False)


def test_policy_lora_mutation_cannot_change_critic(loaded):
    model, batch = loaded
    before = tuple(value.detach().clone() for value in _critic_outputs(model, batch))
    saved = {
        name: tensor.detach().clone()
        for name, tensor in model.policy_option_lora.state_dict().items()
    }
    _randomize(model.policy_option_lora)
    after = _critic_outputs(model, batch)
    assert all(torch.equal(left, right) for left, right in zip(before, after))
    model.policy_option_lora.load_state_dict(saved, strict=True)


def test_gradient_ownership_is_disjoint(loaded):
    model, batch = loaded
    policy_prefixes = (
        "actor.action_decoder.", "allocation_head.", "policy_option_lora."
    )
    critic_prefixes = ("value_head.", "value_adapter.", "prize_aux.")

    model.zero_grad(set_to_none=True)
    _actor_logits(model, batch).float().sum().backward()
    policy_grad_names = {
        name for name, parameter in model.named_parameters() if parameter.grad is not None
    }
    assert policy_grad_names
    assert all(name.startswith(policy_prefixes) for name in policy_grad_names)
    assert not any(name.startswith(critic_prefixes) for name in policy_grad_names)

    model.zero_grad(set_to_none=True)
    value, prize, meta = _critic_outputs(model, batch)
    (value.sum() + prize.sum() + meta.sum()).backward()
    critic_grad_names = {
        name for name, parameter in model.named_parameters() if parameter.grad is not None
    }
    assert critic_grad_names
    assert all(name.startswith(critic_prefixes) for name in critic_grad_names)
    assert not any(name.startswith(policy_prefixes) for name in critic_grad_names)


def test_removed_generalist_modules_are_absent(loaded):
    model, _ = loaded
    assert not hasattr(model, "policy_strategy_adapter")
    assert not hasattr(model, "meta_actor_residual")
    assert all(
        "policy_strategy_adapter" not in name and "meta_actor_residual" not in name
        for name, _ in model.named_parameters()
    )
