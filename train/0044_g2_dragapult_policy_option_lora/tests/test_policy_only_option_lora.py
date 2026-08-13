from __future__ import annotations

import importlib

import torch


actor_critic = importlib.import_module(
    "train.0044_g2_dragapult_policy_option_lora.policy.actor_critic"
)
assets = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.assets")
runtime = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.runtime")


def _model():
    registry = assets.AssetRegistry.load(actor_critic.PROJECT_ROOT)
    asset = next(row for row in registry.decks if row.deck_id == "007")
    deck = tuple(
        map(int, (actor_critic.PROJECT_ROOT / asset.deck_path).read_text().splitlines())
    )
    return actor_critic.load_actor_critic(deck=deck)[0]


def test_u0_is_exact_g2_and_inventory_is_0040_compatible() -> None:
    model = _model()
    batch = dict(runtime.synthetic_batch(batch_size=2))
    _, _, options = model.encode_dual_options(batch)
    assert torch.equal(options.policy_options, options.value_options)
    assert len(tuple(model.policy_option_lora.parameters())) == 8
    assert sum(row.numel() for row in model.policy_option_lora.parameters()) == 10_240


def test_policy_loss_reaches_lora_b_but_value_loss_cannot_reach_lora() -> None:
    model = _model()
    batch = dict(runtime.synthetic_batch(batch_size=2))
    validated, state, options = model.encode_dual_options(batch)
    lora = tuple(model.policy_option_lora.parameters())
    policy_gradients = torch.autograd.grad(
        options.policy_options.square().sum(), lora, retain_graph=True,
        allow_unused=True,
    )
    assert any(
        gradient is not None and bool(gradient.abs().sum() > 0)
        for gradient in policy_gradients
    )
    value, _ = model.value_and_aux_from_encoded(
        validated, state, options.value_options
    )
    value_gradients = torch.autograd.grad(
        value.square().sum(), lora, allow_unused=True
    )
    assert all(
        gradient is None or torch.equal(gradient, torch.zeros_like(gradient))
        for gradient in value_gradients
    )


def test_nonzero_lora_changes_policy_options_without_changing_value_options() -> None:
    model = _model()
    batch = dict(runtime.synthetic_batch(batch_size=2))
    _, _, before = model.encode_dual_options(batch)
    baseline_value = before.value_options.detach().clone()
    with torch.no_grad():
        for name, parameter in model.policy_option_lora.named_parameters():
            if name.endswith("_b"):
                parameter.fill_(0.1)
    _, _, after = model.encode_dual_options(batch)
    assert torch.equal(after.value_options, baseline_value)
    assert not torch.equal(after.policy_options, after.value_options)
