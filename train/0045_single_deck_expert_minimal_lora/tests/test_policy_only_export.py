from __future__ import annotations

import importlib
from pathlib import Path

import torch


PKG = "train.0045_single_deck_expert_minimal_lora"
PROJECT = Path(__file__).resolve().parents[1]


def test_policy_only_export_has_exact_actor_logits_and_no_critic(tmp_path):
    assets = importlib.import_module(f"{PKG}.assets")
    actor_critic = importlib.import_module(f"{PKG}.policy.actor_critic")
    export = importlib.import_module(f"{PKG}.policy.policy_only_export")
    runtime = importlib.import_module(f"{PKG}.runtime")
    row = next(item for item in assets.AssetRegistry.load(PROJECT).decks if item.deck_id == "007")
    deck = tuple(map(int, (PROJECT / row.deck_path).read_text().splitlines()))
    model, _ = actor_critic.load_actor_critic(deck=deck, deck_id="007")
    batch = runtime.synthetic_batch(batch_size=2)
    with torch.inference_mode():
        validated, state, options = model.encode_policy(batch)
        before = model.head.logits(
            validated, options,
            model.actor.action_decoder.initialize(validated, model.actor_summary(state)),
        )
    path = tmp_path / "policy-only.pt"
    identity = export.export_policy_only(model, path)
    deployed = export.load_policy_only(path)
    with torch.inference_mode():
        after = deployed.root_logits(batch)
    assert identity.tensor_count > 0
    assert torch.equal(before, after)
    assert not hasattr(deployed, "value_head")
    assert not hasattr(deployed, "value_adapter")
    assert not hasattr(deployed, "prize_aux")
