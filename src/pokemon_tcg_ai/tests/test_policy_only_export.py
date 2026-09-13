from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch


PKG = "pokemon_tcg_ai"
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


def test_kaggle_runtime_never_consumes_critic_for_0045_candidate(tmp_path):
    assets = importlib.import_module(f"{PKG}.assets")
    actor_critic = importlib.import_module(f"{PKG}.policy.actor_critic")
    candidate = importlib.import_module(f"{PKG}.evaluation.candidate")
    compound = importlib.import_module(
        f"{PKG}.semantic_runtime.deployment.compound_inference"
    )
    run_v1 = importlib.import_module(f"{PKG}.training.run_v1")
    runtime = importlib.import_module(f"{PKG}.runtime")
    row = next(
        item for item in assets.AssetRegistry.load(PROJECT).decks
        if item.deck_id == "007"
    )
    deck = tuple(map(int, (PROJECT / row.deck_path).read_text().splitlines()))
    model, _ = actor_critic.load_actor_critic(deck=deck, deck_id="007")
    checkpoint = tmp_path / "update-000040.pt"
    torch.save(run_v1._checkpoint(model, 40), checkpoint)
    portable = tmp_path / "model.bin"
    candidate.materialize(
        checkpoint=checkpoint,
        base_portable=(
            PROJECT / "assets/policies/definitions/champion_g002/model.bin"
        ),
        deck=deck,
        deck_id="007",
        own_archetype_id=0,
        output=portable,
        device=torch.device("cpu"),
    )
    deployed = compound.PortableCompoundSemanticPolicy.from_checkpoint(portable, deck)
    assert deployed.policy_strategy_adapter is None
    assert deployed.meta_actor_residual is None
    batch = runtime.synthetic_batch(batch_size=2)
    with torch.inference_mode():
        validated, state, _, options = deployed._encode_options(batch)
        before = deployed._greedy_strategy(validated, state, options)[0].clone()
        for module in (deployed.value_head, deployed.value_adapter):
            for parameter in module.parameters():
                parameter.add_(10.0 * torch.randn_like(parameter))
        validated, state, _, options = deployed._encode_options(batch)
        after = deployed._greedy_strategy(validated, state, options)[0]
    assert torch.equal(before, after)


def test_portable_candidate_loader_preserves_exported_identity_and_fails_closed(tmp_path):
    assets = importlib.import_module(f"{PKG}.assets")
    actor_critic = importlib.import_module(f"{PKG}.policy.actor_critic")
    candidate = importlib.import_module(f"{PKG}.evaluation.candidate")
    run_v1 = importlib.import_module(f"{PKG}.training.run_v1")
    row = next(item for item in assets.AssetRegistry.load(PROJECT).decks if item.deck_id == "007")
    deck = tuple(map(int, (PROJECT / row.deck_path).read_text().splitlines()))
    model, _ = actor_critic.load_actor_critic(deck=deck, deck_id="007")
    checkpoint = tmp_path / "update-000040.pt"
    torch.save(run_v1._checkpoint(model, 40), checkpoint)
    portable = tmp_path / "model.bin"
    _, exported = candidate.materialize(
        checkpoint=checkpoint,
        base_portable=PROJECT / "assets/policies/definitions/champion_g002/model.bin",
        deck=deck,
        deck_id="007",
        own_archetype_id=0,
        output=portable,
        device=torch.device("cpu"),
    )

    loaded, audit = candidate.load_portable_candidate(
        portable=portable,
        deck=deck,
        deck_id="007",
        own_archetype_id=0,
        device=torch.device("cpu"),
        expected_source_checkpoint_sha256=exported.source_checkpoint_sha256,
        expected_portable_checkpoint_sha256=exported.portable_checkpoint_sha256,
        expected_effective_candidate_sha256=exported.effective_candidate_sha256,
    )
    assert audit == exported
    assert all(
        value.dtype == torch.float32
        for value in loaded.parameters()
        if value.is_floating_point()
    )
    with pytest.raises(RuntimeError, match="effective candidate identity mismatch"):
        candidate.load_portable_candidate(
            portable=portable,
            deck=deck,
            deck_id="007",
            own_archetype_id=0,
            device=torch.device("cpu"),
            expected_source_checkpoint_sha256=exported.source_checkpoint_sha256,
            expected_portable_checkpoint_sha256=exported.portable_checkpoint_sha256,
            expected_effective_candidate_sha256="0" * 64,
        )
