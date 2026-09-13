from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn


MODULE = "pokemon_tcg_ai.evaluation.public_deck_router_v3_rules"


def _batch(rows: list[list[tuple[int, int, int]]]):
    width = max(1, max(map(len, rows)))
    card_cat = torch.zeros((len(rows), width, 9), dtype=torch.long)
    card_mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for row_index, cards in enumerate(rows):
        for slot, (card_id, owner, knowledge) in enumerate(cards):
            card_cat[row_index, slot, 0] = card_id
            card_cat[row_index, slot, 2] = owner
            card_cat[row_index, slot, 8] = knowledge
            card_mask[row_index, slot] = True
    return SimpleNamespace(card_cat=card_cat, card_mask=card_mask)


def test_v3_routes_all_eight_public_deck_families():
    rules = importlib.import_module(MODULE)
    memory = rules.PublicDeckMemory(8, torch.device("cpu"), default_update=125)
    routes = memory.observe(
        _batch([
            [(646, 2, 1)], [(741, 2, 1)], [(848, 2, 1)], [(119, 2, 1)],
            [(96, 2, 1)], [(677, 2, 1)], [(344, 2, 1)], [(917, 2, 1)],
        ]),
        torch.arange(8),
    )
    assert routes.tolist() == [165, 170, 25, 125, 5, 140, 160, 145]
    assert memory.predicted_deck_codes.tolist() == [1, 2, 3, 7, 8, 9, 11, 71]


def test_v3_ogerpon_is_provisional_then_hydrapple_overrides():
    rules = importlib.import_module(MODULE)
    memory = rules.PublicDeckMemory(1, torch.device("cpu"), default_update=160)
    assert memory.observe(_batch([[(96, 2, 1)]]), torch.tensor([0])).tolist() == [5]
    assert memory.provisional.tolist() == [True]
    assert memory.observe(_batch([[(149, 2, 2)]]), torch.tensor([0])).tolist() == [145]
    assert memory.predicted_deck_codes.tolist() == [71]
    assert memory.provisional.tolist() == [False]


def test_v3_hidden_unknown_and_conflict_use_injected_default():
    rules = importlib.import_module(MODULE)
    memory = rules.PublicDeckMemory(2, torch.device("cpu"), default_update=170)
    first = memory.observe(
        _batch([[(646, 2, 3), (741, 1, 1), (999, 2, 1)], [(646, 2, 1)]]),
        torch.arange(2),
    )
    assert first.tolist() == [170, 165]
    second = memory.observe(_batch([[(741, 2, 1)]]), torch.tensor([1]))
    assert second.tolist() == [170]
    assert memory.conflict.tolist() == [False, True]


def test_v3_default_and_manifest_are_fail_closed():
    rules = importlib.import_module(MODULE)
    assert rules.ROUTED_UPDATES == (5, 25, 125, 140, 145, 160, 165, 170)
    assert rules.DECK_TO_UPDATE == {
        "001": 165, "002": 170, "003": 25, "007": 125,
        "008": 5, "009": 140, "011": 160, "012": 25, "071": 145,
    }
    with pytest.raises(ValueError, match="default update"):
        rules.PublicDeckMemory(1, torch.device("cpu"), default_update=135)
    manifest = rules.route_manifest(125)
    assert manifest["default_update"] == 125
    assert manifest["forbidden_inputs"] == [
        "opponent_exact_deck_id", "hidden_opponent_cards",
        "identity_knowledge_candidate", "critic_outputs",
    ]


def test_v3_actor_audit_routes_v19_state_ffn_and_option_norm_differences():
    runtime = importlib.import_module(
        "pokemon_tcg_ai.evaluation.public_deck_router_v3"
    )
    states = {
        update: {
            "prototype_encoder.weight": torch.tensor([1.0]),
            "state_encoder.layers.1.linear1.weight": torch.tensor(
                [2.0 if update >= 125 else 1.0]
            ),
            "option_encoder.cross_attention_transformer.norm.weight": torch.tensor(
                [3.0 if update >= 125 else 1.0]
            ),
        }
        for update in (5, 25, 125, 140, 145, 160, 165, 170)
    }
    audit = runtime.audit_effective_actor_states(states)
    assert audit["globally_equal_tensor_names"] == ["prototype_encoder.weight"]
    assert audit["routed_tensor_names"] == [
        "option_encoder.cross_attention_transformer.norm.weight",
        "state_encoder.layers.1.linear1.weight",
    ]


def test_v3_cross_route_storage_alias_is_rejected():
    runtime = importlib.import_module(
        "pokemon_tcg_ai.evaluation.public_deck_router_v3"
    )
    shared = nn.Linear(2, 2)
    modules = {
        update: (shared if update in (5, 25) else nn.Linear(2, 2))
        for update in (5, 25, 125, 140, 145, 160, 165, 170)
    }
    with pytest.raises(RuntimeError, match="alias storage"):
        runtime.assert_no_cross_route_storage_aliases(modules)


def test_v3_deployment_memory_matches_evaluation_rules():
    rules = importlib.import_module(MODULE)
    deployment = importlib.import_module(
        "pokemon_tcg_ai.semantic_runtime.deployment."
        "public_deck_memory_v3"
    )
    assert deployment.route_manifest(25) == rules.route_manifest(25)
    memory = deployment.PublicDeckMemory(
        2, torch.device("cpu"), default_update=25
    )
    routes = memory.observe(
        _batch([[(646, 2, 1)], [(999, 2, 1)]]), torch.arange(2)
    )
    assert routes.tolist() == [165, 25]


def test_v3_deployment_activation_switches_complete_actor():
    deployment = importlib.import_module(
        "pokemon_tcg_ai.semantic_runtime.deployment."
        "public_deck_router_v3"
    )
    policy_type = deployment.PublicDeckV3RoutedCompoundPolicy
    policy = object.__new__(policy_type)
    routes = {
        update: SimpleNamespace(
            actor=nn.Linear(2, 2),
            policy_option_lora=nn.Linear(2, 2),
            allocation_head=nn.Linear(2, 2),
        )
        for update in (5, 25, 125, 140, 145, 160, 165, 170)
    }
    policy.policies = routes
    policy._activate(165)
    assert policy.actor is routes[165].actor
    assert policy.policy_option_lora is routes[165].policy_option_lora
    assert policy.allocation_head is routes[165].allocation_head
    assert policy.active_update == 165


def test_v3_staryu_and_starmie_public_evidence_routes_u25():
    rules = importlib.import_module(MODULE)
    memory = rules.PublicDeckMemory(2, torch.device("cpu"), default_update=170)
    routes = memory.observe(
        _batch([[(1030, 2, 1)], [(1031, 2, 2)]]), torch.arange(2)
    )
    assert routes.tolist() == [25, 25]
    assert memory.predicted_deck_codes.tolist() == [12, 12]
