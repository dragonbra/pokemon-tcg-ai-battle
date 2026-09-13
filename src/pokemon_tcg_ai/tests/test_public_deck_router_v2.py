from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize


MODULE = "pokemon_tcg_ai.evaluation.public_deck_router_v2_rules"


def _batch(rows: list[list[tuple[int, int, int]]]):
    """Build (card_id, relative_owner, identity_knowledge) semantic rows."""

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


def test_public_signatures_route_all_nondefault_specialists():
    rules = importlib.import_module(MODULE)
    memory = rules.PublicDeckMemory(5, torch.device("cpu"))
    routes = memory.observe(
        _batch([
            [(646, 2, 1)],  # 001 Marnie's Impidimp
            [(741, 2, 1)],  # 002 Abra
            [(848, 2, 1)],  # 003 Buneary
            [(119, 2, 1)],  # 007 Dreepy
            [(96, 2, 1)],   # 008/071 Ogerpon provisional
        ]),
        torch.arange(5),
    )
    assert routes.tolist() == [5, 15, 25, 95, 5]
    assert memory.predicted_deck_codes.tolist() == [1, 2, 3, 7, 8]
    assert memory.provisional.tolist() == [False, False, False, False, True]


def test_default_families_are_classified_without_changing_u70_route():
    rules = importlib.import_module(MODULE)
    memory = rules.PublicDeckMemory(3, torch.device("cpu"))
    routes = memory.observe(
        _batch([[(677, 2, 1)], [(344, 2, 1)], [(917, 2, 1)]]),
        torch.arange(3),
    )
    assert routes.tolist() == [70, 70, 70]
    assert memory.predicted_deck_codes.tolist() == [9, 11, 71]
    assert memory.provisional.tolist() == [False, False, False]


def test_ogerpon_provisional_changes_to_071_when_public_evidence_arrives():
    rules = importlib.import_module(MODULE)
    memory = rules.PublicDeckMemory(1, torch.device("cpu"))
    assert memory.observe(
        _batch([[(96, 2, 1)]]), torch.tensor([0])
    ).tolist() == [5]
    assert memory.provisional.tolist() == [True]
    assert memory.observe(
        _batch([[(149, 2, 2)]]), torch.tensor([0])
    ).tolist() == [70]
    assert memory.predicted_deck_codes.tolist() == [71]
    assert memory.provisional.tolist() == [False]
    assert memory.route_change_count.tolist() == [2]


def test_hidden_candidate_self_and_unknown_cards_stay_default():
    rules = importlib.import_module(MODULE)
    memory = rules.PublicDeckMemory(1, torch.device("cpu"))
    routes = memory.observe(
        _batch([[(646, 2, 3), (741, 1, 1), (999, 2, 1)]]),
        torch.tensor([0]),
    )
    assert routes.tolist() == [70]
    assert memory.predicted_deck_codes.tolist() == [-1]
    assert memory.seen.sum().item() == 0


def test_conflicting_strong_families_fail_closed_to_default():
    rules = importlib.import_module(MODULE)
    memory = rules.PublicDeckMemory(1, torch.device("cpu"))
    assert memory.observe(
        _batch([[(646, 2, 1)]]), torch.tensor([0])
    ).tolist() == [5]
    assert memory.observe(
        _batch([[(741, 2, 1)]]), torch.tensor([0])
    ).tolist() == [70]
    assert memory.predicted_deck_codes.tolist() == [-2]
    assert memory.conflict.tolist() == [True]
    assert memory.route_change_count.tolist() == [2]


def test_rule_manifest_and_source_updates_are_exact():
    rules = importlib.import_module(MODULE)
    assert rules.DEFAULT_UPDATE == 70
    assert rules.ROUTED_UPDATES == (5, 15, 25, 70, 95)
    assert rules.DECK_TO_UPDATE == {
        "001": 5, "002": 15, "003": 25, "007": 95,
        "008": 5, "009": 70, "011": 70, "071": 70,
    }
    assert rules.ROUTE_MANIFEST["forbidden_inputs"] == [
        "opponent_exact_deck_id",
        "hidden_opponent_cards",
        "identity_knowledge_candidate",
        "critic_outputs",
    ]


def test_shared_actor_audit_excludes_only_action_decoder_tensors():
    runtime = importlib.import_module(
        "pokemon_tcg_ai.evaluation.public_deck_router_v2"
    )
    states = {
        update: {
            "state_encoder.shared": torch.tensor([1.0, 2.0]),
            "action_decoder.weight": torch.tensor([float(update)]),
        }
        for update in (5, 15, 25, 70, 95)
    }
    audit = runtime.audit_shared_actor_states(states)
    assert audit["status"] == "PASS"
    assert audit["reference_update"] == 70
    assert audit["shared_tensor_names"] == ["state_encoder.shared"]
    assert audit["routed_actor_tensor_names"] == ["action_decoder.weight"]


def test_shared_actor_audit_rejects_cross_checkpoint_backbone_changes():
    runtime = importlib.import_module(
        "pokemon_tcg_ai.evaluation.public_deck_router_v2"
    )
    states = {
        update: {
            "state_encoder.shared": torch.tensor([1.0]),
            "action_decoder.weight": torch.tensor([float(update)]),
        }
        for update in (5, 15, 25, 70, 95)
    }
    states[95]["state_encoder.shared"] = torch.tensor([2.0])
    with pytest.raises(RuntimeError, match="shared Actor tensor mismatch"):
        runtime.audit_shared_actor_states(states)


def test_effective_actor_state_ignores_zero_delta_random_adapter_storage():
    runtime = importlib.import_module(
        "pokemon_tcg_ai.evaluation.public_deck_router_v2"
    )

    class ZeroDelta(nn.Module):
        def __init__(self, seed: int):
            super().__init__()
            generator = torch.Generator().manual_seed(seed)
            self.a = nn.Parameter(torch.randn((2, 2), generator=generator))
            self.b = nn.Parameter(torch.zeros((2, 2)))

        def forward(self, original: torch.Tensor) -> torch.Tensor:
            return original + self.a @ self.b

    actors = []
    for seed in (1, 2):
        actor = nn.Module()
        actor.layer = nn.Linear(2, 2, bias=False)
        actor.layer.weight.data.fill_(1.0)
        parametrize.register_parametrization(actor.layer, "weight", ZeroDelta(seed))
        actors.append(actor)
    left = runtime._effective_actor_state(actors[0])
    right = runtime._effective_actor_state(actors[1])
    assert set(left) == {"layer.weight"}
    assert torch.equal(left["layer.weight"], right["layer.weight"])
