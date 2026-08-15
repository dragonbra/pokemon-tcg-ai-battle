from __future__ import annotations

import importlib
from types import SimpleNamespace

import torch

PublicMetaMemory = importlib.import_module(
    "train.0045_single_deck_expert_minimal_lora.evaluation.public_meta_router_v1"
).PublicMetaMemory
_select_rows = importlib.import_module(
    "train.0045_single_deck_expert_minimal_lora.evaluation.public_meta_router_v1"
)._select_rows


def _batch(rows: list[list[tuple[int, int, int]]]):
    """(card_id, relative_owner, identity_knowledge) rows."""
    width = max(1, max(map(len, rows)))
    card_cat = torch.zeros((len(rows), width, 9), dtype=torch.long)
    card_mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for row, cards in enumerate(rows):
        for slot, (card_id, owner, knowledge) in enumerate(cards):
            card_cat[row, slot, 0] = card_id
            card_cat[row, slot, 2] = owner
            card_cat[row, slot, 8] = knowledge
            card_mask[row, slot] = True
    return SimpleNamespace(card_cat=card_cat, card_mask=card_mask)


def test_basic_pokemon_locks_dragapult_and_lopunny_immediately():
    memory = PublicMetaMemory(2, torch.device("cpu"))
    routes = memory.observe(
        _batch([[(119, 2, 1)], [(848, 2, 1)]]), torch.tensor([0, 1])
    )
    assert routes.tolist() == [282, 40]
    assert memory.locked_meta.tolist() == [0, 1]
    assert memory.first_lock_observation.tolist() == [1, 1]


def test_starmie_is_provisional_until_duskull_is_seen_later():
    memory = PublicMetaMemory(1, torch.device("cpu"))
    assert memory.observe(
        _batch([[(1030, 2, 1)]]), torch.tensor([0])
    ).tolist() == [282]
    assert memory.locked_meta.tolist() == [-1]
    assert memory.provisional_meta.tolist() == [12]
    assert memory.observe(
        _batch([[(131, 2, 2)]]), torch.tensor([0])
    ).tolist() == [40]
    assert memory.locked_meta.tolist() == [17]
    assert memory.first_lock_observation.tolist() == [2]


def test_hidden_candidate_and_self_cards_never_trigger():
    memory = PublicMetaMemory(1, torch.device("cpu"))
    memory.observe(
        _batch([[(848, 2, 3), (119, 1, 1), (743, 2, 0)]]), torch.tensor([0])
    )
    assert memory.routes(torch.tensor([0])).tolist() == [282]
    assert memory.locked_meta.tolist() == [-1]
    assert memory.seen.sum().item() == 0


def test_unclassified_public_pokemon_remains_on_default_u282():
    memory = PublicMetaMemory(1, torch.device("cpu"))
    routes = memory.observe(_batch([[(999, 2, 1)]]), torch.tensor([0]))
    assert routes.tolist() == [282]
    assert memory.locked_meta.tolist() == [-1]


def test_teal_mask_ogerpon_triggers_folded_u200_route_and_lock_is_monotonic():
    memory = PublicMetaMemory(1, torch.device("cpu"))
    memory.observe(_batch([[(96, 2, 1)]]), torch.tensor([0]))
    assert memory.routes(torch.tensor([0])).tolist() == [200]
    assert memory.locked_meta.tolist() == [8]
    memory.observe(_batch([[(741, 2, 1)]]), torch.tensor([0]))
    assert memory.routes(torch.tensor([0])).tolist() == [200]
    memory.observe(_batch([[(848, 2, 1)]]), torch.tensor([0]))
    assert memory.routes(torch.tensor([0])).tolist() == [200]
    assert memory.locked_meta.tolist() == [8]


def test_single_card_routes_cover_nondefault_experts():
    memory = PublicMetaMemory(5, torch.device("cpu"))
    routes = memory.observe(
        _batch([
            [(646, 2, 1)], [(743, 2, 1)], [(89, 2, 1)],
            [(917, 2, 1)], [(150, 2, 1)],
        ]),
        torch.arange(5),
    )
    assert routes.tolist() == [40, 90, 200, 200, 200]
    assert memory.locked_meta.tolist() == [2, 3, 6, 8, 8]


def test_every_meganium_stage_folds_08_27_28_to_u200():
    memory = PublicMetaMemory(4, torch.device("cpu"))
    routes = memory.observe(
        _batch([[(917, 2, 1)], [(709, 2, 1)], [(710, 2, 1)], [(919, 2, 1)]]),
        torch.arange(4),
    )
    assert routes.tolist() == [200, 200, 200, 200]
    assert memory.locked_meta.tolist() == [8, 8, 8, 8]


def test_any_grass_brother_or_ambiguous_u200_card_routes_immediately():
    memory = PublicMetaMemory(8, torch.device("cpu"))
    routes = memory.observe(
        _batch([
            [(92, 2, 1)],   # Festival Applin: route now, classify later.
            [(93, 2, 1)],   # Festival/Hydrapple-ambiguous Dipplin.
            [(96, 2, 1)],   # Teal Mask Ogerpon ex.
            [(149, 2, 1)],  # Hydrapple-line Applin.
            [(402, 2, 1)],  # Smoliv.
            [(650, 2, 1)],  # Bulbasaur.
            [(708, 2, 1)],  # Chikorita.
            [(756, 2, 1)],  # Mega Kangaskhan ex (meta 28).
        ]),
        torch.arange(8),
    )
    assert routes.tolist() == [200] * 8
    assert memory.locked_meta.tolist()[:2] == [-1, -1]
    assert memory.locked_meta.tolist()[2:] == [8] * 6


def test_readonly_mapping_batch_is_actually_sliced():
    class Readonly(dict):
        @classmethod
        def from_mapping(cls, values):
            return cls(values)

    batch = Readonly({"mask": torch.arange(12).view(3, 4), "scalar": "same"})
    selected = _select_rows(batch, torch.tensor([2]), 3)
    assert isinstance(selected, Readonly)
    assert selected["mask"].tolist() == [[8, 9, 10, 11]]
    assert selected["scalar"] == "same"
