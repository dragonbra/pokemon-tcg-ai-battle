from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT = Path(__file__).resolve().parents[1]
module = importlib.import_module(
    "train.0047_meta_routed_moe_rl.semantic_runtime.deployment.public_meta29_memory"
)
PublicMeta29Memory = module.PublicMeta29Memory
PublicMeta29Rulebook = module.PublicMeta29Rulebook


def _batch(rows: list[list[tuple[int, int, int]]]):
    """Build (card_id, relative_owner, identity_knowledge) semantic rows."""
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


def test_rulebook_is_explicit_29_way_meta_not_exact_deck_matching():
    rulebook = PublicMeta29Rulebook.load()
    assert rulebook.class_count == 29
    assert rulebook.other_class_id == 14
    assert len(rulebook.rules) == 28
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "registry.json" not in source
    assert "deck_own_archetype_mapping" not in source


def test_public_cards_identify_core_meta_and_hidden_or_self_cards_do_not():
    memory = PublicMeta29Memory(4, torch.device("cpu"))
    actual = memory.observe(
        _batch([
            [(119, 2, 1)],
            [(848, 2, 2)],
            [(648, 2, 3), (743, 1, 1)],
            [(743, 2, 1)],
        ]),
        torch.arange(4),
    )
    assert actual.tolist() == [0, 1, -1, 3]


def test_accumulated_public_cards_upgrade_base_meta_to_specific_variant():
    memory = PublicMeta29Memory(3, torch.device("cpu"))
    assert memory.observe(
        _batch([[(121, 2, 1)], [(1031, 2, 1)], [(96, 2, 1)]]),
        torch.arange(3),
    ).tolist() == [0, 12, 10]
    assert memory.observe(
        _batch([[(133, 2, 1)], [(131, 2, 1)], [(150, 2, 1)]]),
        torch.arange(3),
    ).tolist() == [15, 17, 27]
    assert memory.classification_change_count.tolist() == [1, 1, 1]


def test_full_public_pokemon_signatures_match_all_frozen_29_way_labels():
    rulebook = PublicMeta29Rulebook.load()
    mapping = json.loads(
        (PROJECT / "assets/taxonomy/deck_own_archetype_mapping_v2.json").read_text()
    )
    registry = json.loads((PROJECT / "assets/decks/registry.json").read_text())
    prototypes = json.loads(
        (PROJECT / "semantic_runtime/assets/official_full_engine_prototypes_v2.json")
        .read_text()
    )
    pokemon = {
        int(row["card_id"])
        for row in prototypes["cards"]
        if int(row["card_type"]) == 0
    }
    expected = {
        str(row["deck_id"]): int(row["archetype_id"])
        for row in mapping["decks"]
    }
    actual = {}
    for row in registry["decks"]:
        deck_id = str(row["deck_id"])
        cards = {
            int(value)
            for value in (PROJECT / row["deck_path"]).read_text().splitlines()
        }
        actual[deck_id] = rulebook.classify(cards & pokemon)
    assert actual == expected
