from collections import Counter
import importlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
assets = importlib.import_module("train.0045_single_deck_expert_minimal_lora.assets")
own_archetype = importlib.import_module(
    "train.0045_single_deck_expert_minimal_lora.own_archetype"
)


def test_deck_070_is_exact_registered_training_asset() -> None:
    registry = assets.AssetRegistry.load(PROJECT_ROOT)
    audit = registry.validate_all()
    deck = next(row for row in registry.decks if row.deck_id == "070")
    cards = tuple(
        int(value)
        for value in (PROJECT_ROOT / deck.deck_path).read_text().splitlines()
    )
    assert audit.deck_count == audit.training_deck_count == 70
    assert deck.roles == ("training",)
    assert len(cards) == 60
    assert Counter(cards) == Counter({
        2: 3, 5: 3, 7: 2,
        112: 2, 119: 4, 120: 4, 121: 3, 131: 2, 132: 1, 133: 1,
        140: 1, 235: 1, 1071: 1,
        1079: 3, 1080: 1, 1086: 4, 1097: 2, 1121: 4, 1152: 4,
        1182: 3, 1198: 3, 1227: 4, 1231: 2, 1260: 2,
    })


def test_deck_070_uses_existing_dragapult_dusknoir_taxonomy() -> None:
    vocabulary = own_archetype.OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT_ROOT
    )
    mapping = next(row for row in vocabulary.mappings if row.deck_id == "070")
    assert vocabulary.class_count == 29
    assert mapping.archetype_id == 15
    assert vocabulary.classes[15].name == "dragapult_dusknoir"
