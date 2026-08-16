from collections import Counter
import importlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
assets = importlib.import_module("train.0045_single_deck_expert_minimal_lora.assets")
own_archetype = importlib.import_module(
    "train.0045_single_deck_expert_minimal_lora.own_archetype"
)


EXPECTED_CARDS = Counter({
    1: 14,
    93: 2, 96: 4, 140: 1, 149: 1, 150: 2, 346: 1,
    709: 1, 710: 2, 917: 2, 918: 1, 920: 1, 1071: 2,
    1080: 1, 1094: 4, 1097: 1, 1121: 4, 1152: 2,
    1182: 2, 1184: 1, 1213: 1, 1227: 4, 1231: 2, 1261: 4,
})


def _cards(registry: object, deck_id: str) -> Counter[int]:
    deck = next(row for row in registry.decks if row.deck_id == deck_id)
    return Counter(
        int(value)
        for value in (PROJECT_ROOT / deck.deck_path).read_text().splitlines()
    )


def test_deck_071_is_exact_registered_training_asset() -> None:
    registry = assets.AssetRegistry.load(PROJECT_ROOT)
    audit = registry.validate_all()
    deck = next(row for row in registry.decks if row.deck_id == "071")
    assert audit.deck_count == audit.training_deck_count == 71
    assert deck.roles == ("training",)
    assert sum(EXPECTED_CARDS.values()) == 60
    assert _cards(registry, "071") == EXPECTED_CARDS


def test_deck_071_is_the_declared_exact_delta_from_023() -> None:
    registry = assets.AssetRegistry.load(PROJECT_ROOT)
    delta = _cards(registry, "071") - _cards(registry, "023")
    removed = _cards(registry, "023") - _cards(registry, "071")
    assert delta == Counter({1152: 1, 1213: 1, 1231: 1})
    assert removed == Counter({655: 1, 1188: 1, 1201: 1})


def test_deck_071_uses_existing_hydrapple_meganium_taxonomy() -> None:
    vocabulary = own_archetype.OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=PROJECT_ROOT
    )
    mapping = next(row for row in vocabulary.mappings if row.deck_id == "071")
    assert vocabulary.class_count == 29
    assert mapping.archetype_id == 27
    assert vocabulary.classes[27].name == "hydrapple_meganium"
    assert "071" in vocabulary.classes[27].deck_ids
