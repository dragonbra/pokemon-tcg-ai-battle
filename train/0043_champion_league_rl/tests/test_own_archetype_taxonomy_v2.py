from pathlib import Path
import importlib

AssetRegistry = importlib.import_module(
    "train.0043_champion_league_rl.assets"
).AssetRegistry
OwnArchetypeVocabulary = importlib.import_module(
    "train.0043_champion_league_rl.own_archetype"
).OwnArchetypeVocabulary


ROOT = Path(__file__).resolve().parents[1]


def test_exact_67_append_only_taxonomy():
    v1 = OwnArchetypeVocabulary.load_version("0042_own_archetypes_v1")
    v2 = OwnArchetypeVocabulary.load_version("own_archetypes_v2", project_root=ROOT)
    assert v1.class_count == 15
    assert v2.class_count == 29
    assert [(row.archetype_id, row.name) for row in v2.classes[:15]] == [
        (row.archetype_id, row.name) for row in v1.classes
    ]
    assert len(v2.mappings) == 67
    assert len({row.deck_id for row in v2.mappings}) == 67
    assert v2.embedding_width == v1.embedding_width == 16


def test_exact_registry_is_source_of_truth_and_frozen_pool_stays_55():
    registry = AssetRegistry.load(ROOT)
    audit = registry.validate_all()
    assert audit.deck_count == audit.training_deck_count == 67
    assert audit.evaluation_deck_count == 55
    v2 = OwnArchetypeVocabulary.load_version("own_archetypes_v2", project_root=ROOT)
    for deck in registry.decks:
        cards = tuple(int(row) for row in (ROOT / deck.deck_path).read_text().splitlines())
        resolved = v2.resolve_exact_deck(deck.deck_id, cards)
        assert resolved.value == next(
            row.archetype_id for row in v2.mappings if row.deck_id == deck.deck_id
        )


def test_key_strategic_splits_and_other_is_fallback_only():
    v2 = OwnArchetypeVocabulary.load_version("own_archetypes_v2", project_root=ROOT)
    mapping = {row.deck_id: row.archetype_id for row in v2.mappings}
    names = {row.name: row.archetype_id for row in v2.classes}
    assert mapping["007"] == names["dragapult_ex"]
    assert mapping["043"] == mapping["057"] == names["dragapult_dusknoir"]
    assert mapping["066"] == names["dragapult_dusknoir"]
    assert mapping["067"] == names["dragapult_ex"]
    assert mapping["061"] == names["dragapult_blaziken"]
    assert mapping["046"] == mapping["050"] == names["mega_starmie_dusknoir"]
    assert mapping["065"] == names["crustle_great_tusk_mill"]
    assert names["other"] not in mapping.values()


def test_hashes_are_deterministic():
    left = OwnArchetypeVocabulary.load_version("own_archetypes_v2", project_root=ROOT)
    right = OwnArchetypeVocabulary.load_version("own_archetypes_v2", project_root=ROOT)
    assert left.taxonomy_sha256 == right.taxonomy_sha256
    assert left.mapping_sha256 == right.mapping_sha256
