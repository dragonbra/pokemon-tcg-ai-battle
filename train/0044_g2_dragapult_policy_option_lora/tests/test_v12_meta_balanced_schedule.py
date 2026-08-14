from __future__ import annotations

from collections import Counter, defaultdict
import importlib


def _modules():
    schedule = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.league.meta_balanced"
    )
    own = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.own_archetype"
    )
    return schedule, own


def test_512_schedule_balances_active_meta_then_member_decks() -> None:
    schedule, own = _modules()
    vocabulary = own.OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=schedule.PROJECT_ROOT
    )
    rows = schedule.balanced_meta_deck_schedule(
        quota_seed=440120000,
        shuffle_seed=440120101,
        mappings=vocabulary.mappings,
        lanes=512,
    )
    assert len(rows) == 512
    assert tuple(row.lane_id for row in rows) == tuple(range(512))
    meta_counts = Counter(row.archetype_id for row in rows)
    assert len(meta_counts) == 28
    assert set(meta_counts.values()) == {18, 19}
    assert set(row.deck_id for row in rows) == {
        f"{value:03d}" for value in range(1, 68)
    }
    by_meta: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_meta[row.archetype_id][row.deck_id] += 1
    assert all(
        max(counts.values()) - min(counts.values()) <= 1
        for counts in by_meta.values()
    )
    assert rows == schedule.balanced_meta_deck_schedule(
        quota_seed=440120000,
        shuffle_seed=440120101,
        mappings=vocabulary.mappings,
        lanes=512,
    )


def test_focal_and_opponent_share_meta_marginal_without_pair_coupling() -> None:
    schedule, own = _modules()
    vocabulary = own.OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=schedule.PROJECT_ROOT
    )
    focal = schedule.balanced_meta_deck_schedule(
        quota_seed=440120007,
        shuffle_seed=440120107,
        mappings=vocabulary.mappings,
        lanes=512,
    )
    opponent = schedule.balanced_meta_deck_schedule(
        quota_seed=440120007,
        shuffle_seed=440120207,
        mappings=vocabulary.mappings,
        lanes=512,
    )
    assert Counter(row.archetype_id for row in focal) == Counter(
        row.archetype_id for row in opponent
    )
    assert any(
        left.archetype_id != right.archetype_id or left.deck_id != right.deck_id
        for left, right in zip(focal, opponent, strict=True)
    )
    assert sum(
        left.deck_id == right.deck_id
        for left, right in zip(focal, opponent, strict=True)
    ) < 64


def test_invalid_meta_schedule_inputs_fail_closed() -> None:
    schedule, own = _modules()
    vocabulary = own.OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=schedule.PROJECT_ROOT
    )
    try:
        schedule.balanced_meta_deck_schedule(
            quota_seed=1, shuffle_seed=2,
            mappings=vocabulary.mappings[:-1], lanes=512,
        )
    except ValueError as error:
        assert "001-067" in str(error)
    else:
        raise AssertionError("incomplete exact-deck mapping was accepted")
