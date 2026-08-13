from __future__ import annotations

import importlib
from collections import Counter


module = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.initial_run")


def test_initial_focal_schedule_is_exact_002_007() -> None:
    lanes = module.focal_schedule(430044001)
    assert Counter(row.deck_id for row in lanes) == {"002": 128, "007": 128}
    assert [row.lane_id for row in lanes] == list(range(256))


def test_balanced_focal_schedule_covers_all_67_decks_every_update() -> None:
    decks = tuple(f"{index:03d}" for index in range(1, 68))
    lanes = module.balanced_focal_schedule(430044918, deck_ids=decks)
    counts = Counter(row.deck_id for row in lanes)
    assert len(lanes) == 256
    assert [row.lane_id for row in lanes] == list(range(256))
    assert set(counts) == set(decks)
    assert set(counts.values()) == {3, 4}
    assert sum(count == 4 for count in counts.values()) == 55


def test_balanced_focal_schedule_is_seeded_and_rotates_extra_lanes() -> None:
    decks = tuple(f"{index:03d}" for index in range(1, 68))
    first = module.balanced_focal_schedule(430044918, deck_ids=decks)
    same = module.balanced_focal_schedule(430044918, deck_ids=decks)
    next_update = module.balanced_focal_schedule(430044919, deck_ids=decks)
    assert first == same
    assert first != next_update
    assert Counter(row.deck_id for row in first) != Counter(
        row.deck_id for row in next_update
    )


def test_67_by_2_opponent_pair_acceptance() -> None:
    report = module.acceptance(schedules=16)
    assert report["status"] == "PASS"
    assert report["opponents"]["deck_count"] == 67
    assert report["opponents"]["policy_ids"] == ["Policy-0809", "Champion-G1"]
    assert report["opponents"]["deck_policy_pairs_covered"] == 134
    assert report["formal_launch_authorized"] is False
