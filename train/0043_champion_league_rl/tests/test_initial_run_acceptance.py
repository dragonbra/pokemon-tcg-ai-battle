from __future__ import annotations

import importlib
from collections import Counter


module = importlib.import_module("train.0043_champion_league_rl.initial_run")


def test_initial_focal_schedule_is_exact_002_007() -> None:
    lanes = module.focal_schedule(430043001)
    assert Counter(row.deck_id for row in lanes) == {"002": 128, "007": 128}
    assert [row.lane_id for row in lanes] == list(range(256))


def test_67_by_2_opponent_pair_acceptance() -> None:
    report = module.acceptance(schedules=16)
    assert report["status"] == "PASS"
    assert report["opponents"]["deck_count"] == 67
    assert report["opponents"]["policy_ids"] == ["Policy-0809", "Champion-G1"]
    assert report["opponents"]["deck_policy_pairs_covered"] == 134
    assert report["formal_launch_authorized"] is False
