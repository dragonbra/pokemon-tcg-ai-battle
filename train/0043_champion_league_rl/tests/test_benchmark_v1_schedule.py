from __future__ import annotations

import importlib
from collections import Counter
from pathlib import Path


schedule = importlib.import_module(
    "train.0043_champion_league_rl.evaluation.benchmark_v1_schedule"
)
ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_v1_is_meta_balanced_exact_g2_cuda2048() -> None:
    result = schedule.materialize(
        ROOT, focal_deck_id="002", focal_deployment_identity="a" * 64
    )
    jobs = result["jobs"]
    assert len(jobs) == 2048
    assert result["opponent_policy_id"] == "Champion-G2"
    assert result["empty_class_ids"] == [14]
    meta = Counter(row["opponent_meta_archetype_id"] for row in jobs)
    assert set(meta.values()) == {73, 74}
    assert set(meta) == set(range(29)) - {14}
    for class_id in meta:
        decks = Counter(
            row["opponent_deck_id"] for row in jobs
            if row["opponent_meta_archetype_id"] == class_id
        )
        assert max(decks.values()) - min(decks.values()) <= 1
    assert {row["opponent_deck_id"] for row in jobs} == {
        f"{value:03d}" for value in range(1, 68)
    }
    for name in ("engine_seed", "search_seed", "policy_seed"):
        assert len({row[name] for row in jobs}) == 2048


def test_benchmark_v1_schedule_is_reproducible_and_identity_bound() -> None:
    first = schedule.materialize(ROOT, focal_deck_id="007", focal_deployment_identity="b" * 64)
    second = schedule.materialize(ROOT, focal_deck_id="007", focal_deployment_identity="b" * 64)
    changed = schedule.materialize(ROOT, focal_deck_id="007", focal_deployment_identity="c" * 64)
    assert first == second
    assert first["schedule_sha256"] != changed["schedule_sha256"]
