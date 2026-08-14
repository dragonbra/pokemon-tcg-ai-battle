from __future__ import annotations

import importlib
from collections import Counter
from pathlib import Path


schedule = importlib.import_module(
    "train.0044_g2_dragapult_policy_option_lora.evaluation.benchmark_v2_schedule"
)
ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_v2_is_meta_balanced_policy0809_cuda2048() -> None:
    result = schedule.materialize(
        ROOT, focal_deck_id="007", focal_deployment_identity="a" * 64
    )
    jobs = result["jobs"]
    assert len(jobs) == 2048
    assert result["opponent_policy_id"] == "Policy-0809"
    assert result["selected_class_ids"] == list(range(14)) + [17, 27]
    assert result["excluded_class_ids"] == list(range(14, 17)) + list(range(18, 27)) + [28]
    assert result["seed_derivation_excludes"] == ["focal_deployment_identity"]
    assert result["common_random_numbers"] is True
    meta = Counter(row["opponent_meta_archetype_id"] for row in jobs)
    assert set(meta) == set(range(14)) | {17, 27}
    assert set(meta.values()) == {128}
    by_meta: dict[int, Counter[str]] = {}
    for meta_id in meta:
        by_meta[meta_id] = Counter(
            row["opponent_deck_id"] for row in jobs
            if row["opponent_meta_archetype_id"] == meta_id
        )
    assert all(max(counts.values()) - min(counts.values()) <= 1 for counts in by_meta.values())
    assert {row["opponent_deck_id"] for row in jobs} == {
        "001", "002", "003", "004", "005", "006", "007", "008", "009", "010",
        "011", "012", "013", "014", "017", "018", "019", "020", "021", "022",
        "023", "024", "028", "029", "030", "031", "033", "034", "035", "036",
        "037", "038", "039", "040", "041", "042", "044", "045", "046", "048",
        "050", "051", "052", "053", "055", "059", "060", "062", "063", "067",
    }
    assert "068" not in {row["opponent_deck_id"] for row in jobs}
    assert "069" not in {row["opponent_deck_id"] for row in jobs}
    for name in ("engine_seed", "search_seed", "policy_seed", "coin_winner_seed"):
        assert len({row[name] for row in jobs}) == 2048


def test_benchmark_v2_jobs_are_candidate_independent() -> None:
    first = schedule.materialize(
        ROOT, focal_deck_id="007", focal_deployment_identity="b" * 64
    )
    changed = schedule.materialize(
        ROOT, focal_deck_id="007", focal_deployment_identity="c" * 64
    )
    assert first["jobs"] == changed["jobs"]
    assert first["schedule_sha256"] != changed["schedule_sha256"]
    assert first["common_random_schedule_sha256"] == changed["common_random_schedule_sha256"]
