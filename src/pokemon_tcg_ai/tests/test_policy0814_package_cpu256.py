from __future__ import annotations

from collections import Counter
import importlib
from pathlib import Path


PKG = "pokemon_tcg_ai"
PROJECT = Path(__file__).resolve().parents[1]


def test_cpu256_schedule_is_exact_deterministic_half_of_v18_distribution():
    schedule = importlib.import_module(
        f"{PKG}.evaluation.policy0814_exact_deck_cpu256_schedule"
    )
    first = schedule.materialize(
        PROJECT,
        focal_deployment_identity="a" * 64,
    )
    second = schedule.materialize(
        PROJECT,
        focal_deployment_identity="b" * 64,
    )

    expected = {
        "001": 72,
        "002": 34,
        "003": 24,
        "007": 32,
        "008": 25,
        "009": 22,
        "011": 15,
        "071": 32,
    }
    assert schedule.CONTRACT_ID == "0045_v18_distribution_cpu256_gpu_inference_diagnostic_v1"
    assert first["games"] == 256
    assert first["opponent_policy_id"] == "Policy-0814"
    assert first["target_deck_counts"] == expected
    assert first["realized_deck_counts"] == expected
    assert Counter(row["opponent_deck_id"] for row in first["jobs"]) == expected
    assert len({row["source_schedule_slot"] for row in first["jobs"]}) == 256
    assert first["jobs"] == second["jobs"]
    assert first["schedule_sha256"] == second["schedule_sha256"]
    assert first["source_schedule"]["games"] == 512
    assert first["source_schedule"]["realized_deck_counts"] == {
        "001": 143,
        "002": 68,
        "003": 48,
        "007": 64,
        "008": 49,
        "009": 45,
        "011": 31,
        "071": 64,
    }
