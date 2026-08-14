from __future__ import annotations

from collections import Counter
import importlib
import json
from pathlib import Path

import pytest
import torch


def _modules():
    runner = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v10"
    )
    common = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v1"
    )
    return runner, common


def test_generalist_jobs_cover_all_decks_without_policy_batch_fragmentation() -> None:
    runner, common = _modules()
    registry = common.AssetRegistry.load(common.PROJECT_ROOT)
    jobs, _, _, _ = common._jobs(
        0,
        registry,
        focal_deck_ids=runner.FOCAL_DECK_IDS,
        focal_deck_id=runner.INITIALIZATION_DECK_ID,
        focal_schedule_mode=runner.FOCAL_SCHEDULE,
        opponent_sampling_mode="uniform_001_067",
    )
    counts = Counter(job.focal_deck_id for job in jobs)
    assert set(counts) == set(runner.FOCAL_DECK_IDS)
    assert set(counts.values()) == {3, 4}
    assert len(jobs) == 256
    assert {job.opponent_policy_id for job in jobs} == {"Champion-G2"}
    assert all(type(job.focal_won_toss) is bool for job in jobs)
    assert all(job.coin_winner_seed > 0 for job in jobs)
    assert len({job.coin_winner_seed for job in jobs}) == 256
    # The legacy constructor field only places the seeded toss winner at the
    # resident context-41 chooser slot.  It is not actual-seat evidence.
    assert all(job.focal_first is job.focal_won_toss for job in jobs)
    vocabulary = common.OwnArchetypeVocabulary.load_version(
        "own_archetypes_v2", project_root=common.PROJECT_ROOT
    )
    own_by_deck = {row.deck_id: row.archetype_id for row in vocabulary.mappings}
    assert all(job.focal_own_archetype_id == own_by_deck[job.focal_deck_id] for job in jobs)


def test_fixed_mode_still_rejects_multiple_focal_decks() -> None:
    _, common = _modules()
    registry = common.AssetRegistry.load(common.PROJECT_ROOT)
    with pytest.raises(ValueError, match="fixed-focal"):
        common._jobs(
            0, registry, focal_deck_ids=("002", "007"), focal_deck_id="066",
            focal_schedule_mode="fixed", opponent_sampling_mode="uniform_001_067",
        )


def test_v10_readiness_binds_exact_v9_u50(tmp_path: Path) -> None:
    runner, _ = _modules()
    checkpoint = tmp_path / "update-000050.pt"
    torch.save({
        "schema_version": "0044_focal_v1_model_only_v1",
        "update": 50,
        "metadata": {"version": runner.PARENT_VERSION},
        "state_dict": {},
    }, checkpoint)
    sha256 = runner.sha256_file(checkpoint)
    report = tmp_path / "report.json"
    report.write_text('{"status":"PASS"}\n', encoding="utf-8")
    manifest = tmp_path / "g3_handoff_u50.json"
    manifest.write_text(json.dumps({
        "schema_version": "0044_v9_u50_g3_handoff_v1",
        "status": "PASS",
        "parent_version": runner.PARENT_VERSION,
        "parent_checkpoint_update": 50,
        "parent_checkpoint_sha256": sha256,
        "benchmark_v2": {
            "status": "PASS", "checkpoint_update": 50,
            "report": str(report), "report_sha256": runner.sha256_file(report),
        },
    }), encoding="utf-8")
    result = runner.readiness(
        version_root=tmp_path / "unused-v10",
        parent_checkpoint=checkpoint,
        handoff_manifest=manifest,
    )
    assert result["parent"]["optimizer"] == "fresh"
    assert result["parent"]["pfsp_state"] is None
    assert result["training"]["duration"] == "indefinite"
    assert result["training"]["focal_deck_ids"] == list(runner.FOCAL_DECK_IDS)
    assert result["training"]["sampling"]["pfsp"] == 0
    assert result["benchmark_v2"]["focal_deck_id"] == "066"
