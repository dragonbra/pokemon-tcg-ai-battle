from __future__ import annotations

import importlib
from collections import Counter
from pathlib import Path


runner = importlib.import_module("train.0043_champion_league_rl.training.run_v3")
shared = importlib.import_module("train.0043_champion_league_rl.training.run_v1")
assets = importlib.import_module("train.0043_champion_league_rl.assets")


def test_v3_identity_and_parent_are_exact() -> None:
    assert runner.VERSION == "V3_generalist_focal_001_067"
    assert runner.START_UPDATE == 207
    assert runner.FOCAL_DECK_IDS == tuple(f"{index:03d}" for index in range(1, 68))
    assert runner.PARENT_CHECKPOINT.name == "update-000207.pt"
    assert runner.PARENT_PFSP_STATE.name == "pfsp_state.json"
    assert runner.PARENT_CHECKPOINT.is_file()
    assert runner.PARENT_PFSP_STATE.is_file()


def test_v3_readiness_is_non_mutating_and_binds_u207(tmp_path: Path) -> None:
    result = runner.readiness(version_root=tmp_path / runner.VERSION)
    assert result["status"] == "READY_AWAITING_USER_LAUNCH"
    assert result["long_training_started"] is False
    assert result["parent"]["checkpoint_update"] == 207
    assert result["parent"]["path"].endswith("update-000207.pt")
    assert len(result["focal"]["deck_ids"]) == 67
    assert result["focal"]["lanes_per_rollout"] == 256
    assert result["focal"]["per_deck_games"] == [3, 4]
    assert result["optimizer"]["initialization"] == "fresh"
    assert result["reference_anchor_update"] == 0
    assert not (tmp_path / runner.VERSION).exists()


def test_v3_jobs_bind_all_focal_decks_and_own_archetypes() -> None:
    registry = assets.AssetRegistry.load(runner.PROJECT_ROOT)
    jobs, _, _, _ = shared._jobs(
        207, registry, focal_deck_ids=runner.FOCAL_DECK_IDS
    )
    counts = Counter(job.focal_deck_id for job in jobs)
    assert len(jobs) == 256
    assert set(counts) == set(runner.FOCAL_DECK_IDS)
    assert set(counts.values()) == {3, 4}
    assert all(len(job.focal_deck) == 60 for job in jobs)
    assert all(0 <= job.focal_own_archetype_id < 29 for job in jobs)


def test_pristine_restart_rejects_any_metric_or_later_checkpoint(tmp_path: Path) -> None:
    paths = {name: tmp_path / name for name in ("artifact", "checkpoint", "tensorboard", "wandb")}
    paths["artifact"].mkdir(); paths["checkpoint"].mkdir()
    (paths["artifact"] / "training_metrics.jsonl").write_text("")
    (paths["artifact"] / "training_config.json").write_text(
        __import__("json").dumps({
            "start_update": 207,
            "parent_checkpoint_sha256": runner.sha256_file(runner.PARENT_CHECKPOINT),
            "focal_deck_ids": list(runner.FOCAL_DECK_IDS),
            "optimizer_initialization": "fresh",
        })
    )
    import torch
    torch.save({"update": 207}, paths["checkpoint"] / "update-000207.pt")
    assert shared._pristine_restart_allowed(
        paths, start_update=207, parent_checkpoint=runner.PARENT_CHECKPOINT,
        focal_deck_ids=runner.FOCAL_DECK_IDS,
    )
    (paths["artifact"] / "training_metrics.jsonl").write_text("{}\n")
    assert not shared._pristine_restart_allowed(
        paths, start_update=207, parent_checkpoint=runner.PARENT_CHECKPOINT,
        focal_deck_ids=runner.FOCAL_DECK_IDS,
    )
