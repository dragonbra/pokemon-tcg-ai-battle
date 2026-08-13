from __future__ import annotations

import importlib
import json
from pathlib import Path

import torch


runner = importlib.import_module("train.0043_champion_league_rl.training.run_v6")


def test_v6_restart_identity_is_exact() -> None:
    assert runner.VERSION == "V6_generalist_focal_001_067_u233_restart"
    assert runner.START_UPDATE == 233
    assert runner.PARENT_CHECKPOINT.name == "update-000233.pt"
    assert runner.FOCAL_DECK_IDS == tuple(f"{index:03d}" for index in range(1, 68))
    assert runner.FORWARD_MICROBATCH_SIZE == 512


def test_v6_readiness_binds_clean_u233_boundary(tmp_path: Path) -> None:
    result = runner.readiness(version_root=tmp_path / runner.VERSION)
    assert result["status"] == "READY_AWAITING_USER_LAUNCH"
    assert result["parent"]["checkpoint_update"] == 233
    assert result["parent"]["pfsp_max_source_update"] == 232
    assert "source-U233" in result["parent"]["pfsp_boundary"]
    assert result["optimizer"] == {
        "initialization": "fresh",
        "logical_minibatch_size": 2048,
        "forward_microbatch_size": 512,
        "epochs": 3,
        "semantic_change": False,
    }
    assert len(result["focal"]["deck_ids"]) == 67
    assert not (tmp_path / runner.VERSION).exists()


def test_v6_allows_only_strict_pristine_restart(monkeypatch, tmp_path: Path) -> None:
    version_root = tmp_path / runner.VERSION
    paths = {name: version_root / name for name in ("artifact", "checkpoint", "tensorboard", "wandb")}
    for path in paths.values():
        path.mkdir(parents=True)
    (paths["artifact"] / "training_metrics.jsonl").write_text("", encoding="utf-8")
    (paths["artifact"] / "training_config.json").write_text(
        json.dumps({
            "start_update": runner.START_UPDATE,
            "parent_checkpoint_sha256": runner.sha256_file(runner.PARENT_CHECKPOINT),
            "focal_deck_ids": list(runner.FOCAL_DECK_IDS),
            "optimizer_initialization": "fresh",
        }),
        encoding="utf-8",
    )
    torch.save(
        {"update": runner.START_UPDATE},
        paths["checkpoint"] / f"update-{runner.START_UPDATE:06d}.pt",
    )
    monkeypatch.setattr(runner, "VERSION_ROOT", version_root)
    monkeypatch.setattr(runner, "VERSION", runner.VERSION)
    monkeypatch.setattr(runner, "_paths", lambda _version: paths)

    result = runner.readiness(
        version_root=version_root, allow_pristine_restart=True,
    )
    assert result["status"] == "READY_AWAITING_USER_LAUNCH"
