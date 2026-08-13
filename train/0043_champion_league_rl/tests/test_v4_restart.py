from __future__ import annotations

import importlib
from pathlib import Path


runner = importlib.import_module("train.0043_champion_league_rl.training.run_v4")


def test_v4_restart_identity_is_exact() -> None:
    assert runner.VERSION == "V4_generalist_focal_001_067_u208_restart"
    assert runner.START_UPDATE == 208
    assert runner.PARENT_CHECKPOINT.name == "update-000208.pt"
    assert runner.FOCAL_DECK_IDS == tuple(f"{index:03d}" for index in range(1, 68))


def test_v4_readiness_binds_u208_without_mutation(tmp_path: Path) -> None:
    result = runner.readiness(version_root=tmp_path / runner.VERSION)
    assert result["status"] == "READY_AWAITING_USER_LAUNCH"
    assert result["parent"]["checkpoint_update"] == 208
    assert "discarded from failed PPO U209" in result["parent"]["pfsp_note"]
    assert result["optimizer"]["initialization"] == "fresh"
    assert len(result["focal"]["deck_ids"]) == 67
    assert not (tmp_path / runner.VERSION).exists()
