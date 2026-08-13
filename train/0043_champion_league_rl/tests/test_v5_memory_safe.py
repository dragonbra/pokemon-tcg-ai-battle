from __future__ import annotations

import importlib
from pathlib import Path

r = importlib.import_module("train.0043_champion_league_rl.training.run_v5")
shared = importlib.import_module("train.0043_champion_league_rl.training.run_v1")


def test_micro512_preserves_logical_ppo_contract() -> None:
    c = shared._config(forward_microbatch_size=512)
    assert c.batch_size == 2048
    assert c.forward_microbatch_size == 512
    assert c.epochs == 3
    assert c.gradient_accumulation == 1


def test_v5_readiness_binds_u208_and_clean_pfsp(tmp_path: Path) -> None:
    d = r.readiness(version_root=tmp_path / r.VERSION)
    assert d["parent"]["checkpoint_update"] == 208
    assert "V2 final committed state" in d["parent"]["pfsp_boundary"]
    assert d["optimizer"]["logical_minibatch_size"] == 2048
    assert d["optimizer"]["forward_microbatch_size"] == 512
    assert d["optimizer"]["semantic_change"] is False
