from __future__ import annotations

import importlib


runtime = importlib.import_module("train.0043_champion_league_rl.runtime")


def test_runtime_tree_and_both_policy_loaders_execute() -> None:
    assert len(runtime.audit_runtime_tree()) == 64
    anchor = runtime.forward_parity("Policy-0809", deck_id="001", gpu=False)
    champion = runtime.forward_parity("Champion-G1", deck_id="048", gpu=False)
    assert anchor.cpu_shape == (2, 6)
    assert champion.cpu_shape == (2,)


def test_synthetic_batch_excludes_source_identity() -> None:
    batch = runtime.synthetic_batch()
    assert "source_id" not in batch
    assert "team_name" not in batch
