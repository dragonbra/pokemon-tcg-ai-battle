from __future__ import annotations

import importlib
from pathlib import Path


runner = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.training.run_v6")


def test_v6_binds_u4_and_final_core16_contract(tmp_path: Path) -> None:
    result = runner.readiness(version_root=tmp_path / runner.VERSION)
    assert result["parent"]["checkpoint_update"] == 4
    assert result["parent"]["optimizer"] == "fresh"
    assert result["parent"]["pfsp_state"] is None
    assert result["training"]["sampling"] == {
        "mode": "uniform_001_067", "pfsp": 0, "uniform": 256,
    }
    assert result["benchmark_v2"]["selected_meta_ids"] == list(range(14)) + [17, 27]
    assert result["benchmark_v2"]["games_per_meta"] == 128
    assert result["benchmark_v2"]["baseline_checkpoint_update"] == 4
