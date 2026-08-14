from __future__ import annotations

import importlib
from pathlib import Path


def test_v9_restarts_v8_u20_as_local_u0_deck066(tmp_path: Path) -> None:
    runner = importlib.import_module(
        "train.0044_g2_dragapult_policy_option_lora.training.run_v9"
    )
    result = runner.readiness(version_root=tmp_path / runner.VERSION)
    assert result["status"] == "READY_AWAITING_USER_LAUNCH"
    assert result["version"] == "V9_u20_deck066_core16_benchmark_v2"
    assert result["parent"] == {
        "version": "V8_u33_deck003_core16_benchmark_v2",
        "source_checkpoint_update": 20,
        "local_start_update": 0,
        "checkpoint_sha256": "1ebe9f66df8adbcf38d922c39cd8a3971ca6459f823551c54695270a5086e7ac",
        "optimizer": "fresh",
        "pfsp_state": None,
    }
    assert result["training"]["focal_deck_ids"] == ["066"]
    assert result["training"]["sampling"] == {
        "mode": "uniform_001_067", "pfsp": 0, "uniform": 256,
    }
    assert result["benchmark_v2"]["baseline_checkpoint_update"] == 0
    assert result["benchmark_v2"]["focal_deck_id"] == "066"
    assert result["benchmark_v2"]["selected_meta_ids"] == list(range(14)) + [17, 27]
    assert result["benchmark_v2"]["games"] == 2048
