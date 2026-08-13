from __future__ import annotations

import importlib
from pathlib import Path


runner = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.training.run_v5")


def test_v5_is_same_semantic_contract_and_audited_retry(tmp_path: Path) -> None:
    result = runner.readiness(version_root=tmp_path / runner.VERSION)
    assert runner.VERSION == "V5_g2_uniform_0042_v1_lr_benchmark_v2"
    assert result["retry_of"]["version"].startswith("V4_")
    assert result["retry_of"]["training_updates_completed"] == 0
    assert result["opponent"]["sampling"]["pfsp"] == 0
    assert result["optimizer"]["decoder_learning_rate"] == 5e-6
    assert result["benchmark_v2"]["baseline_checkpoint_update"] == 0
