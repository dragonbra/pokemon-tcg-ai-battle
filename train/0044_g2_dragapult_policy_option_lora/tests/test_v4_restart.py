from __future__ import annotations

import importlib
from pathlib import Path


runner = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.training.run_v4")


def test_v4_identity_is_exact() -> None:
    assert runner.VERSION == "V4_g2_uniform_0042_v1_lr_benchmark_v2"
    assert runner.START_UPDATE == 0
    assert runner.FOCAL_DECK_IDS == ("007",)
    assert runner.OPPONENT_DECK_IDS == tuple(f"{index:03d}" for index in range(1, 68))
    assert runner.OPPONENT_POLICY_IDS == ("Champion-G2",)
    assert runner.OPPONENT_SAMPLING_MODE == "uniform_001_067"
    assert runner.ACTOR_LEARNING_RATE_SCALE == 0.5


def test_v4_readiness_binds_g2_uniform_low_lr_and_u0_eval(tmp_path: Path) -> None:
    result = runner.readiness(version_root=tmp_path / runner.VERSION)
    assert result["status"] == "READY_AWAITING_USER_LAUNCH"
    assert result["initialization"]["policy_id"] == "Champion-G2"
    assert result["initialization"]["checkpoint_update"] == 407
    assert result["initialization"]["parent_training_checkpoint"] is None
    assert result["initialization"]["parent_pfsp_state"] is None
    assert result["opponent"]["sampling"] == {
        "mode": "uniform_001_067", "pfsp": 0, "uniform": 256,
    }
    assert result["optimizer"]["decoder_learning_rate"] == 5e-6
    assert result["optimizer"]["policy_adapter_learning_rate"] == 5e-6
    assert result["optimizer"]["allocation_learning_rate"] == 5e-6
    assert result["optimizer"]["option_lora_learning_rate"] == 1e-5
    assert result["benchmark_v2"]["baseline_checkpoint_update"] == 0
    assert not (tmp_path / runner.VERSION).exists()
