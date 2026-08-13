from __future__ import annotations

import importlib
from pathlib import Path

import pytest


module = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.training.config")
ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "experiments/0044_g2_dragapult_policy_option_lora/active_training_config.json"


def test_approved_training_config_regression_passes() -> None:
    audit = module.audit_training_config(CONFIG)
    assert audit.status == "PASS"
    assert audit.games_per_update == 256
    assert audit.ppo_epochs == 3
    assert audit.ppo_minibatch_size == 2048


def test_mutated_training_config_fails(tmp_path) -> None:
    changed = tmp_path / "config.json"
    changed.write_text(CONFIG.read_text().replace('"games_per_update": 256', '"games_per_update": 512'))
    with pytest.raises(ValueError, match="SHA mismatch"):
        module.audit_training_config(changed)
