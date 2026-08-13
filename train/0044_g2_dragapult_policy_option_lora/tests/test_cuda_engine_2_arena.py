from __future__ import annotations

import importlib
from pathlib import Path

import pytest


module = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.cuda_engine_2.arena")
ROOT = Path(__file__).resolve().parents[3]
RULES = ROOT / ".tmp/cuda_0032_rules/official_rules.bin"


def test_rejects_invalid_batch_before_gpu_allocation() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        module.create_official_cuda_arena(ROOT, rule_pack=RULES, batch_size=0)


def test_identity_first_official_arena_smoke() -> None:
    arena = module.create_official_cuda_arena(ROOT, rule_pack=RULES, batch_size=2)
    assert arena.identity.status == "PASS"
    assert arena.identity.engine_label == "cuda_engine_2_0"
    assert arena.engine.batch_size == 2
