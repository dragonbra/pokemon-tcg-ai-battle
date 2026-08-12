from __future__ import annotations

import importlib
from pathlib import Path

import pytest


module = importlib.import_module("train.0043_champion_league_rl.cuda_engine_2.identity")
ROOT = Path(__file__).resolve().parents[3]
RULES = ROOT / ".tmp/cuda_0032_rules/official_rules.bin"


def test_resolves_pinned_cuda_2_source_abi_rules_and_gpu() -> None:
    identity = module.CudaEngineIdentity.resolve(ROOT, rule_pack=RULES)
    assert identity.status == "PASS"
    assert identity.engine_label == "cuda_engine_2_0"
    assert identity.official_state_abi == 7
    assert identity.compute_capability == (12, 0)


def test_rejects_wrong_rule_pack(tmp_path: Path) -> None:
    changed = tmp_path / "official_rules.bin"
    changed.write_bytes(RULES.read_bytes() + b"x")
    with pytest.raises(module.CudaEngineIdentityError, match="rule-pack identity"):
        module.CudaEngineIdentity.resolve(ROOT, rule_pack=changed)


def test_extension_is_explicit_hard_gate() -> None:
    with pytest.raises(module.CudaEngineIdentityError, match="extension is required"):
        module.CudaEngineIdentity.resolve(ROOT, rule_pack=RULES, require_extension=True)
