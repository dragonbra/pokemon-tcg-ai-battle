from __future__ import annotations

import importlib
from pathlib import Path


module = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.cuda_engine_2.build")


def test_build_is_sm120_and_strictly_serial() -> None:
    command, environment = module.configure_command()
    assert "-DCMAKE_CUDA_ARCHITECTURES=120" in command
    assert "-DPTCG_CUDA_BUILD_TORCH=ON" in command
    assert environment["TORCH_CUDA_ARCH_LIST"] == "12.0"
    assert "CUDA_INC_PATH" in environment


def test_default_build_excludes_heavy_all_target() -> None:
    source = Path(module.__file__).read_text()
    assert 'for target in ("ptcg_cuda_smoke", "_ptcg_cuda")' in source
    assert '"--parallel", "1"' in source
