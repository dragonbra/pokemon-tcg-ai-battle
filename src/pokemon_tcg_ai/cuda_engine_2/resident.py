"""Fail-closed loader for the CUDA Engine 2.0 resident rollout backend."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Any

from .build import DEFAULT_BUILD_DIR, ENGINE_ROOT, REPOSITORY_ROOT
from .identity import CudaEngineIdentity


@dataclass(frozen=True, slots=True)
class ResidentBackend:
    """Only the device-resident production entry points exposed to 0044."""

    identity: CudaEngineIdentity
    resident_job: type
    resident_router: type
    run_resident_greedy_jobs: Any
    device_feature_encoder: str = "OfficialCudaEngine.encode_semantic0031_v2_lanes"
    feature_d2h_bytes: int = 0
    features_device_resident: bool = True


def load_resident_backend(
    *,
    rule_pack: Path,
    build_dir: Path = DEFAULT_BUILD_DIR,
) -> ResidentBackend:
    """Load Engine 2.0 after validating source, ABI, GPU and binary identity."""

    build_dir = Path(build_dir).resolve()
    extension = build_dir / "_ptcg_cuda.so"
    binary = build_dir / "ptcg_cuda_smoke"
    identity = CudaEngineIdentity.resolve(
        REPOSITORY_ROOT,
        rule_pack=Path(rule_pack).resolve(),
        binary=binary,
        extension=extension,
        require_gpu=True,
        require_extension=True,
    )
    for entry in (str(build_dir), str(ENGINE_ROOT / "python")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    resident = importlib.import_module("ptcg_cuda_engine.semantic0031_resident")
    router = importlib.import_module("ptcg_cuda_engine.semantic0031_router")
    return ResidentBackend(
        identity=identity,
        resident_job=resident.ResidentJob,
        resident_router=router.Semantic0031ResidentRouter,
        run_resident_greedy_jobs=resident.run_resident_greedy_jobs,
    )


__all__ = ["ResidentBackend", "load_resident_backend"]
