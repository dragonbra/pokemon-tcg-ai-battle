"""Identity-first construction of the 0043 official CUDA arena."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Any

from .build import DEFAULT_BUILD_DIR, ENGINE_ROOT
from .identity import CudaEngineIdentity


@dataclass(frozen=True, slots=True)
class OfficialCudaArena:
    engine: Any
    identity: CudaEngineIdentity
    batch_size: int


def create_official_cuda_arena(
    repository_root: Path, *, rule_pack: Path, batch_size: int,
    build_dir: Path = DEFAULT_BUILD_DIR, device_index: int = 0,
) -> OfficialCudaArena:
    """Verify every effective backend identity before allocating GPU state."""

    if batch_size <= 0:
        raise ValueError("CUDA arena batch_size must be positive")
    root = Path(repository_root).resolve()
    build = Path(build_dir).resolve()
    binary = build / "ptcg_cuda_smoke"
    extension = build / "_ptcg_cuda.so"
    identity = CudaEngineIdentity.resolve(
        root, rule_pack=Path(rule_pack).resolve(), binary=binary,
        extension=extension, require_gpu=True, require_extension=True,
    )
    # Shared engine infrastructure is allowed; all deck/policy choices have
    # already been resolved by the project-local adapter before arena use.
    for entry in (str(build), str(ENGINE_ROOT / "python")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    native = importlib.import_module("ptcg_cuda_engine.native")
    engine = native.create_official_engine(
        Path(rule_pack).read_bytes(), batch_size=batch_size,
        device_index=device_index,
    )
    return OfficialCudaArena(engine=engine, identity=identity, batch_size=batch_size)


__all__ = ["OfficialCudaArena", "create_official_cuda_arena"]
