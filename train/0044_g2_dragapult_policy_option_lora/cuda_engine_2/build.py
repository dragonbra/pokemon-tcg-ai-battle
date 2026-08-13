"""Low-memory SM120 build and runtime-smoke support for CUDA Engine 2.0."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
ENGINE_ROOT = REPOSITORY_ROOT / "engine_cuda_2_0"
DEFAULT_BUILD_DIR = ENGINE_ROOT / "build/native"


class CudaEngineBuildError(RuntimeError):
    pass


def _real_executable(name: str) -> Path:
    executable = shutil.which(name)
    if executable is None:
        raise CudaEngineBuildError(f"required executable is unavailable: {name}")
    return Path(executable).resolve()


def discover() -> dict[str, str]:
    import torch

    nvcc = _real_executable("nvcc")
    toolkit = nvcc.parent.parent
    target = toolkit / "targets/x86_64-linux"
    include = target / "include"
    if not (include / "cuda.h").is_file():
        raise CudaEngineBuildError(f"CUDA target headers are missing: {include}")
    torch_cmake = Path(torch.utils.cmake_prefix_path) / "Torch"
    if not (torch_cmake / "TorchConfig.cmake").is_file():
        raise CudaEngineBuildError(f"Torch CMake package is missing: {torch_cmake}")
    return {
        "nvcc": str(nvcc),
        "toolkit": str(toolkit),
        "target": str(target),
        "target_include": str(include),
        "torch_cmake": str(torch_cmake),
        "python": str(Path(sys.executable).resolve()),
        "torch_version": torch.__version__,
        "torch_cuda_version": str(torch.version.cuda),
    }


def configure_command(build_dir: Path = DEFAULT_BUILD_DIR) -> tuple[list[str], dict[str, str]]:
    found = discover()
    command = [
        "cmake", "-UCUDA_TOOLKIT_INCLUDE", "-S", str(ENGINE_ROOT),
        "-B", str(build_dir), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_FLAGS=-idirafter /usr/include",
        f"-DCMAKE_CUDA_COMPILER={found['nvcc']}",
        f"-DCUDAToolkit_ROOT={found['toolkit']}",
        f"-DCUDA_TOOLKIT_ROOT_DIR={found['toolkit']}",
        f"-DCUDA_TOOLKIT_TARGET_DIR={found['target']}",
        "-DCMAKE_CUDA_ARCHITECTURES=120",
        "-DPTCG_CUDA_BUILD_BENCHMARK=ON",
        "-DPTCG_CUDA_BUILD_TORCH=ON",
        f"-DTorch_DIR={found['torch_cmake']}",
        f"-DPython3_EXECUTABLE={found['python']}",
    ]
    environment = {
        "CUDA_INC_PATH": found["target_include"],
        "TORCH_CUDA_ARCH_LIST": "12.0",
    }
    return command, environment


def run_build(build_dir: Path = DEFAULT_BUILD_DIR) -> None:
    command, extra_environment = configure_command(build_dir)
    environment = os.environ.copy()
    environment.update(extra_environment)
    subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)
    # One diagnostic CUDA translation unit consumed about 30 GiB on CUDA
    # 12.8. Never build the default/all target from the 0044 entrypoint.
    for target in ("ptcg_cuda_smoke", "_ptcg_cuda"):
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--target", target, "--parallel", "1"],
            cwd=REPOSITORY_ROOT, check=True,
        )


def smoke_extension(rule_pack: Path, build_dir: Path = DEFAULT_BUILD_DIR) -> dict[str, Any]:
    import torch

    extension = build_dir / "_ptcg_cuda.so"
    if not extension.is_file():
        raise CudaEngineBuildError(f"CUDA extension is missing: {extension}")
    for entry in (str(build_dir), str(ENGINE_ROOT / "python")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    native = importlib.import_module("ptcg_cuda_engine.native")
    module = importlib.import_module("_ptcg_cuda")
    packed = rule_pack.read_bytes()
    engine = native.create_official_engine(packed, batch_size=2, device_index=0)
    state_bytes = int(module.OFFICIAL_STATE_BYTES)
    states = bytearray(2 * state_bytes)
    for lane in range(2):
        struct.pack_into("<I", states, lane * state_bytes, int(module.OFFICIAL_STATE_ABI_VERSION))
    engine.reset_states(torch.frombuffer(states, dtype=torch.uint8))
    engine.classify()
    statuses = engine.statuses()
    if statuses.device.type != "cuda" or tuple(statuses.shape) != (2,):
        raise CudaEngineBuildError("official status view is not a CUDA [2] tensor")
    torch.cuda.synchronize()
    return {
        "schema_version": "0044_cuda_engine_2_build_smoke_v1",
        "status": "PASS",
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "state_abi": int(module.OFFICIAL_STATE_ABI_VERSION),
        "rule_abi": int(module.OFFICIAL_RULE_ABI_VERSION),
        "state_bytes": state_bytes,
        "batch_size": engine.batch_size,
        "allocated_bytes": engine.allocated_bytes,
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "rule_pack_sha256": hashlib.sha256(packed).hexdigest(),
        "scope": "extension load, official rule upload, state reset/classify and CUDA views; not strength evidence",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("print", "build", "smoke"))
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--rule-pack", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "print":
        command, environment = configure_command(args.build_dir)
        payload: dict[str, Any] = {
            "environment": environment,
            "configure": command,
            "build_parallel": 1,
        }
    elif args.command == "build":
        run_build(args.build_dir)
        payload = {"status": "PASS", "build_dir": str(args.build_dir), "build_parallel": 1}
    else:
        if args.rule_pack is None:
            parser.error("smoke requires --rule-pack")
        payload = smoke_extension(args.rule_pack.resolve(), args.build_dir.resolve())
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded)
        temporary.replace(args.output)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
