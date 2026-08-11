"""Build the machine-local `_ptcg_cuda` extension required by 0042.

The extension is ABI- and GPU-architecture-specific and therefore remains outside
Git.  This entry point makes its local construction and provenance reproducible.
It does not create the private official rule pack required by formal runs.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Sequence

import torch


ROOT = Path(__file__).resolve().parents[3]
BUILD_MANIFEST_SCHEMA = "0042_cuda_extension_build_manifest_v1"


def normalize_architecture(value: str | Sequence[int]) -> str:
    if isinstance(value, str):
        normalized = value.removeprefix("sm_")
        if not normalized.isascii() or not normalized.isdigit():
            raise ValueError(f"invalid CUDA architecture: {value!r}")
        major_minor = normalized
    else:
        if len(value) != 2:
            raise ValueError(f"invalid CUDA compute capability: {value!r}")
        major, minor = value
        if not isinstance(major, int) or not isinstance(minor, int) or major < 1 or minor < 0:
            raise ValueError(f"invalid CUDA compute capability: {value!r}")
        major_minor = f"{major}{minor}"
    if not re.fullmatch(r"[1-9][0-9]{1,2}", major_minor):
        raise ValueError(f"invalid CUDA architecture: {value!r}")
    return major_minor


def default_build_dir(repository_root: Path = ROOT) -> Path:
    # Kept stable because the formal 0042 runner audits this exact machine-local path.
    return repository_root / ".tmp/engine_cuda_benchmark/build_sm120_staged"


def build_commands(
    *, repository_root: Path, build_dir: Path, torch_cmake_dir: Path,
    architecture: str, jobs: int,
) -> tuple[list[str], list[str]]:
    if jobs < 1:
        raise ValueError("jobs must be positive")
    architecture = normalize_architecture(architecture)
    configure = [
        "cmake", "-S", str(repository_root / "engine_cuda"), "-B", str(build_dir),
        "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CUDA_ARCHITECTURES={architecture}",
        "-DPTCG_CUDA_BUILD_TORCH=ON", f"-DTorch_DIR={torch_cmake_dir}",
    ]
    compile_command = [
        "cmake", "--build", str(build_dir), "--target", "_ptcg_cuda",
        "--parallel", str(jobs),
    ]
    return configure, compile_command


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output(command: list[str], *, cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def build(*, architecture: str, build_dir: Path, jobs: int) -> dict[str, object]:
    torch_cmake_dir = Path(torch.__file__).resolve().parent / "share/cmake/Torch"
    if not torch_cmake_dir.is_dir():
        raise FileNotFoundError(f"Torch CMake package is unavailable: {torch_cmake_dir}")
    configure, compile_command = build_commands(
        repository_root=ROOT,
        build_dir=build_dir,
        torch_cmake_dir=torch_cmake_dir,
        architecture=architecture,
        jobs=jobs,
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(configure, cwd=ROOT, check=True)
    subprocess.run(compile_command, cwd=ROOT, check=True)
    extension = build_dir / "_ptcg_cuda.so"
    if not extension.is_file() or extension.stat().st_size == 0:
        raise RuntimeError(f"CUDA extension build did not produce {extension}")
    probe = (
        "import _ptcg_cuda; "
        "print(_ptcg_cuda.__file__); "
        "print(_ptcg_cuda.OFFICIAL_STATE_ABI_VERSION)"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(build_dir), str(ROOT / "engine_cuda/python"), environment.get("PYTHONPATH", "")]
    )
    probe_output = subprocess.check_output(
        [sys.executable, "-c", probe], cwd=ROOT, env=environment, text=True
    ).strip().splitlines()
    if not probe_output or Path(probe_output[0]).resolve() != extension.resolve():
        raise RuntimeError(f"Python imported a different _ptcg_cuda: {probe_output[:1]}")
    manifest = {
        "schema_version": BUILD_MANIFEST_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "repository_commit": _output(["git", "rev-parse", "HEAD"], cwd=ROOT),
        "repository_dirty": bool(_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_architecture": architecture,
        "cmake": _output(["cmake", "--version"], cwd=ROOT).splitlines()[0],
        "ninja": _output(["ninja", "--version"], cwd=ROOT),
        "extension": str(extension.relative_to(ROOT)),
        "extension_sha256": _sha256(extension),
        "official_state_abi_version": int(probe_output[1]),
        "configure_command": configure,
        "build_command": compile_command,
        "private_rule_pack_included": False,
    }
    _atomic_json(build_dir / "0042_build_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        help="CMake CUDA architecture such as 120 or 86; defaults to the active GPU",
    )
    parser.add_argument("--build-dir", type=Path, default=default_build_dir())
    parser.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    args = parser.parse_args()
    if args.architecture:
        architecture = normalize_architecture(args.architecture)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; pass --architecture only on a CUDA build host")
        architecture = normalize_architecture(torch.cuda.get_device_capability(0))
    manifest = build(
        architecture=architecture,
        build_dir=args.build_dir.resolve(),
        jobs=args.jobs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
