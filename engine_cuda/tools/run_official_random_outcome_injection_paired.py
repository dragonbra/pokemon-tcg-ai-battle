from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


CUDA_ROOT = Path(__file__).resolve().parents[1]
ROOT = CUDA_ROOT.parent
DEFAULT_SOURCE = ROOT / "engine" / "source" / "ptcgProgram 22"
DEFAULT_RULES = CUDA_ROOT / "generated/private/official_3aaeaa92/official_rules.bin"
DEFAULT_IMAGE = "nvidia/cuda:13.0.0-devel-ubuntu24.04"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--deck0", type=Path, default=CUDA_ROOT / "fixtures/decks/marnie_prize_control.csv")
    parser.add_argument("--deck1", type=Path, default=CUDA_ROOT / "fixtures/decks/alakazam_battle_cage.csv")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    for label, path in (
        ("official source", args.source),
        ("rule pack", args.rules),
        ("deck0", args.deck0),
        ("deck1", args.deck1),
    ):
        if not path.resolve().exists():
            raise SystemExit(f"missing {label}: {path.resolve()}")
    build = CUDA_ROOT / "build" / "official_random_outcome_injection_paired"
    build.mkdir(parents=True, exist_ok=True)
    executable = build / "official_random_outcome_injection_paired"
    container_command = (
        "set -euo pipefail; nvcc -std=c++20 -O3 "
        "-I/official -I/workspace/engine_cuda/include -I/workspace/engine_cuda/extractor "
        "-I/workspace/engine_cuda/benchmarks "
        "/workspace/engine_cuda/benchmarks/official_random_outcome_injection_paired.cu "
        "/workspace/engine_cuda/src/official_engine_kernels.cu "
        "-o /build/official_random_outcome_injection_paired; "
        f"/build/official_random_outcome_injection_paired /workspace/{args.rules.resolve().relative_to(ROOT)} "
        f"/workspace/{args.deck0.resolve().relative_to(ROOT)} "
        f"/workspace/{args.deck1.resolve().relative_to(ROOT)}"
    )
    docker = shutil.which("docker")
    if args.skip_build:
        if not executable.is_file():
            raise SystemExit(f"missing cached executable: {executable}")
        completed = subprocess.run([str(executable), str(args.rules.resolve()),
                                    str(args.deck0.resolve()), str(args.deck1.resolve())],
                                   capture_output=True, text=True)
        build_backend = "cached_local_binary"
    elif docker:
        completed = subprocess.run([docker, "run", "--rm", "--gpus", "all", "--entrypoint", "/bin/bash",
                                    "-v", f"{ROOT}:/workspace:ro", "-v", f"{args.source.resolve()}:/official:ro",
                                    "-v", f"{build.resolve()}:/build:rw", args.image, "-lc", container_command],
                                   capture_output=True, text=True)
        build_backend = "docker"
    else:
        nvcc = os.environ.get("NVCC", "/home/cyd/.local/cuda-12.8/bin/nvcc")
        subprocess.run([
            nvcc, "--threads", "4", "--split-compile", "1", "-std=c++20", "-O0", "-arch=sm_86",
            "-I", str(args.source.resolve()), "-I", str(CUDA_ROOT / "include"),
            "-I", str(CUDA_ROOT / "extractor"), "-I", str(CUDA_ROOT / "benchmarks"),
            str(CUDA_ROOT / "benchmarks/official_random_outcome_injection_paired.cu"),
            str(CUDA_ROOT / "src/official_engine_kernels.cu"), "-o", str(executable),
        ], check=True)
        completed = subprocess.run([str(executable), str(args.rules.resolve()),
                                    str(args.deck0.resolve()), str(args.deck1.resolve())],
                                   capture_output=True, text=True)
        build_backend = "local_nvcc"
    if completed.returncode != 0:
        raise RuntimeError(
            "explicit random-outcome paired benchmark failed "
            f"(exit {completed.returncode})\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    payload = json.loads(completed.stdout)
    header = CUDA_ROOT / "include/ptcg_cuda/testing/random_outcome_fixture.cuh"
    source = CUDA_ROOT / "benchmarks/official_random_outcome_injection_paired.cu"
    result = {"gate": "D", "status": "INCOMPLETE" if not all(payload.get(key) for key in (
                  "coin_rule_execution", "random_target_rule_execution", "random_effect_rule_execution")) else "PASS",
              **payload, "test_only": True, "production_hot_path_changed": False,
              "build_backend": build_backend,
              "header_sha256": sha256(header), "source_sha256": sha256(source),
              "rule_pack_sha256": sha256(args.rules),
              "executable_sha256": sha256(executable),
              "excluded_from_production_proof": "no production file includes ptcg_cuda/testing/random_outcome_fixture.cuh"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if payload.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
