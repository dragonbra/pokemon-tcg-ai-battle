from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
PYTHON_ROOT = CUDA_ENGINE_ROOT / "python"
sys.path.insert(0, str(PYTHON_ROOT))

from ptcg_cuda_engine.official_ir import load_and_validate_official_ir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the shared official setup POD on CPU and CUDA."
    )
    parser.add_argument(
        "--nvcc-threads", type=int, default=8,
        help="Host threads available to nvcc compilation (default: 8).",
    )
    parser.add_argument(
        "--split-compile", type=int, default=1,
        help="Parallel device optimization jobs; raise only with enough RAM (default: 1).",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=(
            CUDA_ENGINE_ROOT
            / "generated"
            / "private"
            / "official_3aaeaa92"
            / "official_rules.json"
        ),
    )
    parser.add_argument(
        "--deck0",
        type=Path,
        default=CUDA_ENGINE_ROOT / "fixtures" / "decks" / "marnie_prize_control.csv",
    )
    parser.add_argument(
        "--deck1",
        type=Path,
        default=CUDA_ENGINE_ROOT / "fixtures" / "decks" / "alakazam_battle_cage.csv",
    )
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=10_000)
    parser.add_argument("--image", default="nvidia/cuda:13.0.0-devel-ubuntu24.04")
    parser.add_argument(
        "--output",
        type=Path,
        default=CUDA_ENGINE_ROOT / "artifacts" / "setup_pod_cuda_paired_10000.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    rows = ["0,0,0,0"]
    for card in payload["cards"]:
        card_id = int(card["id"])
        if card_id != len(rows):
            raise RuntimeError("official card IDs must be dense and one-based")
        flags = int(card["flags"])
        is_basic = int(bool(flags & (1 << 28)) and int(card["evolution_type"]) == 1)
        can_setup = int(bool(is_basic or flags & (1 << 7)))
        can_setup_active = int(bool(can_setup or flags & (1 << 8)))
        rows.append(f"{card_id},{is_basic},{can_setup},{can_setup_active}")
    path.write_text("\n".join(rows) + "\n", encoding="ascii")


def container_path(path: Path) -> str:
    relative = path.resolve().relative_to(WORKSPACE_ROOT.resolve())
    return "/workspace/" + relative.as_posix()


def main() -> None:
    args = parse_args()
    if args.nvcc_threads <= 0:
        raise SystemExit("nvcc threads must be positive")
    if args.split_compile <= 0:
        raise SystemExit("split compile must be positive")
    if args.seed_start < 0 or args.seed_count <= 0:
        raise SystemExit("seed start must be nonnegative and seed count must be positive")
    payload, summary = load_and_validate_official_ir(args.rules)
    build_dir = CUDA_ENGINE_ROOT / "build" / "setup_pod_cuda_paired"
    build_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = build_dir / "setup_metadata.csv"
    write_metadata(metadata_path, payload)
    executable = build_dir / "official_setup_pod"
    command = (
        "set -euo pipefail; "
        f"nvcc --threads {args.nvcc_threads} --split-compile {args.split_compile} "
        "-std=c++17 -O3 -arch=native "
        "-I/workspace/engine_cuda/include "
        "/workspace/engine_cuda/benchmarks/official_setup_pod.cu "
        "-o /build/official_setup_pod; "
        "/build/official_setup_pod "
        f"--deck0 {container_path(args.deck0)} "
        f"--deck1 {container_path(args.deck1)} "
        "--metadata /build/setup_metadata.csv "
        f"--seed-start {args.seed_start} --seed-count {args.seed_count}"
    )
    try:
        completed = subprocess.run(
            [
                "docker", "run", "--rm", "--gpus", "all",
                "--entrypoint", "/bin/bash",
                "-v", f"{WORKSPACE_ROOT.resolve()}:/workspace:ro",
                "-v", f"{build_dir.resolve()}:/build:rw",
                args.image, "-lc", command,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or str(error)
        raise RuntimeError(f"CUDA setup runner failed:\n{detail}") from error
    result: dict[str, Any] = json.loads(completed.stdout)
    if result.get("passed") is not True or result.get("mismatches") != 0:
        raise RuntimeError("CPU POD/CUDA setup comparison failed")
    evidence = {
        **result,
        "contract": "official_setup_first_min_v1",
        "rules_canonical_sha256": summary.canonical_sha256,
        "rules_file_sha256": sha256_file(args.rules),
        "benchmark_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "benchmarks" / "official_setup_pod.cu"
        ),
        "setup_pod_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "include" / "ptcg_cuda" / "official_setup_pod.cuh"
        ),
        "rng_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "include" / "ptcg_cuda" / "official_rng.cuh"
        ),
        "executable_sha256": sha256_file(executable),
        "metadata_sha256": sha256_file(metadata_path),
        "deck0_sha256": sha256_file(args.deck0),
        "deck1_sha256": sha256_file(args.deck1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**result, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
