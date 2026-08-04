from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
DEFAULT_SOURCE = WORKSPACE_ROOT / "engine" / "source" / "ptcgProgram 22"
DEFAULT_IMAGE = "nvidia/cuda:13.0.0-devel-ubuntu24.04"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare official setup with the shared CPU/CUDA POD contract."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
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
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=CUDA_ENGINE_ROOT / "artifacts" / "official_setup_paired_10000.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def docker_image_id(image: str) -> str:
    completed = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def container_path(path: Path) -> str:
    relative = path.resolve().relative_to(WORKSPACE_ROOT.resolve())
    return "/workspace/" + relative.as_posix()


def main() -> None:
    args = parse_args()
    if args.seed_start < 0 or args.seed_count <= 0:
        raise SystemExit("seed start must be nonnegative and seed count must be positive")
    source = args.source.resolve()
    if not source.is_dir():
        raise SystemExit(f"official source directory does not exist: {source}")
    license_path = source / "LICENSES" / "LicenseRef-PTCG-ABC-Competition-Use-Only.txt"
    if not license_path.is_file():
        raise SystemExit("official competition-only license is missing")
    for deck in (args.deck0, args.deck1):
        if not deck.resolve().is_relative_to(WORKSPACE_ROOT.resolve()):
            raise SystemExit(f"deck must be inside the workspace: {deck}")

    build_dir = CUDA_ENGINE_ROOT / "build" / "official_setup_paired"
    build_dir.mkdir(parents=True, exist_ok=True)
    executable = build_dir / "official_setup_paired"
    if args.skip_build:
        if not executable.is_file():
            raise SystemExit(f"setup executable does not exist: {executable}")
        command = (
            "set -euo pipefail; "
            "/build/official_setup_paired "
            f"--deck0 {container_path(args.deck0)} "
            f"--deck1 {container_path(args.deck1)} "
            f"--seed-start {args.seed_start} --seed-count {args.seed_count}"
        )
    else:
        command = (
            "set -euo pipefail; "
            "g++ -std=c++20 -O3 -I/official -I/workspace/engine_cuda/include "
            "-I/workspace/engine_cuda/extractor "
            "/workspace/engine_cuda/extractor/official_setup_paired.cpp "
            "-o /build/official_setup_paired; "
            "/build/official_setup_paired "
            f"--deck0 {container_path(args.deck0)} "
            f"--deck1 {container_path(args.deck1)} "
            f"--seed-start {args.seed_start} --seed-count {args.seed_count}"
        )
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/bin/bash",
            "-v",
            f"{WORKSPACE_ROOT.resolve()}:/workspace:ro",
            "-v",
            f"{source}:/official:ro",
            "-v",
            f"{build_dir.resolve()}:/build:rw",
            args.image,
            "-lc",
            command,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result: dict[str, Any] = json.loads(completed.stdout)
    if (
        result.get("passed") is not True
        or result.get("full_state_bridge_states") != args.seed_count
    ):
        raise RuntimeError("paired setup executable did not report success")
    evidence = {
        **result,
        "contract": "official_setup_first_min_v1",
        "comparison": [
            "setup decisions and options",
            "deck/hand/prize/active card ID and instance order",
            "mulligan count and first player",
            "MT19937 624 words, cursor, and draw count",
        "final continuation ID",
        "strict official State to OfficialStatePod bridge",
        "setup decision count (not full-battle decision coverage)",
        ],
        "official_source_license": "competition-use-only",
        "docker_image": args.image,
        "docker_image_id": docker_image_id(args.image),
        "harness_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "extractor" / "official_setup_paired.cpp"
        ),
        "setup_pod_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "include" / "ptcg_cuda" / "official_setup_pod.cuh"
        ),
        "rng_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "include" / "ptcg_cuda" / "official_rng.cuh"
        ),
        "executable_sha256": sha256_file(executable),
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
