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
DEFAULT_RULES = (
    CUDA_ENGINE_ROOT
    / "generated"
    / "private"
    / "official_3aaeaa92"
    / "official_rules.bin"
)
DEFAULT_FIXTURE = (
    CUDA_ENGINE_ROOT
    / "generated"
    / "private"
    / "official_3aaeaa92"
    / "official_seeded_reset_marnie_alakazam_s1_10.bin"
)
DEFAULT_IMAGE = "nvidia/cuda:13.0.0-devel-ubuntu24.04"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run real official setup and the independent CPU POD seeded reset, "
            "then freeze the official states in a private paired fixture."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument(
        "--deck0",
        type=Path,
        default=(
            WORKSPACE_ROOT
            / "bc_models"
            / "agent_marnie_prize_control_v4_3121746f_20260728"
            / "deck.csv"
        ),
    )
    parser.add_argument(
        "--deck1",
        type=Path,
        default=(
            WORKSPACE_ROOT
            / "bc_models"
            / "agent_yushin_idonly_bc_v1_20260723"
            / "deck.csv"
        ),
    )
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=CUDA_ENGINE_ROOT / "artifacts" / "official_seeded_reset_oracle.json",
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
    rules = args.rules.resolve()
    fixture = args.fixture.resolve()
    decks = [args.deck0.resolve(), args.deck1.resolve()]
    if not source.is_dir():
        raise SystemExit(f"official source directory does not exist: {source}")
    license_path = source / "LICENSES" / "LicenseRef-PTCG-ABC-Competition-Use-Only.txt"
    if not license_path.is_file():
        raise SystemExit("official competition-only license is missing")
    for path in [rules, *decks]:
        if not path.is_file():
            raise SystemExit(f"required input does not exist: {path}")
        if not path.is_relative_to(WORKSPACE_ROOT.resolve()):
            raise SystemExit(f"input must be inside the workspace: {path}")
    private_root = (CUDA_ENGINE_ROOT / "generated" / "private").resolve()
    if not fixture.is_relative_to(private_root):
        raise SystemExit(f"official state fixture must remain private: {fixture}")

    build_dir = CUDA_ENGINE_ROOT / "build" / "official_seeded_reset_oracle"
    build_dir.mkdir(parents=True, exist_ok=True)
    executable = build_dir / "official_seeded_reset_oracle"
    built_fixture = build_dir / "official_seeded_reset_fixture.bin"
    run_command = (
        "/build/official_seeded_reset_oracle "
        f"{container_path(rules)} {container_path(decks[0])} "
        f"{container_path(decks[1])} {args.seed_start} {args.seed_count} "
        "/build/official_seeded_reset_fixture.bin"
    )
    if args.skip_build:
        if not executable.is_file():
            raise SystemExit(f"seeded-reset oracle executable does not exist: {executable}")
        command = "set -euo pipefail; " + run_command
    else:
        command = (
            "set -euo pipefail; "
            "g++ -std=c++20 -O3 -I/official -I/workspace/engine_cuda/include "
            "-I/workspace/engine_cuda/extractor "
            "/workspace/engine_cuda/extractor/official_seeded_reset_oracle.cpp "
            "-o /build/official_seeded_reset_oracle; "
            + run_command
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
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "official seeded-reset oracle failed "
            f"(exit {completed.returncode})\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    result: dict[str, Any] = json.loads(completed.stdout)
    if (
        result.get("passed") is not True
        or result.get("seed_start") != args.seed_start
        or result.get("seed_count") != args.seed_count
        or result.get("state_mismatches") != 0
        or result.get("status_mismatches") != 0
        or not built_fixture.is_file()
    ):
        raise RuntimeError("official CPU/CPU POD seeded-reset oracle did not pass")
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_bytes(built_fixture.read_bytes())

    evidence = {
        **result,
        "contract": "official_cpu_pod_seeded_reset_oracle_v1",
        "setup_policy": "first_min_v1",
        "comparison_level": "all OfficialStatePod bytes and boundary status",
        "official_source_license": "competition-use-only",
        "official_source_snapshot": "3aaeaa92c9eff5272d0c28582a1f64a4cdc2f772",
        "docker_image": args.image,
        "docker_image_id": docker_image_id(args.image),
        "harness_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "extractor" / "official_seeded_reset_oracle.cpp"
        ),
        "seeded_setup_pod_sha256": sha256_file(
            CUDA_ENGINE_ROOT
            / "include"
            / "ptcg_cuda"
            / "official_seeded_setup_pod.cuh"
        ),
        "rule_pack_sha256": sha256_file(rules),
        "deck0_sha256": sha256_file(decks[0]),
        "deck1_sha256": sha256_file(decks[1]),
        "private_fixture_sha256": sha256_file(fixture),
        "executable_sha256": sha256_file(executable),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**result, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
