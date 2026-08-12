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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official CPU vs CPU POD target/condition micro-fixtures."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--image", default="nvidia/cuda:13.0.0-devel-ubuntu24.04")
    parser.add_argument(
        "--output",
        type=Path,
        default=CUDA_ENGINE_ROOT / "artifacts" / "official_semantic_micro_fixture.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def docker_image_id(image: str) -> str:
    completed = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    rules = args.rules.resolve()
    if not source.is_dir():
        raise SystemExit(f"official source directory does not exist: {source}")
    if not rules.is_file():
        raise SystemExit(f"private rule pack does not exist: {rules}")
    license_path = source / "LICENSES" / "LicenseRef-PTCG-ABC-Competition-Use-Only.txt"
    if not license_path.is_file():
        raise SystemExit("official competition-only license is missing")
    rules_relative = rules.relative_to(WORKSPACE_ROOT.resolve()).as_posix()

    build_dir = CUDA_ENGINE_ROOT / "build" / "official_semantic_micro_fixture"
    build_dir.mkdir(parents=True, exist_ok=True)
    executable = build_dir / "official_semantic_micro_fixture"
    command = (
        "set -euo pipefail; "
        "g++ -std=c++20 -O3 -I/official -I/workspace/engine_cuda_2_0/include "
        "/workspace/engine_cuda_2_0/extractor/official_semantic_micro_fixture.cpp "
        "-ldl -o /build/official_semantic_micro_fixture; "
        f"/build/official_semantic_micro_fixture /workspace/{rules_relative}"
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
    expected_effects = [172, 174, 186, 207, 222, 234, 238, 244]
    if (
        result.get("passed") is not True
        or result.get("checks", 0) < 30
        or result.get("paired_effects") != expected_effects
        or result.get("paired_damage_pipeline") is not True
        or result.get("paired_move_primitives") is not True
        or result.get("paired_delay_effect") is not True
    ):
        raise RuntimeError("official semantic micro-fixture did not pass")
    evidence = {
        **result,
        "contract": "official_semantic_micro_fixture_v2",
        "official_source_license": "competition-use-only",
        "docker_image": args.image,
        "docker_image_id": docker_image_id(args.image),
        "harness_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "extractor" / "official_semantic_micro_fixture.cpp"
        ),
        "targets_pod_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "include" / "ptcg_cuda" / "official_targets_pod.cuh"
        ),
        "conditions_pod_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "include" / "ptcg_cuda" / "official_conditions_pod.cuh"
        ),
        "effects_pod_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "include" / "ptcg_cuda" / "official_effects_pod.cuh"
        ),
        "continual_effects_pod_sha256": sha256_file(
            CUDA_ENGINE_ROOT
            / "include"
            / "ptcg_cuda"
            / "official_continual_effects_pod.cuh"
        ),
        "rule_pack_sha256": sha256_file(rules),
        "executable_sha256": sha256_file(executable),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**result, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
