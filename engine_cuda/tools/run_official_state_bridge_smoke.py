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
        description="Build and run the strict official State-to-POD bridge smoke."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--output",
        type=Path,
        default=CUDA_ENGINE_ROOT / "artifacts" / "official_state_bridge_smoke.json",
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
    if not source.is_dir():
        raise SystemExit(f"official source directory does not exist: {source}")
    license_path = source / "LICENSES" / "LicenseRef-PTCG-ABC-Competition-Use-Only.txt"
    if not license_path.is_file():
        raise SystemExit("official competition-only license is missing")

    build_dir = CUDA_ENGINE_ROOT / "build" / "official_state_bridge_smoke"
    build_dir.mkdir(parents=True, exist_ok=True)
    executable = build_dir / "official_state_bridge_smoke"
    command = (
        "set -euo pipefail; "
        "g++ -std=c++20 -O3 -I/official "
        "-I/workspace/engine_cuda/include -I/workspace/engine_cuda/extractor "
        "/workspace/engine_cuda/extractor/official_state_bridge_smoke.cpp "
        "-ldl -o /build/official_state_bridge_smoke; "
        "/build/official_state_bridge_smoke"
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
        or result.get("state_abi") != 6
        or result.get("state_bytes") != 119_936
        or result.get("direct_fields") is not True
        or result.get("overflow_rejected") is not True
        or result.get("missing_rng_rejected") is not True
    ):
        raise RuntimeError("official state bridge smoke did not pass")

    evidence = {
        **result,
        "contract": "official_state_to_pod_bridge_v1",
        "official_source_license": "competition-use-only",
        "official_source_snapshot": "3aaeaa92c9eff5272d0c28582a1f64a4cdc2f772",
        "docker_image": args.image,
        "docker_image_id": docker_image_id(args.image),
        "bridge_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "extractor" / "official_state_bridge.h"
        ),
        "smoke_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "extractor" / "official_state_bridge_smoke.cpp"
        ),
        "state_pod_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "include" / "ptcg_cuda" / "official_state_pod.cuh"
        ),
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
