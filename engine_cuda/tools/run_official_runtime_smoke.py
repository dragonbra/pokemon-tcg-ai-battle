from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
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
    / "official_attack_resume_fixture.bin"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and run the OfficialStatePod CUDA runtime arena smoke."
    )
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--skip-resume-fixture",
        action="store_true",
        help=(
            "Run only the resident arena/classification/fail-closed gate. "
            "The historical resume fixture remains a separate strict gate."
        ),
    )
    parser.add_argument("--image", default="nvidia/cuda:13.0.0-devel-ubuntu24.04")
    parser.add_argument(
        "--nvcc-threads",
        type=int,
        default=8,
        help="Host threads available to nvcc compilation (default: 8).",
    )
    parser.add_argument(
        "--split-compile", type=int, default=1,
        help="Parallel device optimization jobs; raise only with enough RAM (default: 1).",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=CUDA_ENGINE_ROOT / "artifacts" / "official_runtime_smoke.json",
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
    if args.batch <= 0:
        raise SystemExit("batch must be positive")
    if args.nvcc_threads <= 0:
        raise SystemExit("nvcc threads must be positive")
    if args.split_compile <= 0:
        raise SystemExit("split compile must be positive")
    rules = args.rules.resolve()
    fixture = args.fixture.resolve()
    if not rules.is_file():
        raise SystemExit(f"private rule pack does not exist: {rules}")
    if not args.skip_resume_fixture and not fixture.is_file():
        raise SystemExit(f"private resume fixture does not exist: {fixture}")
    rules_relative = rules.relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    fixture_relative = (
        fixture.relative_to(WORKSPACE_ROOT.resolve()).as_posix()
        if not args.skip_resume_fixture
        else None
    )
    build_dir = CUDA_ENGINE_ROOT / "build" / "official_runtime_smoke"
    build_dir.mkdir(parents=True, exist_ok=True)
    executable = build_dir / "official_runtime_smoke"
    if args.skip_build:
        if not executable.is_file():
            raise SystemExit(f"runtime executable does not exist: {executable}")
        command = f"/build/official_runtime_smoke /workspace/{rules_relative} {args.batch}"
        if fixture_relative is not None:
            command += f" /workspace/{fixture_relative}"
    else:
        command = (
            f"nvcc --threads {args.nvcc_threads} --split-compile {args.split_compile} "
            "-std=c++17 -O0 -arch=sm_86 "
            "-I/workspace/engine_cuda/include "
            "/workspace/engine_cuda/src/official_engine_kernels.cu "
            "/workspace/engine_cuda/benchmarks/official_runtime_smoke.cu "
            "-o /build/official_runtime_smoke; "
            f"/build/official_runtime_smoke /workspace/{rules_relative} {args.batch}"
        )
        if fixture_relative is not None:
            command += f" /workspace/{fixture_relative}"
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--entrypoint",
            "/bin/bash",
            "-v",
            f"{WORKSPACE_ROOT.resolve()}:/workspace:ro",
            "-v",
            f"{build_dir.resolve()}:/build:rw",
            args.image,
            "-lc",
            "set -euo pipefail; " + command,
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "official runtime smoke command failed "
            f"(exit {completed.returncode})\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    result: dict[str, Any] = json.loads(completed.stdout)
    expected_resume_present = not args.skip_resume_fixture
    if (
        result.get("passed") is not True
        or result.get("state_abi") != 6
        or result.get("state_bytes") != 119_936
        or result.get("action_bytes") != 272
        or result.get("device_stack_bytes", 0) < 32 * 1024
        or result.get("batch") != args.batch
        or result.get("classification_mismatches") != 0
        or result.get("fail_closed_mismatches") != 0
        or result.get("resume_fixture_present") is not expected_resume_present
        or result.get("resume_state_mismatches") != 0
    ):
        raise RuntimeError("official runtime arena smoke did not pass")
    scope = [
        "official_state_device_residency",
        "private_rule_pack_device_residency",
        "device_action_batch",
        "device_stack_limit",
        "device_flow_classification",
        "fail_closed_invalid_action",
    ]
    if args.skip_resume_fixture:
        scope.append("historical_resume_fixture_explicitly_quarantined")
    else:
        scope.extend(
            [
                "official_oracle_fixture_gpu_resume",
                "full_state_post_action_byte_parity",
            ]
        )
    evidence = {
        **result,
        "contract": "official_runtime_arena_smoke_v1",
        "scope": scope,
        "limitation": (
            "This gate does not claim complete main-phase card/skill/branch "
            "coverage or full official battle replay."
        ),
        "docker_image": args.image,
        "docker_image_id": docker_image_id(args.image),
        "runtime_source_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "src" / "official_engine_kernels.cu"
        ),
        "runtime_header_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "include" / "ptcg_cuda" / "official_runtime.h"
        ),
        "rule_pack_sha256": sha256_file(rules),
        "private_resume_fixture_sha256": (
            None if args.skip_resume_fixture else sha256_file(fixture)
        ),
        "executable_sha256": sha256_file(executable),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**result, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
