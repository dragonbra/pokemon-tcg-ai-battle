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
DEFAULT_RULES = (
    CUDA_ENGINE_ROOT
    / "generated"
    / "private"
    / "official_3aaeaa92"
    / "official_rules.bin"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official turn/refresh CPU POD/CUDA paired smokes."
    )
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--image", default="nvidia/cuda:13.0.0-devel-ubuntu24.04")
    parser.add_argument(
        "--nvcc-threads", type=int, default=8,
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
        default=CUDA_ENGINE_ROOT / "artifacts" / "official_turn_flow_paired.json",
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
    if args.nvcc_threads <= 0:
        raise SystemExit("nvcc threads must be positive")
    if args.split_compile <= 0:
        raise SystemExit("split compile must be positive")
    rules = args.rules.resolve()
    if not rules.is_file():
        raise SystemExit(f"private rule pack does not exist: {rules}")
    rules_relative = rules.relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    build_dir = CUDA_ENGINE_ROOT / "build" / "official_turn_flow_paired"
    build_dir.mkdir(parents=True, exist_ok=True)
    executable = build_dir / "official_turn_flow_smoke"
    if args.skip_build:
        if not executable.is_file():
            raise SystemExit(f"benchmark executable does not exist: {executable}")
        command = f"/build/official_turn_flow_smoke /workspace/{rules_relative}"
    else:
        command = (
            f"nvcc --threads {args.nvcc_threads} --split-compile {args.split_compile} "
            "-std=c++17 -O0 -arch=sm_86 "
            "-I/workspace/engine_cuda/include "
            "/workspace/engine_cuda/src/official_engine_kernels.cu "
            "/workspace/engine_cuda/benchmarks/official_turn_flow_smoke.cu "
            "-o /build/official_turn_flow_smoke; "
            f"/build/official_turn_flow_smoke /workspace/{rules_relative}"
        )
    command = "set -euo pipefail; " + command
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
            command,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, file=sys.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        completed.check_returncode()
    result: dict[str, Any] = json.loads(completed.stdout)
    if (
        result.get("passed") is not True
        or result.get("state_abi") != 6
        or result.get("state_bytes") != 119_936
        or result.get("device_stack_bytes", 0) < 32 * 1024
        or result.get("cpu_digest") != result.get("gpu_digest")
        or result.get("state_mismatches") != 0
        or result.get("status_mismatches") != 0
        or result.get("next_turn") != 2
        or result.get("drawn_card") != 41
        or result.get("expired_energy_trash") != 1
        or result.get("bench_after_overflow") != 5
        or result.get("tool_after_overflow") != 1
        or result.get("rocket_energy_after_cleanup") != 0
        or result.get("deck_out_result") != 1
        or result.get("deck_out_reason") != 2
        or result.get("errors") != [0, 0, 0]
    ):
        raise RuntimeError("official turn-flow paired smoke did not pass")
    evidence = {
        **result,
        "contract": "official_turn_flow_paired_v1",
        "scope": [
            "turn_end_and_turn_start_state_rollover",
            "turn_history_rollover",
            "turn_end_attachment_expiry",
            "pokemon_checkup_outer_flow",
            "bench_capacity_overflow",
            "tool_capacity_overflow",
            "only_team_rocket_energy_cleanup",
            "deck_out_terminal_logic",
            "production_runtime_arena_action_dispatch",
        ],
        "comparison_level": (
            "CPU POD expected state versus production CUDA runtime arena actual "
            "state; official CPU oracle is separate"
        ),
        "docker_image": args.image,
        "docker_image_id": docker_image_id(args.image),
        "benchmark_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "benchmarks" / "official_turn_flow_smoke.cu"
        ),
        "turn_flow_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "include" / "ptcg_cuda" / "official_turn_flow_pod.cuh"
        ),
        "rule_pack_sha256": sha256_file(rules),
        "executable_sha256": sha256_file(executable),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**result, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
