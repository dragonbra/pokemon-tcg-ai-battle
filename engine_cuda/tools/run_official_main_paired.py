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
    / "official_main_fixture.bin"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MainSelect CPU POD/CUDA paired replay."
    )
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
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
        default=CUDA_ENGINE_ROOT / "artifacts" / "official_main_paired.json",
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
    fixture = args.fixture.resolve()
    if not rules.is_file():
        raise SystemExit(f"private rule pack does not exist: {rules}")
    if not fixture.is_file():
        raise SystemExit(f"private main fixture does not exist: {fixture}")
    rules_relative = rules.relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    fixture_relative = fixture.relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    build_dir = CUDA_ENGINE_ROOT / "build" / "official_main_paired"
    build_dir.mkdir(parents=True, exist_ok=True)
    executable = build_dir / "official_main_smoke"
    if args.skip_build:
        if not executable.is_file():
            raise SystemExit(f"benchmark executable does not exist: {executable}")
        command = (
            f"/build/official_main_smoke /workspace/{rules_relative} "
            f"/workspace/{fixture_relative}"
        )
    else:
        command = (
            f"nvcc --threads {args.nvcc_threads} --split-compile {args.split_compile} "
            "-std=c++17 -O0 -arch=sm_86 "
            "-I/workspace/engine_cuda/include "
            "/workspace/engine_cuda/src/official_engine_kernels.cu "
            "/workspace/engine_cuda/benchmarks/official_main_smoke.cu "
            "-o /build/official_main_smoke; "
            f"/build/official_main_smoke /workspace/{rules_relative} "
            f"/workspace/{fixture_relative}"
        )
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
            "official main paired command failed "
            f"(exit {completed.returncode})\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    result: dict[str, Any] = json.loads(completed.stdout)
    if (
        result.get("passed") is not True
        or result.get("state_abi") != 6
        or result.get("state_bytes") != 119_936
        or result.get("scenario_count") != 35
        or result.get("status_mismatches") != 0
        or result.get("fixture_mismatches") != 0
        or result.get("cuda_mismatches") != 0
        or result.get("cpu_digest") != result.get("gpu_digest")
    ):
        raise RuntimeError("official main paired replay did not pass")
    evidence = {
        **result,
        "contract": "official_main_paired_v1",
        "scope": [
            "main_option_order",
            "main_attack_dispatch",
            "main_end_dispatch",
            "main_play_basic_dispatch",
            "main_play_stadium_dispatch",
            "main_play_item_dispatch",
            "main_play_supporter_dispatch",
            "main_play_item_selection_dispatch",
            "main_play_item_selection_resume",
            "main_attach_energy_dispatch",
            "main_evolve_dispatch",
            "main_ability_dispatch",
            "main_discard_dispatch",
            "main_retreat_energy_dispatch",
            "main_retreat_switch_dispatch",
            "main_retreat_complete_dispatch",
            "main_attach_tool_dispatch",
            "main_retreat_zero_cost_dispatch",
            "main_retreat_multi_energy_dispatch",
            "main_retreat_negative_legality",
            "main_rainbow_dna_evolution",
            "main_ability_selection_dispatch",
            "main_ability_optional_resume",
            "main_ability_negative_legality",
            "next_main_decision",
            "turn_action_count_step_semantics",
            "full_state_cpu_pod_cuda_byte_parity",
        ],
        "comparison_level": "CPU POD fixture replay versus production CUDA arena",
        "docker_image": args.image,
        "docker_image_id": docker_image_id(args.image),
        "benchmark_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "benchmarks" / "official_main_smoke.cu"
        ),
        "runtime_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "src" / "official_engine_kernels.cu"
        ),
        "private_fixture_sha256": sha256_file(fixture),
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
