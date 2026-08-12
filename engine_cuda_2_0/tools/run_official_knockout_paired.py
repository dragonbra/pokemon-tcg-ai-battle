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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official KO/Prize CPU POD/CUDA paired smoke."
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
        default=CUDA_ENGINE_ROOT / "artifacts" / "official_knockout_paired.json",
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
    build_dir = CUDA_ENGINE_ROOT / "build" / "official_knockout_paired"
    build_dir.mkdir(parents=True, exist_ok=True)
    executable = build_dir / "official_knockout_smoke"
    if args.skip_build:
        if not executable.is_file():
            raise SystemExit(f"benchmark executable does not exist: {executable}")
        command = f"/build/official_knockout_smoke /workspace/{rules_relative}"
    else:
        command = (
            f"nvcc --threads {args.nvcc_threads} --split-compile {args.split_compile} "
            "-std=c++17 -O0 -arch=sm_86 "
            "-I/workspace/engine_cuda_2_0/include "
            "/workspace/engine_cuda_2_0/benchmarks/official_knockout_smoke.cu "
            "-o /build/official_knockout_smoke; "
            f"/build/official_knockout_smoke /workspace/{rules_relative}"
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
        check=True,
        capture_output=True,
        text=True,
    )
    result: dict[str, Any] = json.loads(completed.stdout)
    if (
        result.get("passed") is not True
        or result.get("state_abi") != 6
        or result.get("state_bytes") != 119_936
        or result.get("cpu_digest") != result.get("gpu_digest")
        or result.get("replacement_active") != 21
        or result.get("replacement_bench_to_active_runtime") != 1
        or result.get("replacement_bench_to_active_turn_state") != 1
        or result.get("ko_trash_count") != 2
        or result.get("multi_prize_hand_count") != 2
        or result.get("terminal_result") != 1
        or result.get("terminal_reason") != 3
        or result.get("lucky_bonus_bench") != 1
        or result.get("lucky_bonus_extra_hand") != 1
        or result.get("lucky_bonus_rng_draws") != 1
        or result.get("multi_ko_first_select_min") != 1
        or result.get("multi_ko_hand_count") != 3
        or result.get("multi_lucky_extra_before_second") != 1
        or result.get("errors") != [0, 0, 0, 0, 0]
        or result.get("error_details") != [0, 0, 0, 0, 0]
    ):
        raise RuntimeError("official KO/Prize paired smoke did not pass")
    evidence = {
        **result,
        "contract": "official_knockout_paired_v2",
        "scope": [
            "pre_ko_trigger_phase",
            "post_ko_trigger_phase",
            "ko_card_and_attachment_move",
            "single_and_multi_prize_selection",
            "active_replacement",
            "active_replacement_bench_to_active_turn_state",
            "prize_and_no_active_terminal_logic",
            "lucky_bonus_yes_coin_and_extra_prize_continuation",
            "multi_knockout_lifo_prize_requests",
            "lucky_bonus_extra_prize_precedes_next_lucky_choice",
        ],
        "comparison_level": "CPU POD versus CUDA; official CPU oracle is separate",
        "docker_image": args.image,
        "docker_image_id": docker_image_id(args.image),
        "benchmark_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "benchmarks" / "official_knockout_smoke.cu"
        ),
        "knockout_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "include" / "ptcg_cuda" / "official_knockout_pod.cuh"
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
