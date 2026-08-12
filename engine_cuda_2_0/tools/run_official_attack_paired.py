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
        description="Run the official-rule CPU POD/CUDA attack paired smoke."
    )
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
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
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse the existing benchmark executable in the build directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CUDA_ENGINE_ROOT / "artifacts" / "official_attack_paired.json",
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
    build_dir = CUDA_ENGINE_ROOT / "build" / "official_attack_paired"
    build_dir.mkdir(parents=True, exist_ok=True)
    executable = build_dir / "official_attack_smoke"
    if args.skip_build:
        if not executable.is_file():
            raise SystemExit(f"benchmark executable does not exist: {executable}")
        command = f"/build/official_attack_smoke /workspace/{rules_relative}"
    else:
        command = (
            f"nvcc --threads {args.nvcc_threads} --split-compile {args.split_compile} "
            "-std=c++17 -O0 -arch=sm_86 "
            "-I/workspace/engine_cuda_2_0/include "
            "/workspace/engine_cuda_2_0/benchmarks/official_attack_smoke.cu "
            "-o /build/official_attack_smoke; "
            f"/build/official_attack_smoke /workspace/{rules_relative}"
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
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "official attack paired smoke command failed "
            f"(exit {completed.returncode})\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    result: dict[str, Any] = json.loads(completed.stdout)
    if (
        result.get("passed") is not True
        or result.get("state_abi") != 6
        or result.get("state_bytes") != 119_936
        or result.get("scenario_count") != 70
        or result.get("cpu_digest") != result.get("gpu_digest")
        or result.get("target_damage") != 30
        or result.get("next_turn") != 4
        or result.get("errors") != 0
    ):
        raise RuntimeError("official attack paired smoke did not pass")
    evidence = {
        **result,
        "contract": "official_attack_paired_v1",
        "scope": [
            "attack_legality_and_energy",
            "simple_attack_damage",
            "pre_effect_damage_change",
            "confusion_head_and_tail",
            "no_damage_coin_head_and_tail",
            "no_target_effect_coin_bypass_with_rng_consumption",
            "knockout_prize_and_active_replacement",
            "copy_enemy_attack_source_semantics",
            "double_attack_continuation",
            "post_effect_selection",
            "deck_top_attack",
            "deck_top_supporter",
            "enemy_deck_top10_shuffle_and_copy",
            "post_attack_refresh",
            "turn_end_and_next_player_draw",
            "full_state_cpu_pod_cuda_byte_parity",
        ],
        "comparison_level": "CPU POD versus CUDA; official CPU attack oracle is separate",
        "docker_image": args.image,
        "docker_image_id": docker_image_id(args.image),
        "benchmark_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "benchmarks" / "official_attack_smoke.cu"
        ),
        "attack_header_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "include" / "ptcg_cuda" / "official_attack_pod.cuh"
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
