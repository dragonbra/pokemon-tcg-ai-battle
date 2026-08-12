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
        description=(
            "Run end-turn-only official CPU/POD/CUDA full-battle paired replay."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
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
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--decision-limit", type=int, default=256)
    parser.add_argument(
        "--policy",
        choices=(
            "end",
            "basic-play-then-end",
            "basic-play-attach-then-end",
            "basic-play-evolve-attach-then-end",
            "coverage-first-legal",
            "coverage-random-legal",
        ),
        default="end",
    )
    parser.add_argument("--image", default="nvidia/cuda:13.0.0-devel-ubuntu24.04")
    parser.add_argument("--nvcc-threads", type=int, default=4)
    parser.add_argument("--split-compile", type=int, default=1)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--branch-coverage",
        action="store_true",
        help="Enable test-only per-effect rule-offset coverage bitmaps.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            CUDA_ENGINE_ROOT / "artifacts" / "official_battle_end_turn_cuda_paired.json"
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def container_path(path: Path) -> str:
    relative = path.resolve().relative_to(WORKSPACE_ROOT.resolve())
    return "/workspace/" + relative.as_posix()


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
    if (
        args.seed_start < 0
        or args.seed_count <= 0
        or args.decision_limit <= 0
        or args.nvcc_threads <= 0
        or args.split_compile <= 0
    ):
        raise SystemExit("seed, compile and decision arguments must be positive")
    source = args.source.resolve()
    if not source.is_dir():
        raise SystemExit(f"official source directory does not exist: {source}")
    license_path = source / "LICENSES" / "LicenseRef-PTCG-ABC-Competition-Use-Only.txt"
    if not license_path.is_file():
        raise SystemExit("official competition-only license is missing")
    for path in (args.rules, args.deck0, args.deck1):
        if not path.resolve().is_file():
            raise SystemExit(f"required input does not exist: {path}")
        if not path.resolve().is_relative_to(WORKSPACE_ROOT.resolve()):
            raise SystemExit(f"input must be inside the workspace: {path}")

    build_dir = CUDA_ENGINE_ROOT / "build" / "official_battle_end_turn_cuda_paired"
    build_dir.mkdir(parents=True, exist_ok=True)
    executable = build_dir / "official_battle_end_turn_cuda_paired"
    run = (
        "/build/official_battle_end_turn_cuda_paired "
        f"{container_path(args.rules)} "
        f"{container_path(args.deck0)} "
        f"{container_path(args.deck1)} "
        f"{args.seed_start} {args.seed_count} {args.decision_limit} {args.policy}"
    )
    if args.skip_build:
        if not executable.is_file():
            raise SystemExit(f"executable does not exist: {executable}")
        command = "set -euo pipefail; " + run
    else:
        coverage_define = (
            "-DPTCG_OFFICIAL_BRANCH_COVERAGE=1 " if args.branch_coverage else ""
        )
        command = (
            "set -euo pipefail; "
            f"nvcc --threads {args.nvcc_threads} "
            f"--split-compile {args.split_compile} "
            "-std=c++20 -O0 -arch=sm_86 "
            + coverage_define
            + "-I/official -I/workspace/engine_cuda_2_0/include "
            "-I/workspace/engine_cuda_2_0/extractor "
            "/workspace/engine_cuda_2_0/benchmarks/official_battle_end_turn_cuda_paired.cu "
            "/workspace/engine_cuda_2_0/src/official_engine_kernels.cu "
            "-o /build/official_battle_end_turn_cuda_paired; " + run
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
            "official CUDA battle paired command failed "
            f"(exit {completed.returncode})\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    result: dict[str, Any] = json.loads(completed.stdout)
    expected_scope = {
        "end": "end_turn_only_full_battle_cuda_paired",
        "basic-play-then-end": "basic_play_then_end_full_battle_cuda_paired",
        "basic-play-attach-then-end": (
            "basic_play_attach_then_end_full_battle_cuda_paired"
        ),
        "basic-play-evolve-attach-then-end": (
            "basic_play_evolve_attach_then_end_full_battle_cuda_paired"
        ),
        "coverage-first-legal": ("coverage_first_legal_full_battle_cuda_paired"),
        "coverage-random-legal": ("coverage_random_legal_full_battle_cuda_paired"),
    }[args.policy]
    if (
        result.get("passed") is not True
        or result.get("scope") != expected_scope
        or result.get("seed_count") != args.seed_count
        or result.get("state_abi") != 6
        or (not args.branch_coverage and result.get("state_bytes") != 119_936)
        or (
            args.branch_coverage
            and (
                result.get("branch_coverage_enabled") is not True
                or not 119_936 < int(result.get("state_bytes", 0)) <= 131_072
            )
        )
        or result.get("status_mismatches") != 0
        or result.get("state_mismatches") != 0
        or result.get("outcome_mismatches") != 0
        or (
            int(result.get("player0_wins", 0))
            + int(result.get("player1_wins", 0))
            + int(result.get("draws", 0))
            + int(result.get("unfinished_battles", 0))
            != args.seed_count
        )
        or result.get("unfinished_battles", 0) != 0
        or (
            args.policy
            in (
                "basic-play-then-end",
                "basic-play-attach-then-end",
                "basic-play-evolve-attach-then-end",
            )
            and result.get("basic_play_actions", 0) == 0
        )
        or (
            args.policy
            in (
                "basic-play-attach-then-end",
                "basic-play-evolve-attach-then-end",
            )
            and result.get("basic_energy_attach_actions", 0) == 0
        )
        or (
            args.policy == "basic-play-evolve-attach-then-end"
            and result.get("evolve_actions", 0) == 0
        )
        or (
            args.policy in ("coverage-first-legal", "coverage-random-legal")
            and result.get("coverage_attack_actions", 0)
            + result.get("coverage_end_actions", 0)
            == 0
        )
    ):
        raise RuntimeError("official CPU/POD/CUDA battle paired replay did not pass")
    evidence = {
        **result,
        "contract": {
            "end": "official_cpu_pod_cuda_end_turn_full_battle_v1",
            "basic-play-then-end": ("official_cpu_pod_cuda_basic_play_full_battle_v1"),
            "basic-play-attach-then-end": (
                "official_cpu_pod_cuda_basic_play_attach_full_battle_v1"
            ),
            "basic-play-evolve-attach-then-end": (
                "official_cpu_pod_cuda_basic_play_evolve_attach_full_battle_v1"
            ),
            "coverage-first-legal": (
                "official_cpu_pod_cuda_coverage_first_legal_full_battle_v1"
            ),
            "coverage-random-legal": (
                "official_cpu_pod_cuda_coverage_random_legal_full_battle_v1"
            ),
        }[args.policy],
        "policy": args.policy,
        "coverage_boundary": (
            "Real setup-to-terminal battles. The end policy selects only End; "
            "the basic-play policy first benches legal Basic Pokemon without "
            "abilities; the attach policy additionally attaches the first legal "
            "Basic Energy each turn; the evolve policy also selects the first legal "
            "direct evolution. The coverage-first-legal policy executes at most "
            "four legal Ability/Play/Evolve/Attach/Discard/Retreat main actions "
            "per turn, then attacks or ends. The random variant deterministically "
            "samples among legal main actions, attacks, and YesNo branches without "
            "changing game RNG; both variants select min/max cardinalities. Natural "
            "battle coverage is still "
            "not proof of every card/effect branch."
        ),
        "comparison_level": (
            "Raw CPU POD/CUDA state bytes plus canonical official bridge state bytes."
        ),
        "official_source_license": "competition-use-only",
        "official_source_snapshot": "3aaeaa92c9eff5272d0c28582a1f64a4cdc2f772",
        "docker_image": args.image,
        "docker_image_id": docker_image_id(args.image),
        "benchmark_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "benchmarks" / "official_battle_end_turn_cuda_paired.cu"
        ),
        "harness_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "extractor" / "official_battle_end_turn_paired.cpp"
        ),
        "runtime_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "src" / "official_engine_kernels.cu"
        ),
        "deck0_sha256": sha256_file(args.deck0),
        "deck1_sha256": sha256_file(args.deck1),
        "rule_pack_sha256": sha256_file(args.rules),
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
