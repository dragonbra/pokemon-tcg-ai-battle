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
    / "official_main_fixture.bin"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare official CPU MainSelect Attack/End with the CPU POD."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--image", default="nvidia/cuda:13.0.0-devel-ubuntu24.04")
    parser.add_argument(
        "--output",
        type=Path,
        default=CUDA_ENGINE_ROOT / "artifacts" / "official_main_oracle.json",
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

    build_dir = CUDA_ENGINE_ROOT / "build" / "official_main_oracle_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    executable = build_dir / "official_main_oracle"
    built_fixture = build_dir / "official_main_fixture.bin"
    command = (
        "set -euo pipefail; "
        "g++ -std=c++20 -O3 -I/official -I/workspace/engine_cuda_2_0/include "
        "/workspace/engine_cuda_2_0/extractor/official_main_oracle.cpp "
        "-ldl -o /build/official_main_oracle; "
        f"/build/official_main_oracle /workspace/{rules_relative} "
        "/build/official_main_fixture.bin"
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
            "official main oracle command failed "
            f"(exit {completed.returncode})\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    result: dict[str, Any] = json.loads(completed.stdout)
    if (
        result.get("passed") is not True
        or result.get("scenarios") != 35
        or result.get("scope")
        != [
            "main_options",
            "main_attack",
            "main_end",
            "main_play_basic",
            "main_play_stadium",
            "main_play_item",
            "main_play_supporter",
            "main_play_item_selection",
            "main_play_item_resume",
            "main_attach_energy",
            "main_evolve",
            "main_ability",
            "main_discard",
            "main_retreat_energy",
            "main_retreat_switch",
            "main_retreat_complete",
            "main_attach_tool",
            "main_retreat_zero_cost",
            "main_retreat_multi_energy_continue",
            "main_retreat_multi_energy_switch",
            "main_retreat_blocked_already_used",
            "main_retreat_blocked_asleep",
            "main_retreat_blocked_this_turn",
            "main_retreat_blocked_continual",
            "main_retreat_blocked_poison",
            "main_retreat_blocked_pokemon_item",
            "main_retreat_blocked_insufficient_energy",
            "main_evolve_rainbow_dna",
            "main_ability_selection_activation",
            "main_ability_optional_skip",
            "main_ability_selection_apply",
            "main_ability_blocked_area",
            "main_ability_blocked_condition",
            "main_ability_blocked_suppressed",
            "main_ability_blocked_once_turn",
        ]
        or not built_fixture.is_file()
    ):
        raise RuntimeError("official CPU/POD main oracle did not pass")
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_bytes(built_fixture.read_bytes())

    evidence = {
        **result,
        "contract": "official_cpu_pod_main_oracle_v1",
        "official_source_license": "competition-use-only",
        "official_source_snapshot": "3aaeaa92c9eff5272d0c28582a1f64a4cdc2f772",
        "comparison_level": "official CPU versus independent CPU POD",
        "docker_image": args.image,
        "docker_image_id": docker_image_id(args.image),
        "harness_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "extractor" / "official_main_oracle.cpp"
        ),
        "main_header_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "include" / "ptcg_cuda" / "official_main_pod.cuh"
        ),
        "rule_pack_sha256": sha256_file(rules),
        "private_fixture_sha256": sha256_file(args.fixture),
        "executable_sha256": sha256_file(executable),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**result, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
