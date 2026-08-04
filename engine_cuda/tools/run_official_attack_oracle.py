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
        description="Compare official CPU attack flow with the independent CPU POD."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--image", default="nvidia/cuda:13.0.0-devel-ubuntu24.04")
    parser.add_argument(
        "--resume-fixture",
        type=Path,
        default=(
            CUDA_ENGINE_ROOT
            / "generated"
            / "private"
            / "official_3aaeaa92"
            / "official_attack_resume_fixture.bin"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CUDA_ENGINE_ROOT / "artifacts" / "official_attack_oracle.json",
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

    build_dir = CUDA_ENGINE_ROOT / "build" / "official_attack_oracle_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    executable = build_dir / "official_attack_oracle"
    command = (
        "set -euo pipefail; "
        "g++ -std=c++20 -O3 -I/official -I/workspace/engine_cuda/include "
        "/workspace/engine_cuda/extractor/official_attack_oracle.cpp "
        "-ldl -o /build/official_attack_oracle; "
        f"/build/official_attack_oracle /workspace/{rules_relative} "
        "/build/official_attack_resume_fixture.bin"
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
    built_fixture = build_dir / "official_attack_resume_fixture.bin"
    if not built_fixture.is_file():
        raise RuntimeError("official attack oracle did not create the resume fixture")
    args.resume_fixture.parent.mkdir(parents=True, exist_ok=True)
    args.resume_fixture.write_bytes(built_fixture.read_bytes())
    expected_scope = [
        "simple_damage",
        "pre_effect_damage_change",
        "confusion_head",
        "confusion_tail",
        "no_damage_coin_head",
        "no_damage_coin_tail",
        "no_target_bypass",
        "knockout_prize_replacement",
        "copy_enemy_attack",
        "double_attack",
        "post_effect_selection",
        "deck_top_attack",
        "deck_top_supporter",
        "enemy_deck_top10_attack",
    ]
    if (
        result.get("passed") is not True
        or result.get("scenarios") != 14
        or result.get("checks") != 120
        or result.get("scope") != expected_scope
    ):
        raise RuntimeError("official CPU/POD attack oracle did not pass")
    evidence = {
        **result,
        "contract": "official_cpu_pod_attack_oracle_v1",
        "official_source_license": "competition-use-only",
        "official_source_snapshot": "3aaeaa92c9eff5272d0c28582a1f64a4cdc2f772",
        "comparison_level": "official CPU versus independent CPU POD",
        "docker_image": args.image,
        "docker_image_id": docker_image_id(args.image),
        "harness_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "extractor" / "official_attack_oracle.cpp"
        ),
        "attack_header_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "include" / "ptcg_cuda" / "official_attack_pod.cuh"
        ),
        "rule_pack_sha256": sha256_file(rules),
        "private_resume_fixture_sha256": sha256_file(args.resume_fixture),
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
