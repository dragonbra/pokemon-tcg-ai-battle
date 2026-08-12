from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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
DEFAULT_CPU_IMAGE = "nvidia/cuda:13.0.0-devel-ubuntu24.04"
DEFAULT_TORCH_IMAGE = "pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel"

DECKS = [
    (
        "lucario",
        "bc_models/archive/agent_pure_lucario_v1_bc512_e4_probe/deck.csv",
    ),
    (
        "cynthia",
        "bc_models/agent_cynthia_core_meanpool_epochmix_v1_c7b3253f_20260727/deck.csv",
    ),
    (
        "kangaskhan_crustle",
        "bc_models/agent_kangaskhan_crustle_core_meanpool_epochmix_v2_e510e4d4_20260728/deck.csv",
    ),
    (
        "marnie_prize_control",
        "bc_models/agent_marnie_prize_control_v4_3121746f_20260728/deck.csv",
    ),
    (
        "yushin_alakazam",
        "bc_models/agent_yushin_idonly_bc_v1_20260723/deck.csv",
    ),
    (
        "dragapult",
        "bc_models/dragapult_large_data_decoder_2layer/deck.csv",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate real-official seeded fixtures once, then compare all "
            "ordered six-BC matchups with CUDA in one GPU process."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--include-mirrors", action="store_true")
    parser.add_argument("--cpu-image", default=DEFAULT_CPU_IMAGE)
    parser.add_argument("--torch-image", default=DEFAULT_TORCH_IMAGE)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            CUDA_ENGINE_ROOT
            / "artifacts"
            / "official_seeded_reset_ordered_matrix_s1_20.json"
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def docker_image_id(image: str) -> str:
    completed = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def relative_workspace(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()


def main() -> None:
    args = parse_args()
    if args.seed_start < 0 or args.seed_count <= 0:
        raise SystemExit("seed start must be nonnegative and seed count must be positive")
    source = args.source.resolve()
    rules = args.rules.resolve()
    if not source.is_dir():
        raise SystemExit(f"official source directory does not exist: {source}")
    license_path = source / "LICENSES" / "LicenseRef-PTCG-ABC-Competition-Use-Only.txt"
    if not license_path.is_file():
        raise SystemExit("official competition-only license is missing")
    if not rules.is_file():
        raise SystemExit(f"private rule pack does not exist: {rules}")

    deck_rows: list[tuple[str, Path]] = []
    for name, relative in DECKS:
        path = (WORKSPACE_ROOT / relative).resolve()
        if not path.is_file():
            raise SystemExit(f"opponent-pool deck does not exist: {path}")
        deck_rows.append((name, path))

    cases: list[dict[str, Any]] = []
    for index0, (name0, deck0) in enumerate(deck_rows):
        for index1, (name1, deck1) in enumerate(deck_rows):
            if not args.include_mirrors and index0 == index1:
                continue
            case_name = f"{name0}__seat0_vs__{name1}__seat1"
            fixture_name = f"{case_name}.bin"
            cases.append(
                {
                    "name": case_name,
                    "deck0_name": name0,
                    "deck1_name": name1,
                    "deck0": relative_workspace(deck0),
                    "deck1": relative_workspace(deck1),
                    "fixture_name": fixture_name,
                }
            )

    build_dir = CUDA_ENGINE_ROOT / "build" / "official_seeded_reset_matrix"
    build_fixtures = build_dir / "fixtures"
    build_fixtures.mkdir(parents=True, exist_ok=True)
    executable = build_dir / "official_seeded_reset_oracle"
    case_table = build_dir / "cases.tsv"
    # This file is consumed by bash in the Linux container.  Emit literal LF
    # bytes so Windows text-mode CR does not become part of the fixture name.
    case_table.write_bytes(
        "".join(
            f"{case['name']}\t{case['deck0']}\t{case['deck1']}\t{case['fixture_name']}\n"
            for case in cases
        ).encode("utf-8")
    )

    rules_container = "/workspace/" + relative_workspace(rules)
    cpu_command = (
        "set -euo pipefail; "
        "g++ -std=c++20 -O3 -I/official -I/workspace/engine_cuda_2_0/include "
        "-I/workspace/engine_cuda_2_0/extractor "
        "/workspace/engine_cuda_2_0/extractor/official_seeded_reset_oracle.cpp "
        "-o /build/official_seeded_reset_oracle; "
        "while IFS=$'\\t' read -r name deck0 deck1 fixture; do "
        f"/build/official_seeded_reset_oracle {rules_container} "
        '"/workspace/$deck0" "/workspace/$deck1" '
        f'{args.seed_start} {args.seed_count} "/build/fixtures/$fixture"; '
        "done < /build/cases.tsv"
    )
    cpu_completed = subprocess.run(
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
            args.cpu_image,
            "-lc",
            cpu_command,
        ],
        capture_output=True,
        text=True,
    )
    if cpu_completed.returncode != 0:
        raise RuntimeError(
            "ordered official CPU/POD matrix failed "
            f"(exit {cpu_completed.returncode})\n"
            f"stdout:\n{cpu_completed.stdout}\n"
            f"stderr:\n{cpu_completed.stderr}"
        )
    cpu_results = [
        json.loads(line) for line in cpu_completed.stdout.splitlines() if line.strip()
    ]
    if len(cpu_results) != len(cases):
        raise RuntimeError(
            f"CPU matrix returned {len(cpu_results)} results for {len(cases)} cases"
        )
    for case, result in zip(cases, cpu_results, strict=True):
        if (
            result.get("passed") is not True
            or result.get("seed_count") != args.seed_count
            or result.get("state_mismatches") != 0
            or result.get("status_mismatches") != 0
        ):
            raise RuntimeError(f"CPU seeded reset failed for {case['name']}: {result}")

    private_dir = (
        CUDA_ENGINE_ROOT
        / "generated"
        / "private"
        / "official_3aaeaa92"
        / f"seeded_reset_matrix_s{args.seed_start}_{args.seed_count}"
    )
    private_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        source_fixture = build_fixtures / str(case["fixture_name"])
        if not source_fixture.is_file():
            raise RuntimeError(f"matrix fixture was not generated: {source_fixture}")
        target_fixture = private_dir / str(case["fixture_name"])
        shutil.copyfile(source_fixture, target_fixture)
        case["fixture"] = relative_workspace(target_fixture)
        del case["fixture_name"]

    private_manifest = private_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "competition_private": True,
        "setup_policy": "first_min_v1",
        "rules": relative_workspace(rules),
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "ordered_seats": True,
        "mirrors_included": args.include_mirrors,
        "cases": cases,
    }
    private_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    cuda_result_path = build_dir / "cuda_result.json"
    cuda_command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--shm-size=2g",
        "-e",
        "PYTHONPATH=/workspace/engine_cuda_2_0/build/torch_official:/workspace/engine_cuda_2_0/python:/workspace/engine_cuda_2_0/tools",
        "-v",
        f"{WORKSPACE_ROOT.resolve()}:/workspace",
        "-w",
        "/workspace",
        args.torch_image,
        "python",
        "/workspace/engine_cuda_2_0/tools/run_official_seeded_reset_matrix_cuda.py",
        "--manifest",
        "/workspace/" + relative_workspace(private_manifest),
        "--device-index",
        str(args.device_index),
        "--output",
        "/workspace/" + relative_workspace(cuda_result_path),
    ]
    cuda_completed = subprocess.run(
        cuda_command, capture_output=True, text=True
    )
    if cuda_completed.returncode != 0:
        raise RuntimeError(
            "ordered CUDA seeded-reset matrix failed "
            f"(exit {cuda_completed.returncode})\n"
            f"stdout:\n{cuda_completed.stdout}\n"
            f"stderr:\n{cuda_completed.stderr}"
        )
    cuda_result: dict[str, Any] = json.loads(
        cuda_result_path.read_text(encoding="utf-8")
    )
    if cuda_result.get("passed") is not True:
        raise RuntimeError("CUDA seeded-reset matrix did not pass")

    cpu_state_mismatches = sum(int(row["state_mismatches"]) for row in cpu_results)
    cpu_status_mismatches = sum(int(row["status_mismatches"]) for row in cpu_results)
    evidence = {
        **cuda_result,
        "contract": "official_cpu_pod_cuda_seeded_reset_ordered_matrix_v1",
        "opponent_count": len(DECKS),
        "ordered_seats": True,
        "mirrors_included": args.include_mirrors,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "cpu_official_pod_state_mismatches": cpu_state_mismatches,
        "cpu_official_pod_status_mismatches": cpu_status_mismatches,
        "comparison_level": (
            "real official CPU -> full OfficialStatePod fixture -> CPU POD and CUDA; "
            "all 119,936 state bytes plus boundary status"
        ),
        "official_source_license": "competition-use-only",
        "official_source_snapshot": "3aaeaa92c9eff5272d0c28582a1f64a4cdc2f772",
        "cpu_docker_image": args.cpu_image,
        "cpu_docker_image_id": docker_image_id(args.cpu_image),
        "torch_docker_image": args.torch_image,
        "torch_docker_image_id": docker_image_id(args.torch_image),
        "oracle_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "extractor" / "official_seeded_reset_oracle.cpp"
        ),
        "cuda_worker_sha256": sha256_file(
            CUDA_ENGINE_ROOT / "tools" / "run_official_seeded_reset_matrix_cuda.py"
        ),
        "private_manifest_sha256": sha256_file(private_manifest),
        "executable_sha256": sha256_file(executable),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {key: value for key, value in evidence.items() if key != "cases"}
    print(json.dumps({**summary, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
