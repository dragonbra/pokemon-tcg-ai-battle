from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
DEFAULT_DECK_ROOT = CUDA_ENGINE_ROOT / "fixtures" / "0022_deck40"
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
RESET_CUDA_TOOL = (
    CUDA_ENGINE_ROOT / "tools" / "run_official_seeded_reset_matrix_cuda.py"
)
BATTLE_PAIRED_TOOL = (
    CUDA_ENGINE_ROOT / "tools" / "run_official_battle_end_turn_cuda_paired.py"
)
BATTLE_MATRIX_TOOL = (
    CUDA_ENGINE_ROOT / "tools" / "run_official_battle_ordered_matrix_cuda.py"
)
OFFICIAL_SNAPSHOT = "3aaeaa92c9eff5272d0c28582a1f64a4cdc2f772"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the 0022 40-deck catalog against the official CPU engine "
            "and the CUDA engine. Setup parity compares official CPU fixtures "
            "to CUDA reset state. Battle parity compares official CPU/POD/CUDA "
            "state at every decision and reports terminal win/loss/draw counts."
        )
    )
    parser.add_argument("--deck-root", type=Path, default=DEFAULT_DECK_ROOT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--decision-limit", type=int, default=512)
    parser.add_argument(
        "--pair-mode",
        choices=("focal", "all", "mirrors"),
        default="focal",
        help=(
            "focal runs focal deck against every other deck in both seats; "
            "all runs every ordered deck pair; mirrors runs each deck against itself."
        ),
    )
    parser.add_argument("--include-mirrors", action="store_true")
    parser.add_argument(
        "--case-limit",
        type=int,
        default=0,
        help="Optional prefix limit for quick Docker smoke tests; 0 means no limit.",
    )
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
        default="coverage-first-legal",
    )
    parser.add_argument(
        "--setup-mode",
        choices=("seeded-first-min", "interactive-device"),
        default="interactive-device",
    )
    parser.add_argument("--setup-check-interval", type=int, default=16)
    parser.add_argument("--max-setup-actions", type=int, default=128)
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--skip-battle", action="store_true")
    parser.add_argument("--cpu-image", default=DEFAULT_CPU_IMAGE)
    parser.add_argument("--torch-image", default=DEFAULT_TORCH_IMAGE)
    parser.add_argument("--battle-image", default=DEFAULT_CPU_IMAGE)
    parser.add_argument(
        "--branch-coverage",
        action="store_true",
        help="Enable test-only per-effect rule-offset coverage during battles.",
    )
    parser.add_argument("--torch-build-rel", default="engine_cuda/build/torch_official")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=CUDA_ENGINE_ROOT / "artifacts" / "0022_deck40_official_parity.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_checked(command: list[str], *, label: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr_tail = "\n".join(completed.stderr.splitlines()[-40:])
        stdout_tail = "\n".join(completed.stdout.splitlines()[-40:])
        raise RuntimeError(
            f"{label} failed with exit {completed.returncode}\n"
            f"stdout tail:\n{stdout_tail}\n"
            f"stderr tail:\n{stderr_tail}"
        )
    return completed


def workspace_rel(path: Path) -> str:
    resolved = path.resolve()
    root = WORKSPACE_ROOT.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes workspace: {path}")
    return resolved.relative_to(root).as_posix()


def workspace_container(path: Path) -> str:
    return "/workspace/" + workspace_rel(path)


def docker_image_id(image: str) -> str:
    completed = run_checked(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        label=f"inspect docker image {image}",
    )
    return completed.stdout.strip()


def load_decks(deck_root: Path) -> tuple[list[Any], dict[str, Any]]:
    sys.path.insert(0, str(WORKSPACE_ROOT))
    decks_module = importlib.import_module("train.0022_league_training.decks")
    root = deck_root.resolve()
    plugins = list(decks_module.load_deck_plugins(root))
    catalog_path = root / "manifest.json"
    if not catalog_path.is_file():
        raise RuntimeError(f"0022 deck40 catalog manifest is missing: {catalog_path}")
    catalog: dict[str, Any] = json.loads(catalog_path.read_text(encoding="utf-8"))
    rows = catalog.get("decks")
    if not isinstance(rows, list):
        raise RuntimeError("0022 deck40 catalog manifest has no decks list")
    if (
        int(catalog.get("deck_count", -1)) != 40
        or len(rows) != 40
        or len(plugins) != 40
    ):
        raise RuntimeError(
            "0022 deck40 catalog must contain exactly 40 decks "
            f"(manifest deck_count={catalog.get('deck_count')}, rows={len(rows)}, "
            f"loaded={len(plugins)})"
        )
    by_id = {plugin.deck_id: plugin for plugin in plugins}
    manifest_ids = [str(row["deck_id"]) for row in rows]
    if len(set(manifest_ids)) != 40 or set(manifest_ids) != set(by_id):
        raise RuntimeError("0022 deck40 manifest IDs do not match loaded plugin IDs")
    for row in rows:
        deck_id = str(row["deck_id"])
        expected_sha = str(row["deck_sha256"])
        if by_id[deck_id].deck_sha256 != expected_sha:
            raise RuntimeError(
                f"deck SHA mismatch for {deck_id}: "
                f"{by_id[deck_id].deck_sha256} != {expected_sha}"
            )
    focal = str(catalog.get("focal_deck_id", ""))
    if focal not in by_id:
        raise RuntimeError(f"focal deck is missing from deck40 catalog: {focal}")
    return [by_id[deck_id] for deck_id in manifest_ids], catalog


def make_cases(
    plugins: list[Any],
    *,
    focal_deck_id: str,
    pair_mode: str,
    include_mirrors: bool,
    case_limit: int,
) -> list[dict[str, Any]]:
    by_id = {plugin.deck_id: plugin for plugin in plugins}
    if focal_deck_id not in by_id:
        raise RuntimeError(f"missing focal deck: {focal_deck_id}")

    pairs: list[tuple[Any, Any]] = []
    if pair_mode == "mirrors":
        pairs.extend((plugin, plugin) for plugin in plugins)
    elif pair_mode == "focal":
        focal = by_id[focal_deck_id]
        for opponent in plugins:
            if opponent.deck_id == focal_deck_id:
                if include_mirrors:
                    pairs.append((focal, focal))
                continue
            pairs.append((focal, opponent))
            pairs.append((opponent, focal))
    else:
        for deck0 in plugins:
            for deck1 in plugins:
                if deck0.deck_id == deck1.deck_id and not include_mirrors:
                    continue
                pairs.append((deck0, deck1))

    if case_limit < 0:
        raise RuntimeError("--case-limit must be nonnegative")
    if case_limit:
        pairs = pairs[:case_limit]

    cases: list[dict[str, Any]] = []
    for deck0, deck1 in pairs:
        name = f"{deck0.deck_id}__seat0_vs__{deck1.deck_id}__seat1"
        cases.append(
            {
                "name": name,
                "deck0_name": deck0.deck_id,
                "deck1_name": deck1.deck_id,
                "deck0_group": deck0.deck_id.split("_", 1)[0],
                "deck1_group": deck1.deck_id.split("_", 1)[0],
                "deck0": workspace_rel(deck0.root / "deck.csv"),
                "deck1": workspace_rel(deck1.root / "deck.csv"),
                "fixture_name": f"{name}.bin",
            }
        )
    if not cases:
        raise RuntimeError("no 0022 deck40 parity cases were generated")
    return cases


def validate_inputs(args: argparse.Namespace) -> None:
    if args.seed_start < 0 or args.seed_count <= 0 or args.decision_limit <= 0:
        raise RuntimeError("seed and decision arguments must be positive")
    if args.setup_check_interval <= 0 or args.max_setup_actions <= 0:
        raise RuntimeError("setup interval/action arguments must be positive")
    if not args.source.resolve().is_dir():
        raise RuntimeError(f"official source directory does not exist: {args.source}")
    license_path = (
        args.source.resolve()
        / "LICENSES"
        / "LicenseRef-PTCG-ABC-Competition-Use-Only.txt"
    )
    if not license_path.is_file():
        raise RuntimeError("official competition-only license is missing")
    if not args.rules.resolve().is_file():
        raise RuntimeError(f"private official rule pack does not exist: {args.rules}")


def write_private_manifest(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    *,
    private_dir: Path,
) -> Path:
    manifest_cases = []
    for case in cases:
        row = dict(case)
        row.pop("fixture_name", None)
        row["fixture"] = workspace_rel(private_dir / str(case["fixture_name"]))
        manifest_cases.append(row)
    manifest = {
        "schema_version": 1,
        "competition_private": True,
        "catalog": "0022_deck40",
        "setup_policy": "first_min_v1",
        "rules": workspace_rel(args.rules.resolve()),
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "ordered_seats": True,
        "mirrors_included": args.include_mirrors or args.pair_mode == "mirrors",
        "pair_mode": args.pair_mode,
        "case_limit": args.case_limit,
        "cases": manifest_cases,
    }
    path = private_dir / "manifest.json"
    write_json(path, manifest)
    return path


def run_setup_parity(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    *,
    private_dir: Path,
) -> dict[str, Any]:
    build_dir = CUDA_ENGINE_ROOT / "build" / "official_seeded_reset_0022_deck40"
    build_fixtures = build_dir / "fixtures"
    build_fixtures.mkdir(parents=True, exist_ok=True)
    case_table = build_dir / "cases.tsv"
    case_table.write_bytes(
        "".join(
            f"{case['name']}\t{case['deck0']}\t{case['deck1']}\t{case['fixture_name']}\n"
            for case in cases
        ).encode("utf-8")
    )

    rules_container = workspace_container(args.rules.resolve())
    cpu_command = (
        "set -euo pipefail; "
        "g++ -std=c++20 -O3 -I/official -I/workspace/engine_cuda/include "
        "-I/workspace/engine_cuda/extractor "
        "/workspace/engine_cuda/extractor/official_seeded_reset_oracle.cpp "
        "-o /build/official_seeded_reset_oracle; "
        "while IFS=$'\t' read -r name deck0 deck1 fixture; do "
        f"/build/official_seeded_reset_oracle {rules_container} "
        '"/workspace/$deck0" "/workspace/$deck1" '
        f'{args.seed_start} {args.seed_count} "/build/fixtures/$fixture"; '
        "done < /build/cases.tsv"
    )
    cpu_completed = run_checked(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/bin/bash",
            "-v",
            f"{WORKSPACE_ROOT.resolve()}:/workspace:ro",
            "-v",
            f"{args.source.resolve()}:/official:ro",
            "-v",
            f"{build_dir.resolve()}:/build:rw",
            args.cpu_image,
            "-lc",
            cpu_command,
        ],
        label="0022 deck40 official CPU seeded-reset fixture generation",
    )
    cpu_results = [
        json.loads(line) for line in cpu_completed.stdout.splitlines() if line.strip()
    ]
    if len(cpu_results) != len(cases):
        raise RuntimeError(
            f"CPU setup fixture generation returned {len(cpu_results)} results "
            f"for {len(cases)} cases"
        )
    cpu_state_mismatches = 0
    cpu_status_mismatches = 0
    for case, result in zip(cases, cpu_results, strict=True):
        if result.get("passed") is not True:
            raise RuntimeError(f"CPU setup fixture failed for {case['name']}: {result}")
        cpu_state_mismatches += int(result.get("state_mismatches", 0))
        cpu_status_mismatches += int(result.get("status_mismatches", 0))
    if cpu_state_mismatches or cpu_status_mismatches:
        raise RuntimeError("official CPU/POD setup fixture generation had mismatches")

    private_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        source_fixture = build_fixtures / str(case["fixture_name"])
        if not source_fixture.is_file():
            raise RuntimeError(f"matrix fixture was not generated: {source_fixture}")
        shutil.copyfile(source_fixture, private_dir / str(case["fixture_name"]))

    private_manifest = write_private_manifest(args, cases, private_dir=private_dir)
    cuda_result_path = build_dir / "cuda_result.json"
    torch_build = str(args.torch_build_rel).strip().replace("\\", "/").strip("/")
    py_path = f"/workspace/{torch_build}:/workspace/engine_cuda/python:/workspace/engine_cuda/tools"
    cuda_command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--shm-size=2g",
        "-e",
        f"PYTHONPATH={py_path}",
        "-v",
        f"{WORKSPACE_ROOT.resolve()}:/workspace",
        "-w",
        "/workspace",
        args.torch_image,
        "python",
        "/workspace/engine_cuda/tools/run_official_seeded_reset_matrix_cuda.py",
        "--manifest",
        workspace_container(private_manifest),
        "--mode",
        args.setup_mode,
        "--setup-check-interval",
        str(args.setup_check_interval),
        "--max-setup-actions",
        str(args.max_setup_actions),
        "--device-index",
        str(args.device_index),
        "--output",
        workspace_container(cuda_result_path),
    ]
    run_checked(cuda_command, label="0022 deck40 CUDA seeded-reset parity")
    cuda_result: dict[str, Any] = json.loads(
        cuda_result_path.read_text(encoding="utf-8")
    )
    if cuda_result.get("passed") is not True:
        raise RuntimeError("CUDA setup parity did not pass")
    return {
        **cuda_result,
        "contract": "0022_deck40_official_cpu_pod_cuda_setup_parity_v1",
        "mode": args.setup_mode,
        "cpu_official_pod_state_mismatches": cpu_state_mismatches,
        "cpu_official_pod_status_mismatches": cpu_status_mismatches,
        "private_manifest": workspace_rel(private_manifest),
        "private_manifest_sha256": sha256_file(private_manifest),
        "cpu_image": args.cpu_image,
        "cpu_image_id": docker_image_id(args.cpu_image),
        "torch_image": args.torch_image,
        "torch_image_id": docker_image_id(args.torch_image),
        "comparison_level": (
            "official CPU seeded setup fixture -> CPU POD fixture -> CUDA reset; "
            "interactive-device mode also replays setup choices through CUDA codec/actions"
        ),
    }


def run_battle_parity(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    *,
    private_manifest: Path,
) -> dict[str, Any]:
    build_dir = CUDA_ENGINE_ROOT / "build" / "official_battle_end_turn_cuda_paired"
    build_probe = build_dir / "0022_deck40_build_probe.json"
    first = cases[0]
    build_command = [
        sys.executable,
        str(BATTLE_PAIRED_TOOL),
        "--policy",
        args.policy,
        "--deck0",
        str((WORKSPACE_ROOT / str(first["deck0"])).resolve()),
        "--deck1",
        str((WORKSPACE_ROOT / str(first["deck1"])).resolve()),
        "--seed-start",
        str(args.seed_start),
        "--seed-count",
        "1",
        "--decision-limit",
        str(args.decision_limit),
        "--source",
        str(args.source.resolve()),
        "--rules",
        str(args.rules.resolve()),
        "--image",
        args.battle_image,
        "--output",
        str(build_probe),
    ]
    if args.branch_coverage:
        build_command.append("--branch-coverage")
    run_checked(build_command, label="official CUDA battle paired build probe")

    battle_result_path = (
        CUDA_ENGINE_ROOT / "build" / "official_0022_deck40_battle_matrix.json"
    )
    matrix_command = [
        sys.executable,
        str(BATTLE_MATRIX_TOOL),
        "--manifest",
        str(private_manifest.resolve()),
        "--seed-start",
        str(args.seed_start),
        "--seed-count",
        str(args.seed_count),
        "--decision-limit",
        str(args.decision_limit),
        "--policy",
        args.policy,
        "--source",
        str(args.source.resolve()),
        "--rules",
        str(args.rules.resolve()),
        "--image",
        args.battle_image,
        "--output",
        str(battle_result_path),
    ]
    if args.branch_coverage:
        matrix_command.append("--branch-coverage")
    run_checked(matrix_command, label="0022 deck40 official CPU/POD/CUDA battle parity")
    battle_result: dict[str, Any] = json.loads(
        battle_result_path.read_text(encoding="utf-8")
    )
    if battle_result.get("passed") is not True:
        raise RuntimeError("battle parity matrix did not pass")
    return {
        **battle_result,
        "contract": "0022_deck40_official_cpu_pod_cuda_battle_parity_v1",
        "battle_image": args.battle_image,
        "battle_image_id": docker_image_id(args.battle_image),
        "comparison_level": (
            "setup-to-terminal battles; every decision compares legal action choice "
            "semantics, CPU/POD/CUDA raw state bytes, flow status, and terminal outcome"
        ),
    }


def compact_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    omitted = {"cases"}
    return {key: value for key, value in result.items() if key not in omitted}


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    plugins, catalog = load_decks(args.deck_root)
    focal_deck_id = str(catalog["focal_deck_id"])
    cases = make_cases(
        plugins,
        focal_deck_id=focal_deck_id,
        pair_mode=args.pair_mode,
        include_mirrors=args.include_mirrors,
        case_limit=args.case_limit,
    )
    groups: dict[str, int] = {}
    for plugin in plugins:
        group = plugin.deck_id.split("_", 1)[0]
        groups[group] = groups.get(group, 0) + 1

    suffix = f"{args.pair_mode}_s{args.seed_start}_{args.seed_count}" + (
        f"_n{args.case_limit}" if args.case_limit else ""
    )
    private_dir = (
        CUDA_ENGINE_ROOT
        / "generated"
        / "private"
        / "official_3aaeaa92"
        / f"0022_deck40_{suffix}"
    )
    private_dir.mkdir(parents=True, exist_ok=True)
    private_manifest = write_private_manifest(args, cases, private_dir=private_dir)

    setup_result: dict[str, Any] | None = None
    battle_result: dict[str, Any] | None = None
    if not args.skip_setup:
        setup_result = run_setup_parity(args, cases, private_dir=private_dir)
        private_manifest = private_dir / "manifest.json"
    if not args.skip_battle:
        if not private_manifest.is_file():
            private_manifest = write_private_manifest(
                args, cases, private_dir=private_dir
            )
        battle_result = run_battle_parity(
            args, cases, private_manifest=private_manifest
        )

    passed = (
        args.skip_setup
        or (setup_result is not None and setup_result.get("passed") is True)
    ) and (
        args.skip_battle
        or (battle_result is not None and battle_result.get("passed") is True)
    )
    outcome_mismatches = int((battle_result or {}).get("outcome_mismatches", 0))
    state_mismatches = int((battle_result or {}).get("state_mismatches", 0)) + int(
        (setup_result or {}).get("state_mismatches", 0)
    )
    status_mismatches = int((battle_result or {}).get("status_mismatches", 0)) + int(
        (setup_result or {}).get("status_mismatches", 0)
    )
    evidence = {
        "passed": passed
        and outcome_mismatches == 0
        and state_mismatches == 0
        and status_mismatches == 0,
        "contract": "0022_deck40_official_cpu_cuda_semantic_and_outcome_parity_v1",
        "deck_count": len(plugins),
        "deck_groups": groups,
        "focal_deck_id": focal_deck_id,
        "pair_mode": args.pair_mode,
        "case_count": len(cases),
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "battle_policy": args.policy,
        "branch_coverage_enabled": args.branch_coverage,
        "setup_mode": args.setup_mode,
        "state_mismatches": state_mismatches,
        "status_mismatches": status_mismatches,
        "outcome_mismatches": outcome_mismatches,
        "setup_checked": not args.skip_setup,
        "battle_checked": not args.skip_battle,
        "private_manifest": workspace_rel(private_manifest),
        "private_manifest_sha256": sha256_file(private_manifest),
        "official_source_snapshot": OFFICIAL_SNAPSHOT,
        "deck_manifest_sha256": sha256_file(args.deck_root.resolve() / "manifest.json"),
        "setup_result": compact_result(setup_result),
        "battle_result": compact_result(battle_result),
    }
    write_json(args.output, evidence)
    print(
        json.dumps({**evidence, "output": str(args.output)}, indent=2, sort_keys=True)
    )
    if evidence["passed"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
