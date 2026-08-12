from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
BATTLE_TOOL = CUDA_ENGINE_ROOT / "tools" / "run_official_battle_end_turn_cuda_paired.py"
DEFAULT_SOURCE = WORKSPACE_ROOT / "engine" / "source" / "ptcgProgram 22"
DEFAULT_RULES = (
    CUDA_ENGINE_ROOT
    / "generated"
    / "private"
    / "official_3aaeaa92"
    / "official_rules.bin"
)
DEFAULT_IMAGE = "nvidia/cuda:13.0.0-devel-ubuntu24.04"

COUNTER_FIELDS = (
    "decisions_compared",
    "basic_play_actions",
    "basic_energy_attach_actions",
    "evolve_actions",
    "optional_decline_actions",
    "prize_selection_actions",
    "prize_cards_taken",
    "active_replacement_actions",
    "trigger_order_actions",
    "end_actions",
    "coverage_play_actions",
    "coverage_attach_actions",
    "coverage_evolve_actions",
    "coverage_ability_actions",
    "coverage_discard_actions",
    "coverage_retreat_actions",
    "coverage_attack_actions",
    "coverage_end_actions",
    "coverage_yes_actions",
    "coverage_non_main_actions",
    "coverage_zero_cardinality_actions",
    "coverage_max_cardinality_actions",
    "player0_wins",
    "player1_wins",
    "draws",
    "unfinished_battles",
    "outcome_mismatches",
)

SET_FIELDS = (
    "effect_offsets_reached",
    "effect_offsets_applied",
    "effect_offsets_condition_true",
    "effect_offsets_condition_false",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run official CPU/reference/CUDA full-battle replay for every ordered "
            "deck pair in a private seeded-reset matrix manifest."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--decision-limit", type=int, default=512)
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--rules", type=Path, default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument(
        "--case-name",
        action="append",
        default=[],
        help="Run only an exact manifest case name; may be repeated.",
    )
    parser.add_argument(
        "--branch-coverage",
        action="store_true",
        help="Enable test-only per-effect rule-offset coverage bitmaps.",
    )
    parser.add_argument(
        "--persistent-container",
        action="store_true",
        help=(
            "Compile and run every case in one Docker GPU container, reusing "
            "the CUDA context and uploaded rule pack."
        ),
    )
    parser.add_argument(
        "--skip-persistent-build",
        action="store_true",
        help="Reuse the existing persistent CUDA matrix executable.",
    )
    parser.add_argument(
        "--nvcc-threads",
        type=int,
        default=4,
        help="Host threads available to persistent nvcc compilation (default: 4).",
    )
    return parser.parse_args()


def workspace_path(relative: str) -> Path:
    path = (WORKSPACE_ROOT / relative).resolve()
    if not path.is_relative_to(WORKSPACE_ROOT.resolve()):
        raise ValueError(f"matrix path escapes workspace: {relative}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path, suffixes: tuple[str, ...]) -> str:
    """Hash build-relevant files by relative path and content."""

    digest = hashlib.sha256()
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
    if not files:
        raise ValueError(f"build input tree contains no matching files: {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def persistent_build_contract(
    *,
    source: Path,
    rules: Path,
    image: str,
    branch_coverage: bool,
    nvcc_threads: int,
    engine_root: Path = CUDA_ENGINE_ROOT,
) -> dict[str, Any]:
    """Return the exact source/rule contract required to reuse a Gate-A binary."""

    return {
        "schema_version": "official_cuda_gate_a_binary_provenance_v1",
        "official_source_tree_sha256": sha256_tree(
            source, (".h", ".hpp", ".cpp", ".c", ".cc")
        ),
        "cuda_include_tree_sha256": sha256_tree(
            engine_root / "include", (".h", ".hpp", ".cuh")
        ),
        "cuda_extractor_tree_sha256": sha256_tree(
            engine_root / "extractor", (".h", ".hpp", ".cpp", ".cc")
        ),
        "ordered_matrix_source_sha256": sha256_file(
            engine_root / "benchmarks" / "official_battle_ordered_matrix_cuda_paired.cu"
        ),
        "paired_battle_source_sha256": sha256_file(
            engine_root / "benchmarks" / "official_battle_end_turn_cuda_paired.cu"
        ),
        "cuda_kernel_source_sha256": sha256_file(
            engine_root / "src" / "official_engine_kernels.cu"
        ),
        "rule_pack_sha256": sha256_file(rules),
        "container_image": image,
        "nvcc_threads": nvcc_threads,
        "nvcc_contract": {
            "std": "c++20",
            "optimization": "O0",
            "arch": "sm_86",
            "split_compile": 1,
            "branch_coverage": branch_coverage,
        },
    }


def validate_persistent_build_manifest(
    manifest_path: Path,
    executable: Path,
    expected: dict[str, Any],
) -> None:
    """Fail closed when a cached differential binary is stale or unprovenanced."""

    if not manifest_path.is_file():
        raise SystemExit(
            "refusing unprovenanced cached Gate-A binary: "
            f"missing {manifest_path}"
        )
    try:
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"refusing invalid Gate-A build manifest {manifest_path}: {error}"
        ) from error
    mismatches = {
        key: {"expected": value, "recorded": recorded.get(key)}
        for key, value in expected.items()
        if recorded.get(key) != value
    }
    actual_binary_hash = sha256_file(executable)
    if recorded.get("executable_sha256") != actual_binary_hash:
        mismatches["executable_sha256"] = {
            "expected": actual_binary_hash,
            "recorded": recorded.get("executable_sha256"),
        }
    if mismatches:
        raise SystemExit(
            "refusing stale Gate-A binary; rebuild required: "
            + json.dumps(mismatches, sort_keys=True)
        )


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def container_workspace_path(path: Path) -> str:
    resolved = path.resolve()
    root = WORKSPACE_ROOT.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes workspace: {path}")
    return "/workspace/" + resolved.relative_to(root).as_posix()


def run_persistent_container(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any] | None]:
    source = (args.source or DEFAULT_SOURCE).resolve()
    rules = (args.rules or DEFAULT_RULES).resolve()
    image = args.image or DEFAULT_IMAGE
    if not source.is_dir() or not rules.is_file():
        raise SystemExit("persistent matrix source or rule pack is missing")

    build_dir = (
        CUDA_ENGINE_ROOT / "build" / "official_battle_ordered_matrix_cuda_paired"
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    case_table = build_dir / "cases.tsv"
    case_table.write_bytes(
        "".join(
            f"{ordinal}\t{container_workspace_path(workspace_path(str(case['deck0'])))}"
            f"\t{container_workspace_path(workspace_path(str(case['deck1'])))}\n"
            for ordinal, case in enumerate(cases, start=1)
        ).encode("utf-8")
    )
    executable = build_dir / "official_battle_ordered_matrix_cuda_paired"
    if args.skip_persistent_build and not executable.is_file():
        raise SystemExit(f"persistent CUDA matrix executable is missing: {executable}")
    build_manifest_path = build_dir / "build_manifest.json"
    build_contract = persistent_build_contract(
        source=source,
        rules=rules,
        image=image,
        branch_coverage=args.branch_coverage,
        nvcc_threads=args.nvcc_threads,
    )
    if args.skip_persistent_build:
        validate_persistent_build_manifest(
            build_manifest_path, executable, build_contract
        )
    coverage_define = (
        "-DPTCG_OFFICIAL_BRANCH_COVERAGE=1 " if args.branch_coverage else ""
    )
    build_command = (
        ""
        if args.skip_persistent_build
        else (
            f"nvcc --threads {args.nvcc_threads} "
            "--split-compile 1 -std=c++20 -O0 -arch=sm_86 "
            + coverage_define
            + "-I/official -I/workspace/engine_cuda_2_0/include "
            "-I/workspace/engine_cuda_2_0/extractor "
            "/workspace/engine_cuda_2_0/benchmarks/"
            "official_battle_ordered_matrix_cuda_paired.cu "
            "/workspace/engine_cuda_2_0/src/official_engine_kernels.cu "
            "-o /build/official_battle_ordered_matrix_cuda_paired; "
        )
    )
    command = (
        "set -euo pipefail; "
        + build_command
        + "/build/official_battle_ordered_matrix_cuda_paired "
        f"{container_workspace_path(rules)} /build/cases.tsv "
        f"{args.seed_start} {args.seed_count} {args.decision_limit} {args.policy}"
    )
    process = subprocess.Popen(
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
            image,
            "-lc",
            command,
        ],
        cwd=WORKSPACE_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    evidence_by_ordinal: dict[int, dict[str, Any]] = {}
    output_tail: list[str] = []
    active_ordinal = 0
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        output_tail.append(line)
        output_tail = output_tail[-40:]
        if line.startswith("START\t"):
            active_ordinal = int(line.split("\t", 1)[1])
            print(
                f"[{active_ordinal}/{len(cases)}] {cases[active_ordinal - 1]['name']}",
                flush=True,
            )
        elif line.startswith("CASE\t"):
            _, ordinal_text, payload = line.split("\t", 2)
            evidence_by_ordinal[int(ordinal_text)] = json.loads(payload)
    returncode = process.wait()
    if returncode == 0:
        if not args.skip_persistent_build:
            completed_manifest = {
                **build_contract,
                "executable_sha256": sha256_file(executable),
            }
            temporary_manifest = build_manifest_path.with_suffix(".json.tmp")
            write_result(temporary_manifest, completed_manifest)
            os.replace(temporary_manifest, build_manifest_path)
        return evidence_by_ordinal, None
    return evidence_by_ordinal, {
        "name": str(cases[active_ordinal - 1]["name"])
        if active_ordinal > 0
        else "persistent_container_build",
        "ordinal": active_ordinal,
        "returncode": returncode,
        "last_error_line": next(
            (line for line in reversed(output_tail) if line.strip()), ""
        ),
    }


def main() -> None:
    args = parse_args()
    if (
        args.seed_start < 0
        or args.seed_count <= 0
        or args.decision_limit <= 0
        or args.nvcc_threads <= 0
    ):
        raise SystemExit("seed and decision arguments must be positive")
    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"matrix manifest does not exist: {manifest_path}")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit("battle matrix manifest contains no cases")
    if manifest.get("competition_private") is not True:
        raise SystemExit("battle matrix requires a competition-private manifest")
    if args.case_name:
        requested = set(args.case_name)
        available = {str(case["name"]) for case in cases}
        missing = sorted(requested - available)
        if missing:
            raise SystemExit(f"requested matrix cases are missing: {missing}")
        cases = [case for case in cases if str(case["name"]) in requested]

    totals = {field: 0 for field in COUNTER_FIELDS}
    set_totals: dict[str, set[int]] = {field: set() for field in SET_FIELDS}
    case_results: list[dict[str, Any]] = []
    first_failure: dict[str, Any] | None = None
    min_decisions: int | None = None
    max_decisions = 0
    build_root = CUDA_ENGINE_ROOT / "build"
    build_root.mkdir(parents=True, exist_ok=True)
    persistent_evidence: dict[int, dict[str, Any]] = {}
    persistent_failure: dict[str, Any] | None = None
    if args.persistent_container:
        persistent_evidence, persistent_failure = run_persistent_container(args, cases)

    with tempfile.TemporaryDirectory(
        prefix="official_battle_matrix_", dir=build_root
    ) as temporary:
        temporary_root = Path(temporary)
        for ordinal, case in enumerate(cases, start=1):
            name = str(case["name"])
            deck0 = workspace_path(str(case["deck0"]))
            deck1 = workspace_path(str(case["deck1"]))
            if not deck0.is_file() or not deck1.is_file():
                raise SystemExit(f"matrix deck is missing for {name}")
            if args.persistent_container:
                if ordinal not in persistent_evidence:
                    first_failure = persistent_failure or {
                        "name": name,
                        "ordinal": ordinal,
                        "returncode": 1,
                        "last_error_line": "persistent result is missing",
                    }
                else:
                    evidence = persistent_evidence[ordinal]
            else:
                child_output = temporary_root / f"case_{ordinal:03d}.json"
                command = [
                    sys.executable,
                    str(BATTLE_TOOL),
                    "--skip-build",
                    "--policy",
                    args.policy,
                    "--deck0",
                    str(deck0),
                    "--deck1",
                    str(deck1),
                    "--seed-start",
                    str(args.seed_start),
                    "--seed-count",
                    str(args.seed_count),
                    "--decision-limit",
                    str(args.decision_limit),
                    "--output",
                    str(child_output),
                ]
                if args.source is not None:
                    command.extend(["--source", str(args.source)])
                if args.rules is not None:
                    command.extend(["--rules", str(args.rules)])
                if args.image is not None:
                    command.extend(["--image", str(args.image)])
                if args.branch_coverage:
                    command.append("--branch-coverage")
                print(f"[{ordinal}/{len(cases)}] {name}", flush=True)
                completed = subprocess.run(
                    command,
                    cwd=WORKSPACE_ROOT,
                    capture_output=True,
                    text=True,
                )
                if completed.returncode != 0:
                    error_lines = [
                        line for line in completed.stderr.splitlines() if line.strip()
                    ]
                    first_failure = {
                        "name": name,
                        "ordinal": ordinal,
                        "returncode": completed.returncode,
                        "last_error_line": error_lines[-1] if error_lines else "",
                    }
                else:
                    evidence = json.loads(child_output.read_text(encoding="utf-8"))
            if first_failure is not None:
                case_results.append(
                    {
                        "name": name,
                        "deck0_name": str(case["deck0_name"]),
                        "deck1_name": str(case["deck1_name"]),
                        "passed": False,
                        "deck0_sha256": sha256_file(deck0),
                        "deck1_sha256": sha256_file(deck1),
                    }
                )
                break
            for field in COUNTER_FIELDS:
                totals[field] += int(evidence.get(field, 0))
            for field in SET_FIELDS:
                set_totals[field].update(
                    int(value) for value in evidence.get(field, [])
                )
            if (
                args.branch_coverage
                and evidence.get("branch_coverage_enabled") is not True
            ):
                raise RuntimeError(
                    f"branch coverage was requested but not enabled for {name}"
                )
            case_min = int(evidence["min_decisions_per_battle"])
            case_max = int(evidence["max_decisions_per_battle"])
            min_decisions = (
                case_min if min_decisions is None else min(min_decisions, case_min)
            )
            max_decisions = max(max_decisions, case_max)
            case_results.append(
                {
                    "name": name,
                    "deck0_name": str(case["deck0_name"]),
                    "deck1_name": str(case["deck1_name"]),
                    "passed": evidence.get("passed") is True,
                    "decisions_compared": int(evidence["decisions_compared"]),
                    "min_decisions_per_battle": case_min,
                    "max_decisions_per_battle": case_max,
                    "state_mismatches": int(evidence["state_mismatches"]),
                    "status_mismatches": int(evidence["status_mismatches"]),
                    "player0_wins": int(evidence.get("player0_wins", 0)),
                    "player1_wins": int(evidence.get("player1_wins", 0)),
                    "draws": int(evidence.get("draws", 0)),
                    "unfinished_battles": int(evidence.get("unfinished_battles", 0)),
                    "outcome_mismatches": int(evidence.get("outcome_mismatches", 0)),
                    "coverage_attack_actions": int(
                        evidence.get("coverage_attack_actions", 0)
                    ),
                    "coverage_non_main_actions": int(
                        evidence.get("coverage_non_main_actions", 0)
                    ),
                    "coverage_zero_cardinality_actions": int(
                        evidence.get("coverage_zero_cardinality_actions", 0)
                    ),
                    "branch_coverage_enabled": evidence.get("branch_coverage_enabled")
                    is True,
                    **{
                        field: sorted(int(value) for value in evidence.get(field, []))
                        for field in SET_FIELDS
                    },
                    "deck0_sha256": sha256_file(deck0),
                    "deck1_sha256": sha256_file(deck1),
                }
            )

    completed_cases = len(case_results) - (1 if first_failure is not None else 0)
    passed = (
        first_failure is None
        and completed_cases == len(cases)
        and all(case["passed"] for case in case_results)
    )
    result: dict[str, Any] = {
        "passed": passed,
        "contract": "official_cpu_reference_cuda_ordered_battle_matrix_v1",
        "policy": args.policy,
        "case_name_filters": list(args.case_name),
        "case_count": len(cases),
        "completed_cases": completed_cases,
        "seed_start": args.seed_start,
        "seeds_per_case": args.seed_count,
        "battles_compared": completed_cases * args.seed_count,
        "min_decisions_per_battle": min_decisions or 0,
        "max_decisions_per_battle": max_decisions,
        "state_mismatches": sum(
            int(case.get("state_mismatches", 0)) for case in case_results
        ),
        "status_mismatches": sum(
            int(case.get("status_mismatches", 0)) for case in case_results
        ),
        "outcome_mismatches": sum(
            int(case.get("outcome_mismatches", 0)) for case in case_results
        ),
        **totals,
        "branch_coverage_enabled": args.branch_coverage,
        **{field: sorted(values) for field, values in set_totals.items()},
        "first_failure": first_failure,
        "manifest_sha256": sha256_file(manifest_path),
        "official_source_snapshot": "3aaeaa92c9eff5272d0c28582a1f64a4cdc2f772",
        "comparison_boundary": (
            "Test-only per-decision Official CPU/reference/CUDA raw state and "
            "status comparison; not a resident hot-path performance claim."
        ),
        "cases": case_results,
    }
    write_result(args.output, result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "cases"}, indent=2
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
