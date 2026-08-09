from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
CUDA_ROOT = ROOT / "engine_cuda"
STATS = importlib.import_module(
    "train.0038_action_boundary_rl.semantic_parity.statistical_equivalence"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def counts(row: dict, key: str) -> dict[int, int]:
    return {index: int(value) for index, value in enumerate(row[key])}


def analyze(rows: list[dict]) -> dict:
    by_key = {(row["backend"], int(row["logical_batch_size"])): row for row in rows}
    comparisons = []
    overall_pass = True
    for batch in (1, 8, 256, 512):
        cpu = by_key[("official_cpu", batch)]
        cuda = by_key[("cuda", batch)]
        binary = {}
        for field in ("coin_heads", "opening_inclusion", "prize_inclusion"):
            result = STATS.binary_difference_interval(
                int(cpu[field]), int(cpu["samples"]), int(cuda[field]), int(cuda["samples"]), margin=0.01,
            )
            binary[field] = {"estimate": result.estimate, "lower_95": result.lower,
                             "upper_95": result.upper, "margin": result.margin,
                             "passed": result.passed}
        position = STATS.multinomial_total_variation(
            counts(cpu, "card_position_counts"), counts(cuda, "card_position_counts"), margin=0.02,
        )
        target = STATS.multinomial_total_variation(
            counts(cpu, "random_target_counts"), counts(cuda, "random_target_counts"), margin=0.01,
        )
        correlation_bound = 2.5758293035489004 / (int(cpu["samples"]) ** 0.5)
        passed = (
            all(item["passed"] for item in binary.values())
            and position.passed and target.passed
            and abs(float(cpu["coin_lag1"])) <= correlation_bound
            and abs(float(cuda["coin_lag1"])) <= correlation_bound
            and int(cpu["adjacent_fingerprint_duplicates"]) == 0
            and int(cuda["adjacent_fingerprint_duplicates"]) == 0
        )
        overall_pass &= passed
        comparisons.append({
            "logical_batch_size": batch, "passed": passed, "binary": binary,
            "card_position_tv": {"estimate": position.estimate, "upper_95": position.upper_95,
                                 "margin": position.margin, "passed": position.passed},
            "random_target_tv": {"estimate": target.estimate, "upper_95": target.upper_95,
                                  "margin": target.margin, "passed": target.passed},
            "coin_lag1": {"cpu": cpu["coin_lag1"], "cuda": cuda["coin_lag1"],
                          "lower_99": -correlation_bound, "upper_99": correlation_bound},
            "adjacent_fingerprint_duplicates": {
                "cpu": cpu["adjacent_fingerprint_duplicates"], "cuda": cuda["adjacent_fingerprint_duplicates"]},
        })
    batch_stability = []
    for backend in ("official_cpu", "cuda"):
        baseline = by_key[(backend, 1)]
        for batch in (8, 256, 512):
            candidate = by_key[(backend, batch)]
            binary = {}
            for field in ("coin_heads", "opening_inclusion", "prize_inclusion"):
                result = STATS.binary_difference_interval(
                    int(baseline[field]), int(baseline["samples"]),
                    int(candidate[field]), int(candidate["samples"]), margin=0.01,
                )
                binary[field] = {
                    "estimate": result.estimate,
                    "lower_95": result.lower,
                    "upper_95": result.upper,
                    "margin": result.margin,
                    "passed": result.passed,
                }
            position = STATS.multinomial_total_variation(
                counts(baseline, "card_position_counts"),
                counts(candidate, "card_position_counts"),
                margin=0.02,
            )
            target = STATS.multinomial_total_variation(
                counts(baseline, "random_target_counts"),
                counts(candidate, "random_target_counts"),
                margin=0.01,
            )
            passed = (
                all(item["passed"] for item in binary.values())
                and position.passed
                and target.passed
            )
            overall_pass &= passed
            batch_stability.append({
                "backend": backend,
                "baseline_batch_size": 1,
                "candidate_batch_size": batch,
                "passed": passed,
                "binary": binary,
                "card_position_tv": {
                    "estimate": position.estimate,
                    "upper_95": position.upper_95,
                    "margin": position.margin,
                    "passed": position.passed,
                },
                "random_target_tv": {
                    "estimate": target.estimate,
                    "upper_95": target.upper_95,
                    "margin": target.margin,
                    "passed": target.passed,
                },
            })
    return {
        "gate": "E",
        "status": "PASS" if overall_pass else "FAIL",
        "comparisons": comparisons,
        "batch_stability": batch_stability,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1_048_576)
    parser.add_argument("--seed-base", type=int, default=0x003800000000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    build = CUDA_ROOT / "build" / "official_rng_distribution"
    build.mkdir(parents=True, exist_ok=True)
    executable = build / "official_rng_distribution"
    source = CUDA_ROOT / "benchmarks" / "official_rng_distribution.cu"
    if not args.skip_build:
        command = [os.environ.get("NVCC", "/home/cyd/.local/cuda-12.8/bin/nvcc"),
                   "-std=c++20", "-O3", "-Xcompiler", "-fopenmp",
                   "-I", str(CUDA_ROOT / "include"), str(source), "-o", str(executable)]
        subprocess.run(command, check=True)
    started = time.perf_counter()
    completed = subprocess.run([str(executable), "--samples", str(args.samples),
                                "--seed-base", str(args.seed_base)], check=True,
                               capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    result = {"schema_version": "0038_rng_distribution_v1", "samples_per_backend_batch": args.samples,
              "seed_contract": "independent disjoint CPU/CUDA ranges per logical batch",
              "batch_sizes": [1, 8, 256, 512], "elapsed_seconds": elapsed,
              "source_sha256": sha256(source), "executable_sha256": sha256(executable),
              "raw": rows, **analyze(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("gate", "status", "elapsed_seconds", "samples_per_backend_batch", "comparisons")}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
