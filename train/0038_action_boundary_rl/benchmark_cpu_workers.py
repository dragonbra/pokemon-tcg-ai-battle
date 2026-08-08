"""Calibrate official CPU rollout workers under one fixed full-semantic contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .training.run_full_semantic import run_gate


def _row(
    report: dict,
    worker_processes: int,
    engines_per_worker: int,
    inference_channels_per_role: int,
    games: int,
    coalesce_ms: float,
) -> dict:
    wall = float(report["rollout_seconds"])
    metrics = report["metrics"]
    return {
        "worker_processes": worker_processes,
        "engines_per_worker": engines_per_worker,
        "inference_channels_per_role": inference_channels_per_role,
        "coalesce_ms": coalesce_ms,
        "games": games,
        "wall_seconds": wall,
        "games_per_second": games / wall,
        "engine_selections": metrics["rollout/engine_selections"],
        "engine_selections_per_second": metrics["rollout/engine_selections"] / wall,
        "focal_decisions_per_second": metrics["rollout/decisions"] / wall,
        "max_inference_batch": metrics["rollout/max_batch_size"],
        "error_games": metrics["rollout/error_games"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, nargs="+", default=(8, 16, 32))
    parser.add_argument("--engines-per-worker", type=int, default=1)
    parser.add_argument("--inference-channels-per-role", type=int, default=1)
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--coalesce-ms", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--single-worker-run", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.single_worker_run:
        if len(args.workers) != 1:
            parser.error("--single-worker-run requires exactly one worker value")
        report = run_gate(
            output=args.output,
            games=args.games,
            worker_processes=args.workers[0],
            engines_per_worker=args.engines_per_worker,
            inference_channels_per_role=args.inference_channels_per_role,
            run_ppo=False,
            mode="greedy",
            coalesce_ms=args.coalesce_ms,
        )
        print(json.dumps(_row(
            report,
            args.workers[0],
            args.engines_per_worker,
            args.inference_channels_per_role,
            args.games,
            args.coalesce_ms,
        ), sort_keys=True))
        return 0

    rows = []
    for workers in args.workers:
        report_path = args.output.parent / f"workers_{workers}.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "train.0038_action_boundary_rl.benchmark_cpu_workers",
                "--single-worker-run",
                "--workers",
                str(workers),
                "--games",
                str(args.games),
                "--engines-per-worker",
                str(args.engines_per_worker),
                "--inference-channels-per-role",
                str(args.inference_channels_per_role),
                "--coalesce-ms",
                str(args.coalesce_ms),
                "--output",
                str(report_path),
            ],
            check=True,
        )
        row = _row(
            json.loads(report_path.read_text()),
            workers,
            args.engines_per_worker,
            args.inference_channels_per_role,
            args.games,
            args.coalesce_ms,
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    valid = [row for row in rows if row["error_games"] == 0]
    selected = max(valid, key=lambda row: row["games_per_second"])
    payload = {
        "schema": "0038_rl_engine_pool_benchmark_v1",
        "fixed_games": args.games,
        "rows": rows,
        "selected_workers": selected["workers"],
        "selected_coalesce_ms": selected["coalesce_ms"],
        "selection_metric": "games_per_second_with_zero_error",
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected_workers": selected["workers"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
