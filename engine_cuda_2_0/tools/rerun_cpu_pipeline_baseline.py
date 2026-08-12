from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun a recorded CPU-engine/GPU-policy baseline with tuned workers."
    )
    parser.add_argument("--source-provenance", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine-threads", type=int, required=True)
    parser.add_argument("--opponent-workers", type=int, required=True)
    parser.add_argument(
        "--games-per-iter",
        type=int,
        help="Override the recorded number of games for a longer stability run.",
    )
    parser.add_argument(
        "--rollout-envs",
        type=int,
        help="Override the recorded number of concurrent rollout environments.",
    )
    parser.add_argument(
        "--shard-same-opponent", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--actor-profile", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def replace_option(command: list[str], name: str, value: str) -> None:
    try:
        index = command.index(name)
    except ValueError as error:
        raise RuntimeError(f"recorded command is missing {name}") from error
    if index + 1 >= len(command):
        raise RuntimeError(f"recorded command has no value for {name}")
    command[index + 1] = value


def set_boolean_flag(command: list[str], positive: str, negative: str, value: bool) -> None:
    command[:] = [item for item in command if item not in {positive, negative}]
    command.append(positive if value else negative)


def main() -> None:
    args = parse_args()
    positive_counts = {
        "engine_threads": args.engine_threads,
        "opponent_workers": args.opponent_workers,
        "games_per_iter": args.games_per_iter,
        "rollout_envs": args.rollout_envs,
    }
    invalid_counts = [
        name
        for name, value in positive_counts.items()
        if value is not None and value <= 0
    ]
    if invalid_counts:
        raise SystemExit(f"counts must be positive: {', '.join(invalid_counts)}")
    payload = json.loads(args.source_provenance.read_text(encoding="utf-8"))
    command = [str(value) for value in payload["command_line"]]
    if not command or command[0] != "tools/train_pure_lucario_ppo.py":
        raise RuntimeError("source provenance is not a pure PPO trainer command")
    replace_option(command, "--out", str(args.out.resolve()))
    replace_option(command, "--engine-threads", str(args.engine_threads))
    replace_option(command, "--opponent-policy-workers", str(args.opponent_workers))
    if args.games_per_iter is not None:
        replace_option(command, "--games-per-iter", str(args.games_per_iter))
    if args.rollout_envs is not None:
        replace_option(command, "--rollout-envs", str(args.rollout_envs))
    replace_option(command, "--checkpoint-mode", "none")
    set_boolean_flag(
        command,
        "--opponent-policy-shard-same-agent",
        "--no-opponent-policy-shard-same-agent",
        args.shard_same_opponent,
    )
    set_boolean_flag(
        command,
        "--opponent-actor-profile",
        "--no-opponent-actor-profile",
        args.actor_profile,
    )
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(args.engine_threads)
    environment.setdefault("MKL_NUM_THREADS", "2")
    environment.setdefault("OPENBLAS_NUM_THREADS", "2")
    print(
        json.dumps(
            {
                "event": "cpu_pipeline_baseline_command",
                "engine_threads": args.engine_threads,
                "opponent_workers": args.opponent_workers,
                "games_per_iter": args.games_per_iter,
                "rollout_envs": args.rollout_envs,
                "shard_same_opponent": args.shard_same_opponent,
                "command": [sys.executable, *command],
            }
        ),
        flush=True,
    )
    completed = subprocess.run(
        [sys.executable, *command],
        cwd=WORKSPACE_ROOT,
        env=environment,
        check=False,
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
