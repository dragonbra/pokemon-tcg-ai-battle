"""CLI for 0030 smoke, benchmark, and versioned PPO training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import run as run_benchmark, run_dual_evaluation
from .smoke import run_smoke
from .training.run import RunConfig, run, run_smoke_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--games", type=int, default=4)
    smoke.add_argument("--workers", type=int, default=2)
    smoke.add_argument("--device", default="cuda:0")
    smoke.add_argument(
        "--opponent-foundation", choices=("0019", "0028", "mixed"), default="0028"
    )
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--games", type=int, default=16)
    benchmark.add_argument("--workers", type=int, default=8)
    benchmark.add_argument("--device", default="cuda:0")
    benchmark.add_argument("--initialization-checkpoint", type=Path)
    benchmark.add_argument(
        "--opponent-foundation",
        choices=("0019", "0028", "mixed"),
        default="mixed",
    )
    benchmark.add_argument(
        "--output",
        type=Path,
        default=Path(".tmp/evaluation/0030_performance/performance_gate.json"),
    )
    dual_eval = commands.add_parser("dual-eval")
    dual_eval.add_argument("--initialization-checkpoint", type=Path, required=True)
    dual_eval.add_argument("--workers", type=int, default=128)
    dual_eval.add_argument("--device", default="cuda:0")
    dual_eval.add_argument("--seed", type=int, default=20260804)
    dual_eval.add_argument(
        "--output",
        type=Path,
        default=Path(".tmp/evaluation/0030_dual_foundation_smoke/result.json"),
    )
    smoke_update = commands.add_parser("smoke-update")
    smoke_update.add_argument("--version", required=True)
    smoke_update.add_argument("--games", type=int, default=4)
    smoke_update.add_argument("--workers", type=int, default=2)
    smoke_update.add_argument("--device", default="cuda:0")
    train = commands.add_parser("train")
    train.add_argument("--version", required=True)
    train.add_argument("--updates", type=int, default=10_000)
    train.add_argument("--games-per-update", type=int, default=512)
    train.add_argument("--workers", type=int, default=64)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--seed", type=int, default=20260802)
    train.add_argument("--eval-every", type=int, default=5)
    train.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="online")
    train.add_argument("--initialization-checkpoint")
    args = parser.parse_args()
    if args.command == "smoke":
        result = run_smoke(
            games=args.games,
            workers=args.workers,
            device=args.device,
            opponent_foundation=args.opponent_foundation,
        )
    elif args.command == "benchmark":
        result = run_benchmark(
            games=args.games,
            workers=args.workers,
            device_name=args.device,
            output=args.output,
            initialization_checkpoint=args.initialization_checkpoint,
            opponent_foundation=args.opponent_foundation,
        )
    elif args.command == "dual-eval":
        result = run_dual_evaluation(
            initialization_checkpoint=args.initialization_checkpoint,
            workers=args.workers,
            device_name=args.device,
            seed=args.seed,
            output=args.output,
        )
    elif args.command == "smoke-update":
        result = run_smoke_gate(
            version=args.version, games=args.games,
            workers=args.workers, device_name=args.device,
        )
    else:
        result = run(RunConfig(
            version=args.version, updates=args.updates,
            games_per_update=args.games_per_update, workers=args.workers,
            device=args.device, seed=args.seed, eval_every=args.eval_every,
            wandb_mode=args.wandb_mode,
            initialization_checkpoint=args.initialization_checkpoint,
        ))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = ["main"]
