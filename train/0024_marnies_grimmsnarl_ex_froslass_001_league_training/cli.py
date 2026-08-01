"""Command-line contract for the 0024 Marnie's Grimmsnarl League framework."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .decks import load_deck_plugins
from .foundation import verify_foundation
from .league import DEFAULT_DECK_ROOT, audit_league_version, initialize_league_version
from .smoke import run_rollout_smoke
from .canary import run_ppo_canary
from .benchmark import run_worker_benchmark
from .training.run import LeagueTrainingConfig, run_training
from .training.league_run import run_league_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="0024 Marnie's Grimmsnarl ex / Froslass League Training"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify-foundation")
    validate = subparsers.add_parser("validate-decks")
    validate.add_argument("--deck-root", type=Path, default=DEFAULT_DECK_ROOT)
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--version", required=True)
    initialize.add_argument("--deck-root", type=Path, default=DEFAULT_DECK_ROOT)
    audit = subparsers.add_parser("audit-version")
    audit.add_argument("--version", required=True)
    smoke = subparsers.add_parser("smoke-rollout")
    smoke.add_argument("--device", default="cuda:0")
    smoke.add_argument("--workers", type=int, default=4)
    smoke.add_argument("--deck-root", type=Path, default=DEFAULT_DECK_ROOT)
    canary = subparsers.add_parser("canary-ppo")
    canary.add_argument("--device", default="cuda:0")
    canary.add_argument("--workers", type=int, default=4)
    canary.add_argument("--deck-root", type=Path, default=DEFAULT_DECK_ROOT)
    benchmark = subparsers.add_parser("benchmark-workers")
    benchmark.add_argument("--device", default="cuda:0")
    benchmark.add_argument("--workers", type=int, required=True)
    benchmark.add_argument("--games", type=int, default=256)
    benchmark.add_argument("--coalesce-ms", type=float, default=5.0)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--deck-root", type=Path, default=DEFAULT_DECK_ROOT)
    train = subparsers.add_parser("train")
    train.add_argument("--version", required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--workers", type=int, default=8)
    train.add_argument("--coalesce-ms", type=float, default=0.5)
    train.add_argument("--games-per-update", type=int, default=512)
    train.add_argument("--duration-hours", type=float)
    train.add_argument("--frozen-eval-interval", type=int, default=5)
    train.add_argument("--deck-root", type=Path, default=DEFAULT_DECK_ROOT)
    league = subparsers.add_parser("train-league")
    league.add_argument("--version", required=True)
    league.add_argument("--device", default="cuda:0")
    league.add_argument("--workers", type=int, default=128)
    league.add_argument("--coalesce-ms", type=float, default=5.0)
    league.add_argument("--games-per-update", type=int, default=512)
    league.add_argument("--frozen-eval-interval", type=int, default=5)
    league.add_argument("--max-updates", type=int)
    league.add_argument("--deck-root", type=Path, default=DEFAULT_DECK_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify-foundation":
        result: object = verify_foundation().as_dict()
    elif args.command == "validate-decks":
        plugins = load_deck_plugins(args.deck_root)
        result = {
            "deck_count": len(plugins),
            "decks": [plugin.snapshot() for plugin in plugins],
            "valid": True,
        }
    elif args.command == "initialize":
        paths = initialize_league_version(args.version, deck_root=args.deck_root)
        result = {
            "run_root": str(paths.run_root),
            "version": paths.version_name,
            "initialized": True,
        }
    elif args.command == "audit-version":
        result = audit_league_version(args.version)
    elif args.command == "smoke-rollout":
        result = run_rollout_smoke(device=args.device, workers=args.workers, deck_root=args.deck_root)
    elif args.command == "canary-ppo":
        result = run_ppo_canary(device=args.device, workers=args.workers, deck_root=args.deck_root)
    elif args.command == "benchmark-workers":
        result = run_worker_benchmark(
            workers=args.workers, games=args.games, coalesce_ms=args.coalesce_ms,
            device=args.device, output=args.output, deck_root=args.deck_root,
        )
    elif args.command == "train-league":
        return run_league_training(LeagueTrainingConfig(
            version=args.version, device=args.device, workers=args.workers,
            coalesce_ms=args.coalesce_ms, games_per_update=args.games_per_update,
            frozen_eval_interval=args.frozen_eval_interval,
        ), deck_root=args.deck_root, max_updates=args.max_updates)
    else:
        return run_training(LeagueTrainingConfig(
            version=args.version, device=args.device, workers=args.workers,
            coalesce_ms=args.coalesce_ms,
            games_per_update=args.games_per_update, duration_hours=args.duration_hours,
            frozen_eval_interval=args.frozen_eval_interval,
        ), deck_root=args.deck_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = ["build_parser", "main"]
