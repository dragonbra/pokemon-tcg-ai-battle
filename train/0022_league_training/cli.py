"""Command-line contract for the 0022 League framework."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .decks import load_deck_plugins
from .foundation import verify_foundation
from .league import DEFAULT_DECK_ROOT, audit_league_version, initialize_league_version


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="0022 shared-Foundation League Training")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify-foundation")
    validate = subparsers.add_parser("validate-decks")
    validate.add_argument("--deck-root", type=Path, default=DEFAULT_DECK_ROOT)
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--version", required=True)
    initialize.add_argument("--deck-root", type=Path, default=DEFAULT_DECK_ROOT)
    audit = subparsers.add_parser("audit-version")
    audit.add_argument("--version", required=True)
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
    else:
        result = audit_league_version(args.version)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = ["build_parser", "main"]
