"""Thin command surface for staged project implementation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from . import PROJECT_ID
from .model.registry import MODEL_REGISTRY
from .protocol import PreRunProtocol

PROTOCOL_PATH = Path(__file__).with_name("configs") / "pre_run_protocol.json"
FORMAL_COMMANDS = frozenset({"build-dataset", "train", "schedule", "select-project", "export", "evaluate"})

COMMANDS = (
    "validate-protocol",
    "audit-visibility",
    "build-dataset",
    "smoke-model",
    "train",
    "schedule",
    "select-project",
    "export",
    "evaluate",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"python3 -m train.{PROJECT_ID}",
        description="Semantic Goal Policy staged workflow",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        command_parser = subparsers.add_parser(command, help=f"Run the {command} stage")
        if command in FORMAL_COMMANDS:
            command_parser.add_argument("--protocol-sha256", required=True)
        if command == "train":
            command_parser.add_argument("--m0-smoke-throughput", type=float)
            command_parser.add_argument("--variant", choices=tuple(MODEL_REGISTRY), default="M0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command not in FORMAL_COMMANDS and args.command != "validate-protocol":
        parser.error(f"{args.command} is not implemented in the scaffold")
    try:
        protocol = PreRunProtocol.from_json(PROTOCOL_PATH)
    except (OSError, ValueError) as error:
        parser.error(f"invalid pre-run protocol: {error}")
    if args.command == "validate-protocol":
        print(protocol.sha256())
        return 0
    if args.command in FORMAL_COMMANDS and args.protocol_sha256 != protocol.sha256():
        parser.error("protocol hash mismatch")
    if args.command == "train":
        authorized_variants = protocol.data["minimum_formal_allocation"]["models"]
        if args.variant not in authorized_variants:
            parser.error(
                f"variant {args.variant} is not authorized by the bound formal protocol"
            )
        if args.m0_smoke_throughput is None:
            parser.error("train requires --m0-smoke-throughput")
        try:
            protocol.resolve_runtime_floor(args.m0_smoke_throughput)
        except (TypeError, ValueError) as error:
            parser.error(str(error))
    parser.error(f"{args.command} is not implemented in the scaffold")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
