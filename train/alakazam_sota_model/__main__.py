from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import TrainingConfig
from .dataset import build_id_only_dataset
from .training import train


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Alakazam ID-only SOTA-model reproduction")
    subcommands = parser.add_subparsers(dest="command", required=True)

    build = subcommands.add_parser("build-dataset")
    build.add_argument("--source-dataset", type=Path, required=True)
    build.add_argument("--archive-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--expected-source-sha256")

    run = subcommands.add_parser("train")
    run.add_argument("--dataset-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--resume-checkpoint", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = TrainingConfig.load(args.config)
    if args.command == "build-dataset":
        result = build_id_only_dataset(
            args.source_dataset,
            args.archive_root,
            args.output,
            model_config=config.model,
            expected_source_sha256=args.expected_source_sha256,
        )
    else:
        result = train(
            args.dataset_root,
            args.output,
            config,
            resume_checkpoint=args.resume_checkpoint,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
