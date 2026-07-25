from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig
from .dataset import build_reward_dataset
from .export_candidate import export_candidate
from .training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="0011 Alakazam SOTA reward-weighted BC")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-dataset")
    build.add_argument("--base-root", type=Path, required=True)
    build.add_argument("--reward-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)

    training = subparsers.add_parser("train")
    training.add_argument("--dataset-root", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--config", type=Path, required=True)

    export = subparsers.add_parser("export-candidate")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--source-package", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "build-dataset":
        result = build_reward_dataset(args.base_root, args.reward_root, args.output)
    elif args.command == "train":
        result = train(
            args.dataset_root,
            args.output,
            ExperimentConfig.load(args.config),
        )
    else:
        result = export_candidate(args.checkpoint, args.source_package, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
