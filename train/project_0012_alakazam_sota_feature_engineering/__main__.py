from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig
from .dataset import build_feature_dataset
from .training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Alakazam SOTA feature-engineering BC")
    parser.add_argument("command", choices=("train", "build-dataset"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--source-dataset", type=Path)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    if args.command == "build-dataset":
        if args.source_dataset is None or args.archive_root is None:
            parser.error("build-dataset requires --source-dataset and --archive-root")
        if not args.expected_source_sha256:
            parser.error("build-dataset requires --expected-source-sha256")
        result = build_feature_dataset(
            args.source_dataset,
            args.archive_root,
            args.output,
            model_config=config.model,
            expected_source_sha256=args.expected_source_sha256,
        )
    else:
        if args.dataset_root is None:
            parser.error("train requires --dataset-root")
        result = train(args.dataset_root, args.output, config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
