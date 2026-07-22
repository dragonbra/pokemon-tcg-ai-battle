from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl.core.storage import DEFAULT_MIN_FREE_GIB, DEFAULT_STORAGE_PATH

from .dataset import write_behavior_cloning_dataset
from rl.model.features import feature_config_for_schema


DEFAULT_OUTPUT = Path("/tmp/ptcg_bc_dataset.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a legal-candidate Pokémon TCG behavior-cloning JSONL dataset"
    )
    parser.add_argument("traces", nargs="+", type=Path, help="local battle trace JSON files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE_PATH)
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    parser.add_argument(
        "--teacher-player-index",
        default="auto",
        choices=("auto", "0", "1"),
        help="teacher physical player index, or auto from evaluation result metadata",
    )
    parser.add_argument(
        "--include-effect-selections",
        action="store_true",
        help="include single-option effect selections; main actions are the default",
    )
    parser.add_argument(
        "--feature-schema",
        default="ptcg_features_v3",
        choices=(
            "ptcg_features_v1",
            "ptcg_features_v2",
            "ptcg_features_v3",
            "ptcg_features_v4",
            "ptcg_features_v5",
            "ptcg_features_v6",
        ),
    )
    args = parser.parse_args()
    teacher_player_index = (
        None if args.teacher_player_index == "auto" else int(args.teacher_player_index)
    )
    result = write_behavior_cloning_dataset(
        args.traces,
        args.output,
        teacher_player_index=teacher_player_index,
        feature_config=feature_config_for_schema(args.feature_schema),
        include_effect_selections=args.include_effect_selections,
        storage_path=args.storage_path,
        min_free_gib=args.min_free_gib,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
