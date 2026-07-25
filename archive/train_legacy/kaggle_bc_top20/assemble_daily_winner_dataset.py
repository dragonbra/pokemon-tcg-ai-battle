"""Assemble audited daily BC shards into one streaming training dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assemble(dataset_root: Path, output: Path) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    output = output.resolve()
    prepare_status = _load(dataset_root / "daily_prepare_status.json")
    if prepare_status.get("status") != "complete":
        raise ValueError("all daily shards must be complete before assembly")
    daily_rows = prepare_status.get("daily_shards") or []
    if len(daily_rows) != 10:
        raise ValueError(f"expected ten daily shards, found {len(daily_rows)}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite assembled dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    split_paths = {
        split: output.with_name(f"{output.stem}.{split}{output.suffix}")
        for split in ("train", "validation", "test")
    }
    if any(path.exists() for path in split_paths.values()):
        raise FileExistsError("refusing to overwrite an assembled split dataset")
    records_by_split: Counter[str] = Counter()
    episodes_by_split: dict[str, set[int]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    decision_keys: set[tuple[int, int, int]] = set()
    trajectory_keys: set[tuple[int, int]] = set()
    deck_counts: Counter[tuple[int, ...]] = Counter()
    card_metadata_hashes: set[str] = set()
    daily_audit: list[dict[str, Any]] = []
    with ExitStack() as stack:
        output_handle = stack.enter_context(output.open("w", encoding="utf-8"))
        split_handles: dict[str, TextIO] = {
            split: stack.enter_context(path.open("w", encoding="utf-8"))
            for split, path in split_paths.items()
        }
        for daily_row in daily_rows:
            date = str(daily_row["date"])
            dataset = Path(daily_row["dataset"])
            observed_hash = _sha256(dataset)
            if observed_hash != daily_row["dataset_sha256"]:
                raise ValueError(f"daily shard hash mismatch: {date}")
            metadata_path = dataset.with_suffix(dataset.suffix + ".card_metadata.json")
            card_metadata_hashes.add(_sha256(metadata_path))
            records = 0
            with dataset.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("expert_team_name") != "Yushin Ito":
                        raise ValueError(f"mixed expert at {date}:{line_number}")
                    if record.get("feature_schema_version") != "ptcg_features_universal":
                        raise ValueError(f"feature schema drift at {date}:{line_number}")
                    split = str(record.get("split"))
                    if split not in split_handles:
                        raise ValueError(f"unknown split at {date}:{line_number}: {split}")
                    expected_split = "validation" if date == "2026-07-22" else "train"
                    if split != expected_split:
                        raise ValueError(f"calendar split drift at {date}:{line_number}")
                    episode_id = int(record["episode_id"])
                    player_index = int(record["player_index"])
                    episode_step = int(record["episode_step"])
                    decision_key = (episode_id, player_index, episode_step)
                    if decision_key in decision_keys:
                        raise ValueError(f"duplicate BC decision: {decision_key}")
                    decision_keys.add(decision_key)
                    trajectory_key = (episode_id, player_index)
                    if trajectory_key not in trajectory_keys:
                        encoded_deck = record["encoded"].get("deck_card_ids") or []
                        deck = tuple(int(card_id) - 1 for card_id in encoded_deck)
                        if len(deck) != 60 or any(card_id < 0 for card_id in deck):
                            raise ValueError(f"invalid encoded deck: {trajectory_key}")
                        deck_counts[deck] += 1
                        trajectory_keys.add(trajectory_key)
                    episodes_by_split[split].add(episode_id)
                    records_by_split[split] += 1
                    records += 1
                    output_handle.write(line)
                    split_handles[split].write(line)
            if records != int(daily_row["records"]):
                raise ValueError(f"daily record count mismatch: {date}")
            daily_audit.append(
                {
                    "date": date,
                    "dataset": str(dataset.resolve()),
                    "dataset_sha256": observed_hash,
                    "records": records,
                    "episodes": int(daily_row["episodes"]),
                    "team_matches": int(daily_row["team_matches"]),
                    "teacher_wins": int(daily_row["teacher_wins"]),
                    "source_episodes_seen": int(daily_row["source_episodes_seen"]),
                    "source_header_errors": int(daily_row["source_header_errors"]),
                }
            )
    if len(card_metadata_hashes) != 1:
        raise ValueError(f"card metadata changed across daily shards: {card_metadata_hashes}")
    if episodes_by_split["train"] & episodes_by_split["validation"]:
        raise ValueError("an Episode appears in both train and validation")
    if not records_by_split["train"] or not records_by_split["validation"]:
        raise ValueError("assembled dataset is missing train or validation records")
    dominant_deck = max(deck_counts, key=lambda deck: (deck_counts[deck], deck))
    deck_path = output.with_suffix(output.suffix + ".dominant_deck.csv")
    deck_path.write_text(
        "\n".join(str(card_id) for card_id in dominant_deck) + "\n",
        encoding="utf-8",
    )
    first_metadata = Path(daily_rows[0]["dataset"]).with_suffix(
        Path(daily_rows[0]["dataset"]).suffix + ".card_metadata.json"
    )
    metadata_path = output.with_suffix(output.suffix + ".card_metadata.json")
    shutil.copy2(first_metadata, metadata_path)
    dataset_hash = _sha256(output)
    split_datasets = {
        split: {
            "path": str(path.resolve()),
            "records": records_by_split[split],
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for split, path in split_paths.items()
    }
    dates = [row["date"] for row in daily_audit]
    manifest = {
        "schema_version": "ptcg_bc_data_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_identity": {
            "selection": "daily_team_name_winner_only",
            "team_name": "Yushin Ito",
            "run_dates": dates,
            "valid_dates": ["2026-07-22"],
            "policy_count": 1,
        },
        "feature_schema": "ptcg_features_universal",
        "action_contract": "full_action_set_v1",
        "dataset": str(output),
        "dataset_sha256": dataset_hash,
        "records": sum(records_by_split.values()),
        "records_by_split": dict(records_by_split),
        "dataset_episode_count": len(trajectory_keys),
        "episodes_by_split": {
            split: len(values) for split, values in episodes_by_split.items()
        },
        "split_datasets": split_datasets,
        "daily_shards": daily_audit,
        "card_metadata": str(metadata_path),
        "card_metadata_sha256": _sha256(metadata_path),
        "dominant_deck": str(deck_path),
        "dominant_deck_sha256": _sha256(deck_path),
        "dominant_deck_trajectories": deck_counts[dominant_deck],
        "distinct_decks": len(deck_counts),
    }
    manifest_path = output.with_suffix(output.suffix + ".data_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": "ptcg_daily_winner_assembled_v1",
        "status": "passed",
        "teacher_team": "Yushin Ito",
        "run_dates": dates,
        "valid_dates": ["2026-07-22"],
        "records": manifest["records"],
        "records_by_split": manifest["records_by_split"],
        "episodes": manifest["dataset_episode_count"],
        "episodes_by_split": manifest["episodes_by_split"],
        "team_matches": sum(row["team_matches"] for row in daily_audit),
        "teacher_wins": sum(row["teacher_wins"] for row in daily_audit),
        "episodes_seen": sum(row["source_episodes_seen"] for row in daily_audit),
        "source_header_errors": sum(row["source_header_errors"] for row in daily_audit),
        "dataset_sha256": dataset_hash,
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": _sha256(manifest_path),
        "split_datasets": split_datasets,
        "distinct_decks": len(deck_counts),
        "dominant_deck_trajectories": deck_counts[dominant_deck],
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.dataset_root, args.output), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
