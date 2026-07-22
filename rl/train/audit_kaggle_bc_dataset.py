"""Fail-closed audit for a full-action Kaggle BC JSONL dataset."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def audit(
    dataset: Path,
    manifest: Path,
    *,
    require_single_expert: bool = False,
) -> dict[str, Any]:
    source = json.loads(manifest.read_text(encoding="utf-8"))
    expected_episodes = {int(row["episode_id"]) for row in source["episodes"]}
    expected_trajectories: set[tuple[int, int]] = set()
    for row in source["episodes"]:
        experts = row.get("expert_players") or []
        if experts:
            expected_trajectories.update(
                (int(row["episode_id"]), int(player["player_index"]))
                for player in experts
            )
        elif "agent_index" in row:
            expected_trajectories.add((int(row["episode_id"]), int(row["agent_index"])))
    episode_splits: dict[int, set[str]] = collections.defaultdict(set)
    decision_keys: set[tuple[int, int, int]] = set()
    trajectories: set[tuple[int, int]] = set()
    violations: collections.Counter[str] = collections.Counter()
    split_records: collections.Counter[str] = collections.Counter()
    target_counts: collections.Counter[str] = collections.Counter()
    selection_counts: collections.Counter[str] = collections.Counter()
    multi_by_selection: collections.Counter[str] = collections.Counter()
    source_records: collections.Counter[str] = collections.Counter()
    dimension_contract: tuple[int, ...] | None = None
    max_target_count = 0
    records = 0
    empty_selections = 0
    positive_multi_selections = 0

    with dataset.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            records += 1
            episode_id = int(record["episode_id"])
            player_index = int(record["player_index"])
            episode_step = int(record["episode_step"])
            split = str(record["split"])
            key = (episode_id, player_index, episode_step)
            trajectory = (episode_id, player_index)
            if key in decision_keys:
                violations["duplicate_decision_key"] += 1
            decision_keys.add(key)
            trajectories.add(trajectory)
            episode_splits[episode_id].add(split)
            split_records[split] += 1
            if split not in {"train", "validation", "test"}:
                violations["invalid_split"] += 1

            targets = record.get("targets")
            encoded = record.get("encoded")
            if not isinstance(targets, list) or not isinstance(encoded, dict):
                violations["malformed_record"] += 1
                continue
            mask = encoded.get("action_mask")
            if not isinstance(mask, list) or not mask or not any(mask):
                violations["missing_action_mask"] += 1
                continue
            legal_count = sum(bool(value) for value in mask)
            minimum = int(record.get("selection_min_count", -1))
            maximum = int(record.get("selection_max_count", -1))
            target_count = int(record.get("target_count", -1))
            if target_count != len(targets):
                violations["target_count_mismatch"] += 1
            if len(set(targets)) != len(targets):
                violations["duplicate_target"] += 1
            if targets != sorted(targets):
                violations["noncanonical_target_order"] += 1
            if not 0 <= minimum <= len(targets) <= maximum <= legal_count:
                violations["cardinality_outside_legal_range"] += 1
            if any(
                not isinstance(target, int)
                or target < 0
                or target >= len(mask)
                or not mask[target]
                for target in targets
            ):
                violations["illegal_or_masked_target"] += 1
            max_target_count = max(max_target_count, len(targets))
            target_counts[str(len(targets))] += 1
            selection = (
                f"type={record.get('selection_type')},"
                f"context={record.get('selection_context')}"
            )
            selection_counts[selection] += 1
            if len(targets) == 0:
                empty_selections += 1
                multi_by_selection[selection] += 1
            elif len(targets) > 1:
                positive_multi_selections += 1
                multi_by_selection[selection] += 1
            source_records[
                f"{record.get('submission_id')}:{record.get('expert_team_name')}"
            ] += 1

            current_dimensions = (
                len(encoded.get("state_numeric") or []),
                len(encoded.get("state_card_ids") or []),
                len(mask),
                len((encoded.get("action_numeric") or [[None]])[0]),
                len(encoded.get("deck_card_ids") or []),
                len((encoded.get("deck_card_numeric") or [[None]])[0]),
                len(encoded.get("entity_card_ids") or []),
                len((encoded.get("entity_numeric") or [[None]])[0]),
                len(encoded.get("history_card_ids") or []),
                len((encoded.get("history_numeric") or [[None]])[0]),
            )
            if dimension_contract is None:
                dimension_contract = current_dimensions
            elif current_dimensions != dimension_contract:
                violations["inconsistent_feature_dimensions"] += 1

    for splits in episode_splits.values():
        if len(splits) != 1:
            violations["episode_split_leakage"] += 1
    if set(episode_splits) != expected_episodes:
        violations["manifest_episode_coverage_mismatch"] += 1
    if trajectories != expected_trajectories:
        violations["manifest_trajectory_coverage_mismatch"] += 1
    if require_single_expert and len(source_records) != 1:
        violations["multiple_expert_sources"] += 1
    report = {
        "status": "passed" if not violations else "failed",
        "dataset": str(dataset.resolve()),
        "manifest": str(manifest.resolve()),
        "records": records,
        "unique_decision_keys": len(decision_keys),
        "unique_episodes": len(episode_splits),
        "expert_trajectories": len(trajectories),
        "records_by_split": dict(sorted(split_records.items())),
        "records_by_source": dict(sorted(source_records.items())),
        "records_by_selection": dict(sorted(selection_counts.items())),
        "target_count_distribution": dict(sorted(target_counts.items())),
        "empty_selections": empty_selections,
        "positive_multi_selections": positive_multi_selections,
        "multi_by_selection": dict(sorted(multi_by_selection.items())),
        "max_target_count": max_target_count,
        "feature_dimensions": {
            "state_numeric": dimension_contract[0] if dimension_contract else 0,
            "state_tokens": dimension_contract[1] if dimension_contract else 0,
            "max_candidates": dimension_contract[2] if dimension_contract else 0,
            "candidate_numeric": dimension_contract[3] if dimension_contract else 0,
            "deck_tokens": dimension_contract[4] if dimension_contract else 0,
            "deck_numeric": dimension_contract[5] if dimension_contract else 0,
            "entity_tokens": dimension_contract[6] if dimension_contract else 0,
            "entity_numeric": dimension_contract[7] if dimension_contract else 0,
            "history_tokens": dimension_contract[8] if dimension_contract else 0,
            "history_numeric": dimension_contract[9] if dimension_contract else 0,
        },
        "violations": dict(sorted(violations.items())),
    }
    if violations:
        raise ValueError(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-multiple-experts", action="store_true")
    args = parser.parse_args()
    try:
        report = audit(
            args.dataset,
            args.manifest,
            require_single_expert=not args.allow_multiple_experts,
        )
    except ValueError as exc:
        try:
            report = json.loads(str(exc))
        except json.JSONDecodeError:
            raise
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(1) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
