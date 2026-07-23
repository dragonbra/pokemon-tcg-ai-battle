"""Fail-closed audit for a full-action Kaggle BC JSONL dataset."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _all_finite(value: Any) -> bool:
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _entity_has_card_specific_values(numeric: list[Any]) -> bool:
    """Distinguish card fields from player-level status flags repeated by the encoder."""
    # 0:3 identify owner/zone/slot.  8:13 are the owning player's Active
    # condition flags and remain meaningful even when this particular slot is
    # empty.  HP through tools (3:8), appear/serial/evolution (13:16), and card
    # metadata (16:) require an actual entity card token.
    return any(float(value) != 0.0 for value in numeric[3:8]) or any(
        float(value) != 0.0 for value in numeric[13:]
    )


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
    feature_schemas: collections.Counter[str] = collections.Counter()
    action_schemas: collections.Counter[str] = collections.Counter()
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
            feature_schemas[str(record.get("feature_schema_version"))] += 1
            action_schemas[str(record.get("action_schema_version"))] += 1

            required_feature_schema = source.get("feature_schema")
            if (
                required_feature_schema
                and record.get("feature_schema_version") != required_feature_schema
            ):
                violations["unexpected_feature_schema"] += 1
            if record.get("action_schema_version") != "ptcg_action_set_v1":
                violations["unexpected_action_schema"] += 1
            if int(record.get("selection_type", -1)) < 0:
                violations["invalid_selection_type"] += 1
            if int(record.get("selection_context", -1)) < 0:
                violations["invalid_selection_context"] += 1

            universal = record.get("feature_schema_version") == "ptcg_features_universal"
            card_token_fields = [
                "state_card_ids",
                "action_card_ids",
                "action_target_ids",
            ]
            if universal:
                card_token_fields.extend(
                    ["deck_card_ids", "entity_card_ids", "history_card_ids"]
                )
            for field in card_token_fields:
                tokens = encoded.get(field)
                if not isinstance(tokens, list) or any(
                    not isinstance(token, int) or token < 0 or token >= 4096
                    for token in tokens
                ):
                    violations[f"invalid_{field}"] += 1
            deck_tokens = encoded.get("deck_card_ids") or []
            if universal and (
                len(deck_tokens) != 60 or any(int(token) <= 0 for token in deck_tokens)
            ):
                violations["invalid_nonzero_deck_tokens"] += 1
            action_types = encoded.get("action_type_ids")
            if not isinstance(action_types, list) or len(action_types) != len(mask):
                violations["invalid_action_type_tokens"] += 1
            elif any(
                bool(mask[index]) and (not isinstance(token, int) or token <= 0 or token >= 32)
                for index, token in enumerate(action_types)
            ):
                violations["invalid_nonzero_legal_action_tokens"] += 1
            if universal:
                entity_tokens = encoded.get("entity_card_ids") or []
                entity_numeric = encoded.get("entity_numeric") or []
                if len(entity_tokens) != len(entity_numeric) or any(
                    _entity_has_card_specific_values(numeric)
                    and int(entity_tokens[index]) <= 0
                    for index, numeric in enumerate(entity_numeric)
                ):
                    violations["invalid_nonzero_entity_tokens"] += 1
                expert_id = encoded.get("expert_ids")
                if not isinstance(expert_id, int) or not 0 < expert_id < 64:
                    violations["invalid_expert_token"] += 1
            if not _all_finite(encoded):
                violations["non_finite_feature_value"] += 1

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
    identity = source.get("source_identity") or {}
    if identity:
        expected_submission = int(identity.get("submission_id", 0) or 0)
        observed_submissions = {
            int(key.split(":", 1)[0]) for key in source_records
        }
        if observed_submissions != {expected_submission}:
            violations["source_identity_submission_mismatch"] += 1
    dataset_hash = _sha256(dataset)
    if source.get("dataset_sha256") and source["dataset_sha256"] != dataset_hash:
        violations["dataset_hash_mismatch"] += 1
    risk_flags: list[str] = []
    recommended = {"train": 15000, "validation": 1500, "test": 1500}
    if any(split_records[split] < minimum for split, minimum in recommended.items()):
        risk_flags.append("low_decision_density")
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
        "feature_schemas": dict(sorted(feature_schemas.items())),
        "action_schemas": dict(sorted(action_schemas.items())),
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
        "risk_flags": risk_flags,
        "dataset_sha256": dataset_hash,
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
