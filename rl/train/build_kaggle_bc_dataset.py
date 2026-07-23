"""Build full-action BC records from Kaggle official replay JSON files.

Kaggle stores an agent's action on the following step.  The adapter therefore
pairs ``steps[i].observation`` with ``steps[i + 1].action`` and validates the
action against that observation's ``select.option`` before emitting a record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from rl.core.storage import DEFAULT_MIN_FREE_GIB, DEFAULT_STORAGE_PATH, assert_storage_safe
from rl.model.card_metadata import load_card_metadata, serialize_card_metadata

from rl.model.features import (
    PTCGFeatureConfig,
    UNIVERSAL_FEATURE_SCHEMA,
    encode_observation,
    feature_config_for_schema,
    feature_schema_for_config,
)


DATASET_VERSION = "ptcg_kaggle_bc_v1"
UNIVERSAL_DATASET_VERSION = "ptcg_kaggle_bc_universal"
ACTION_SCHEMA_VERSION = "ptcg_action_set_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_version(feature_config: PTCGFeatureConfig) -> str:
    return (
        UNIVERSAL_DATASET_VERSION
        if feature_config.schema_version == UNIVERSAL_FEATURE_SCHEMA
        else DATASET_VERSION
    )


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("steps"), list):
        raise ValueError(f"not a Kaggle replay: {path}")
    return payload


def _agent_index(payload: dict[str, Any], agent_name: str) -> int:
    agents = ((payload.get("info") or {}).get("Agents") or [])
    names = [str(agent.get("Name", "")) for agent in agents if isinstance(agent, dict)]
    matches = [index for index, name in enumerate(names) if name == agent_name]
    if len(matches) != 1:
        raise ValueError(f"expected one {agent_name!r} agent, found {matches} in {names}")
    return matches[0]


def _episode_deck(payload: dict[str, Any], agent_index: int) -> list[int]:
    """Read the public initial deck encoded in the first visualize frame."""
    for step in payload["steps"]:
        if not isinstance(step, list):
            continue
        for row in step:
            if not isinstance(row, dict):
                continue
            for frame in row.get("visualize") or []:
                if not isinstance(frame, dict):
                    continue
                actions = frame.get("action")
                if (
                    isinstance(actions, list)
                    and len(actions) > agent_index
                    and isinstance(actions[agent_index], list)
                    and len(actions[agent_index]) == 60
                ):
                    return [int(card_id) for card_id in actions[agent_index]]
    raise ValueError("replay does not contain a public initial deck frame")


def _action_for_previous_observation(action: Any, select: dict[str, Any]) -> list[int] | None:
    if not isinstance(action, list):
        return None
    options = select.get("option") or []
    minimum = int(select.get("minCount", 0) or 0)
    maximum = int(select.get("maxCount", 0) or 0)
    if not minimum <= len(action) <= maximum:
        return None
    if len(set(action)) != len(action):
        return None
    if not all(isinstance(index, int) and 0 <= index < len(options) for index in action):
        return None
    # Multi-selection order is not semantic.  Canonical order makes exact-set
    # comparisons and duplicate detection stable across replay exporters.
    return sorted(int(index) for index in action)


def _split_for_episode(episode_id: int, split_map: dict[str, str]) -> str:
    split = split_map.get(str(episode_id))
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"missing split for episode {episode_id}")
    return split


def iter_kaggle_records(
    replay_path: str | Path,
    *,
    agent_name: str | None,
    split_map: dict[str, str],
    submission_id: int,
    feature_config: PTCGFeatureConfig,
    agent_index: int | None = None,
    expert_team_name: str | None = None,
    expert_id: int = 1,
) -> Iterator[dict[str, Any]]:
    path = Path(replay_path)
    payload = _read_json(path)
    if agent_index is None:
        if agent_name is None:
            raise ValueError("agent_name is required when agent_index is not provided")
        agent_index = _agent_index(payload, agent_name)
    if agent_index < 0 or agent_index >= 2:
        raise ValueError(f"invalid player index {agent_index} in {path}")
    episode_id = int((payload.get("info") or {}).get("EpisodeId", path.stem.split("-")[1]))
    # Validate that this is a full replay before using any action labels.
    deck = _episode_deck(payload, agent_index)
    if len(deck) != 60:
        raise ValueError(f"invalid 60-card deck in episode {episode_id}")
    reward_values = payload.get("rewards") or []
    # Some official replay exports leave the terminal reward unset (null).
    # The BC labels are action-only, so an unset outcome is a neutral value.
    raw_outcome = reward_values[agent_index] if len(reward_values) > agent_index else 0.0
    try:
        terminal_outcome = float(raw_outcome) if raw_outcome is not None else 0.0
    except (TypeError, ValueError):
        terminal_outcome = 0.0
    history: list[dict[str, int]] = []
    effect_steps: dict[int, int] = {}
    split = _split_for_episode(episode_id, split_map)
    steps = payload["steps"]
    for step_index in range(max(0, len(steps) - 1)):
        row = steps[step_index][agent_index]
        next_row = steps[step_index + 1][agent_index]
        if not isinstance(row, dict) or row.get("status") != "ACTIVE":
            continue
        # Terminal rows (for example TIMEOUT) carry no submitted action.
        # There is no supervised label for the preceding observation in that
        # case, so leave the terminal decision out of the BC dataset.  Some
        # exports retain a valid action while marking that row DONE, so status
        # alone is not a sufficient filter.
        if not isinstance(next_row, dict) or next_row.get("action") is None:
            continue
        observation = row.get("observation")
        if not isinstance(observation, dict) or not isinstance(observation.get("select"), dict):
            continue
        current = observation.get("current") or {}
        if not isinstance(current, dict) or current.get("yourIndex") != agent_index:
            continue
        select = observation["select"]
        action = _action_for_previous_observation(next_row.get("action"), select)
        if action is None:
            raise ValueError(
                f"next-step action is invalid for episode {episode_id}:{step_index}"
            )
        model_observation = dict(observation)
        model_observation["rl_history"] = list(history)
        model_observation["rl_deck"] = list(deck)
        model_observation["rl_expert_id"] = expert_id
        if feature_config.schema_version == UNIVERSAL_FEATURE_SCHEMA:
            model_observation["rl_card_metadata"] = load_card_metadata()
        effect = select.get("effect") or {}
        effect_serial = _as_int(effect.get("serial"), -1)
        model_observation["rl_effect_step"] = effect_steps.get(effect_serial, 0)
        encoded = encode_observation(model_observation, feature_config)
        options = select.get("option") or []
        if not action or action[0] >= len(options):
            # Empty legal selections carry no candidate target and still train
            # the count head; every non-empty selection is checked by the mask.
            if action:
                raise ValueError(f"action target is outside options in episode {episode_id}")
        elif any(not encoded["action_mask"][index] for index in action):
            raise ValueError(f"action target is masked in episode {episode_id}:{step_index}")
        option = options[action[0]] if action and action[0] < len(options) else {}
        event = {
            "type": _as_int(option.get("type"), 0) if isinstance(option, dict) else 0,
            "cardId": _as_int(option.get("cardId"), 0) if isinstance(option, dict) else 0,
            "attackId": _as_int(option.get("attackId"), 0) if isinstance(option, dict) else 0,
            "targetCount": len(action),
            "selectionType": _as_int(select.get("type"), -1),
            "selectionContext": _as_int(select.get("context"), -1),
            "effectStep": effect_steps.get(effect_serial, 0),
        }
        if feature_config.schema_version == UNIVERSAL_FEATURE_SCHEMA:
            history.append(event)
            del history[:-32]
        elif _as_int(select.get("type"), -1) == 0 and _as_int(select.get("context"), -1) == 0:
            history.append(
                {
                    "type": event["type"],
                    "cardId": event["cardId"],
                    "attackId": event["attackId"],
                }
            )
            del history[:-32]
        if effect_serial >= 0:
            effect_steps[effect_serial] = effect_steps.get(effect_serial, 0) + 1
        yield {
            "dataset_version": _dataset_version(feature_config),
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "feature_schema_version": feature_schema_for_config(feature_config),
            "source": str(path.resolve()),
            "submission_id": submission_id,
            "expert_team_name": expert_team_name or agent_name,
            "episode_id": episode_id,
            "episode_step": step_index,
            "player_index": agent_index,
            "selection_type": _as_int(select.get("type"), -1),
            "selection_context": _as_int(select.get("context"), -1),
            "selection_min_count": int(select.get("minCount", 0) or 0),
            "selection_max_count": int(select.get("maxCount", 0) or 0),
            "targets": action,
            "target_count": len(action),
            "terminal_outcome": terminal_outcome,
            "potential_shaping": {"total": 0.0},
            "split": split,
            "encoded": encoded,
        }


def build_dataset(
    replay_paths: Iterable[Path],
    output: Path,
    *,
    agent_name: str,
    split_map: dict[str, str],
    submission_id: int,
    feature_config: PTCGFeatureConfig,
    storage_path: Path = DEFAULT_STORAGE_PATH,
    min_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> dict[str, Any]:
    storage = assert_storage_safe(storage_path, min_free_gib)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    selection_counts: Counter[str] = Counter()
    target_count_distribution: Counter[str] = Counter()
    episode_splits: Counter[str] = Counter()
    episode_outcomes: Counter[str] = Counter()
    decisions = 0
    replay_count = 0
    with output.open("w", encoding="utf-8") as handle:
        for replay_path in replay_paths:
            replay_count += 1
            replay_split: str | None = None
            replay_outcome: float | None = None
            for record in iter_kaggle_records(
                replay_path,
                agent_name=agent_name,
                split_map=split_map,
                submission_id=submission_id,
                feature_config=feature_config,
                expert_id=1,
            ):
                handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
                decisions += 1
                counts[record["split"]] += 1
                replay_split = str(record["split"])
                replay_outcome = float(record["terminal_outcome"])
                selection_counts[
                    f"type={record['selection_type']},context={record['selection_context']}"
                ] += 1
                target_count_distribution[str(record["target_count"])] += 1
            if replay_split is not None:
                episode_splits[replay_split] += 1
            if replay_outcome is not None:
                if replay_outcome > 0:
                    outcome_name = "win"
                elif replay_outcome < 0:
                    outcome_name = "loss"
                else:
                    outcome_name = "draw"
                episode_outcomes[outcome_name] += 1
    card_metadata_path: Path | None = None
    if feature_config.schema_version == UNIVERSAL_FEATURE_SCHEMA:
        card_metadata_path = output.with_suffix(output.suffix + ".card_metadata.json")
        card_metadata_path.write_text(
            json.dumps(serialize_card_metadata(load_card_metadata()), sort_keys=True) + "\n",
            encoding="utf-8",
        )
    summary = {
        "dataset_version": _dataset_version(feature_config),
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "feature_schema_version": feature_schema_for_config(feature_config),
        "replays": replay_count,
        "records": decisions,
        "records_by_split": dict(sorted(counts.items())),
        "episodes_by_split": dict(sorted(episode_splits.items())),
        "episode_outcomes": dict(sorted(episode_outcomes.items())),
        "records_by_selection": dict(sorted(selection_counts.items())),
        "target_count_distribution": dict(sorted(target_count_distribution.items())),
        "output": str(output.resolve()),
        "storage_path": storage.path,
        "storage_free_gib": round(storage.free_gib, 2),
        "card_metadata": str(card_metadata_path.resolve()) if card_metadata_path else None,
    }
    (output.with_suffix(output.suffix + ".summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_aggregate_dataset(
    replay_root: Path,
    manifest: dict[str, Any],
    output: Path,
    *,
    feature_config: PTCGFeatureConfig,
    storage_path: Path = DEFAULT_STORAGE_PATH,
    min_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> dict[str, Any]:
    """Build one BC corpus from a deduplicated multi-expert replay manifest."""
    storage = assert_storage_safe(storage_path, min_free_gib)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    selection_counts: Counter[str] = Counter()
    target_count_distribution: Counter[str] = Counter()
    episode_outcomes: Counter[str] = Counter()
    source_trajectories: Counter[str] = Counter()
    source_records: Counter[str] = Counter()
    episode_splits: dict[int, str] = {}
    seen_trajectories: set[tuple[int, int]] = set()
    trajectory_decisions: Counter[tuple[int, int]] = Counter()
    decisions = 0

    with output.open("w", encoding="utf-8") as handle:
        for episode in manifest.get("episodes") or []:
            episode_id = int(episode["episode_id"])
            split = str(episode["split"])
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"invalid split for episode {episode_id}: {split}")
            episode_splits[episode_id] = split
            replay_path = replay_root / str(episode["file"])
            if not replay_path.is_file():
                raise FileNotFoundError(f"manifest replay is missing: {replay_path}")
            for expert in episode.get("expert_players") or []:
                player_index = int(expert["player_index"])
                trajectory_key = (episode_id, player_index)
                if trajectory_key in seen_trajectories:
                    continue
                seen_trajectories.add(trajectory_key)
                submission_id = int(expert["submission_id"])
                team_name = str(expert["team_name"])
                source_key = f"{submission_id}:{team_name}"
                source_trajectories[source_key] += 1
                trajectory_outcome: float | None = None
                for record in iter_kaggle_records(
                    replay_path,
                    agent_name=None,
                    agent_index=player_index,
                    expert_team_name=team_name,
                    split_map={str(episode_id): split},
                    submission_id=submission_id,
                    feature_config=feature_config,
                    expert_id=1,
                ):
                    handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
                    decisions += 1
                    trajectory_decisions[trajectory_key] += 1
                    counts[split] += 1
                    source_records[source_key] += 1
                    trajectory_outcome = float(record["terminal_outcome"])
                    selection_counts[
                        f"type={record['selection_type']},context={record['selection_context']}"
                    ] += 1
                    target_count_distribution[str(record["target_count"])] += 1
                if trajectory_outcome is not None:
                    outcome_name = (
                        "win"
                        if trajectory_outcome > 0
                        else "loss"
                        if trajectory_outcome < 0
                        else "draw"
                    )
                    episode_outcomes[outcome_name] += 1

    card_metadata_path: Path | None = None
    if feature_config.schema_version == UNIVERSAL_FEATURE_SCHEMA:
        card_metadata_path = output.with_suffix(output.suffix + ".card_metadata.json")
        card_metadata_path.write_text(
            json.dumps(serialize_card_metadata(load_card_metadata()), sort_keys=True) + "\n",
            encoding="utf-8",
        )
    summary = {
        "dataset_version": _dataset_version(feature_config),
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "feature_schema_version": feature_schema_for_config(feature_config),
        "source_manifest_schema_version": manifest.get("schema_version"),
        "unique_replays": len(episode_splits),
        "expert_trajectories": len(seen_trajectories),
        "records": decisions,
        "records_by_split": dict(sorted(counts.items())),
        "episodes_by_split": dict(sorted(Counter(episode_splits.values()).items())),
        "trajectory_outcomes": dict(sorted(episode_outcomes.items())),
        "trajectories_by_source": dict(sorted(source_trajectories.items())),
        "records_by_source": dict(sorted(source_records.items())),
        "records_by_selection": dict(sorted(selection_counts.items())),
        "target_count_distribution": dict(sorted(target_count_distribution.items())),
        "output": str(output.resolve()),
        "storage_path": storage.path,
        "storage_free_gib": round(storage.free_gib, 2),
        "card_metadata": str(card_metadata_path.resolve()) if card_metadata_path else None,
        "dataset_sha256": _sha256(output),
        "card_metadata_sha256": _sha256(card_metadata_path) if card_metadata_path else None,
    }
    data_episodes: list[dict[str, Any]] = []
    zero_decision_trajectories: list[dict[str, Any]] = []
    for episode in manifest.get("episodes") or []:
        experts: list[dict[str, Any]] = []
        for expert in episode.get("expert_players") or []:
            key = (int(episode["episode_id"]), int(expert["player_index"]))
            decision_records = int(trajectory_decisions[key])
            expert_row = {**expert, "decision_records": decision_records}
            if decision_records:
                experts.append(expert_row)
            else:
                zero_decision_trajectories.append(
                    {
                        "episode_id": key[0],
                        "player_index": key[1],
                        "reason": "replay contains no supervised ACTIVE observation + next action",
                    }
                )
        if experts:
            data_episodes.append(
                {
                    "episode_id": int(episode["episode_id"]),
                    "file": str(episode["file"]),
                    "split": str(episode["split"]),
                    "create_time": episode.get("create_time"),
                    "sha256": episode.get("sha256"),
                    "deck_sha256": episode.get("deck_sha256"),
                    "expert_players": experts,
                }
            )
    data_manifest_path = output.with_suffix(output.suffix + ".data_manifest.json")
    data_manifest = {
        "schema_version": "ptcg_bc_data_manifest_v1",
        "source_manifest_schema_version": manifest.get("schema_version"),
        "source_manifest": manifest.get("source_manifest"),
        "source_manifest_sha256": manifest.get("source_manifest_sha256"),
        "source_identity": manifest.get("source_identity"),
        "feature_schema": feature_schema_for_config(feature_config),
        "action_contract": "full_action_set_v1",
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "dataset": str(output.resolve()),
        "dataset_sha256": summary["dataset_sha256"],
        "card_metadata": summary["card_metadata"],
        "card_metadata_sha256": summary["card_metadata_sha256"],
        "raw_episode_count": len(manifest.get("episodes") or []),
        "dataset_episode_count": len(data_episodes),
        "expert_trajectory_count": len(seen_trajectories),
        "dataset_trajectory_count": len(seen_trajectories) - len(zero_decision_trajectories),
        "zero_decision_trajectories": zero_decision_trajectories,
        "records": decisions,
        "records_by_split": summary["records_by_split"],
        "episodes": data_episodes,
    }
    data_manifest_path.write_text(
        json.dumps(data_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["data_manifest"] = str(data_manifest_path.resolve())
    summary["data_manifest_sha256"] = _sha256(data_manifest_path)
    summary["zero_decision_trajectories"] = len(zero_decision_trajectories)
    (output.with_suffix(output.suffix + ".summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent-name", default="Yushin Ito")
    parser.add_argument("--feature-schema", default="ptcg_features_v6")
    parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE_PATH)
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if any((row.get("expert_players") or []) for row in manifest.get("episodes") or []):
        result = build_aggregate_dataset(
            args.replay_root,
            manifest,
            args.output,
            feature_config=PTCGFeatureConfig(**manifest["feature_config"])
            if manifest.get("feature_config")
            else feature_config_for_schema(args.feature_schema),
            storage_path=args.storage_path,
            min_free_gib=args.min_free_gib,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return
    split_map = {str(item["episode_id"]): str(item["split"]) for item in manifest["episodes"]}
    paths = [args.replay_root / str(item["file"]) for item in manifest["episodes"]]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"manifest replay is missing: {missing[0]}")
    result = build_dataset(
        paths,
        args.output,
        agent_name=args.agent_name,
        split_map=split_map,
        submission_id=int(manifest["submission_id"]),
        feature_config=PTCGFeatureConfig(**manifest["feature_config"])
        if manifest.get("feature_config")
        else feature_config_for_schema(args.feature_schema),
        storage_path=args.storage_path,
        min_free_gib=args.min_free_gib,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
