"""Build our universal BC records from the exact corpus selected by a daily Kernel.

The source-selection contract intentionally mirrors the reference Yushin
Notebook: scan the requested daily Episode Dataset directories, match
``info.TeamNames`` after whitespace/case normalization, retain only trajectories
whose reward equals the episode maximum, use the longest ``visualize`` trace,
and validate each frame's ``selected`` list against ``obs.select.option``.

Only the feature/model adapter differs: matching frames are encoded with the
repository's versioned ``ptcg_features_universal`` contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

from rl_environment.storage import DEFAULT_MIN_FREE_GIB, DEFAULT_STORAGE_PATH, assert_storage_safe
from train.alakazam_bc_rl.card_metadata import load_card_metadata, serialize_card_metadata
from train.alakazam_bc_rl.features import (
    PTCGFeatureConfig,
    UNIVERSAL_FEATURE_SCHEMA,
    encode_observation,
    feature_config_for_schema,
    feature_schema_for_config,
)
from train.kaggle_bc_top20.training.build_kaggle_bc_dataset import (
    ACTION_SCHEMA_VERSION,
    UNIVERSAL_DATASET_VERSION,
)

try:
    import ijson  # type: ignore
except ImportError:  # pragma: no cover - depends on the Kaggle base image.
    ijson = None


DEFAULT_DATES = (
    "2026-07-13,2026-07-14,2026-07-15,2026-07-16,2026-07-17,"
    "2026-07-18,2026-07-19,2026-07-20,2026-07-21,2026-07-22"
)
DATE_RE = re.compile(r"2026-\d{2}-\d{2}")
SOURCE_SCHEMA = "ptcg_daily_team_winner_source_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return default if value is None else int(value)
    except (TypeError, ValueError):
        return default


def _normalize_team(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _episode_date(path: Path) -> str:
    match = DATE_RE.search(path.as_posix())
    return match.group(0) if match else "unknown"


def _is_winner(rewards: list[Any], index: int) -> bool:
    if not 0 <= index < len(rewards) or not isinstance(rewards[index], (int, float)):
        return False
    numeric = [float(value) for value in rewards if isinstance(value, (int, float))]
    return bool(numeric) and float(rewards[index]) == max(numeric)


def _valid_action(action: Any, select: dict[str, Any]) -> list[int] | None:
    options = _as_list(select.get("option"))
    if not isinstance(action, list) or not all(isinstance(value, int) for value in action):
        return None
    if len(set(action)) != len(action):
        return None
    minimum = max(0, _as_int(select.get("minCount"), 0))
    maximum = max(minimum, _as_int(select.get("maxCount"), len(options)))
    if not minimum <= len(action) <= maximum:
        return None
    if any(value < 0 or value >= len(options) for value in action):
        return None
    return sorted(action)


def _visual_frames(payload: dict[str, Any]) -> list[dict[str, Any]]:
    traces: list[list[Any]] = []
    singletons: list[Any] = []
    for step in _as_list(payload.get("steps")):
        for record in _as_list(step):
            if not isinstance(record, dict):
                continue
            raw = record.get("visualize", record.get("visual"))
            if isinstance(raw, list) and raw:
                traces.append(raw)
            elif isinstance(raw, dict):
                singletons.append(raw)
    selected = max(traces, key=len) if traces else singletons
    return [frame for frame in selected if isinstance(frame, dict)]


def _decks(frames: list[dict[str, Any]]) -> list[list[int]]:
    for frame in frames:
        actions = frame.get("action")
        if (
            isinstance(actions, list)
            and len(actions) >= 2
            and all(isinstance(deck, list) and len(deck) == 60 for deck in actions[:2])
        ):
            return [[int(card_id) for card_id in deck] for deck in actions[:2]]
    raise ValueError("visualize trace has no public two-player 60-card deck frame")


def _submission_id(payload: dict[str, Any], player_index: int) -> int:
    agents = _as_list((payload.get("info") or {}).get("Agents"))
    if not 0 <= player_index < len(agents) or not isinstance(agents[player_index], dict):
        return 0
    agent = agents[player_index]
    return _as_int(agent.get("SubmissionId", agent.get("submissionId")), 0)


def _reference_entity_count(observation: dict[str, Any], actor: int) -> int:
    """Count the entities that the reference ID-only codec would materialize."""
    current = observation.get("current") or {}
    players = _as_list(current.get("players"))
    total = 0

    def add_cards(cards: Any, *, include_children: bool = False) -> None:
        nonlocal total
        for card in _as_list(cards):
            if not isinstance(card, dict):
                if _as_int(card, 0) > 0:
                    total += 1
                continue
            card_id = _as_int(card.get("id", card.get("cardId")), 0)
            if card_id <= 0:
                continue
            total += 1
            if include_children:
                add_cards(card.get("energyCards", card.get("energies")))
                add_cards(card.get("tools"))
                add_cards(card.get("preEvolution"))

    for player_index in (actor, 1 - actor):
        state = (
            players[player_index]
            if 0 <= player_index < len(players) and isinstance(players[player_index], dict)
            else {}
        )
        add_cards(state.get("active"), include_children=True)
        add_cards(state.get("bench"), include_children=True)
        if player_index == actor:
            add_cards(state.get("hand"))
        add_cards(state.get("discard"))
    add_cards(current.get("stadium"))
    add_cards(current.get("looking"))
    select = observation.get("select") or {}
    add_cards(select.get("deck"))
    return total


def _episode_files(roots: Iterable[Path], dates: set[str]) -> Iterator[Path]:
    for root in roots:
        for path in root.rglob("*.json"):
            if _episode_date(path) in dates:
                yield path


def _scan_list(path: Path, prefix: str) -> list[Any]:
    """Read one top-level list without parsing the multi-megabyte replay trace."""
    if ijson is not None:
        with path.open("rb") as handle:
            return _as_list(next(ijson.items(handle, prefix), []))
    # Dependency-free streaming fallback: official replay headers place info
    # and rewards before the very large steps array. Decode only the requested
    # top-level value, growing the prefix until that JSON value is complete.
    top_level, *nested = prefix.split(".")
    marker = re.compile(rf'"{re.escape(top_level)}"\s*:')
    decoder = json.JSONDecoder()
    buffer = ""
    with path.open("r", encoding="utf-8") as handle:
        for _ in range(128):  # At most 8 MiB of header before correctness fallback.
            chunk = handle.read(64 * 1024)
            if not chunk:
                break
            buffer += chunk
            match = marker.search(buffer)
            if match is None:
                continue
            value_start = match.end()
            while value_start < len(buffer) and buffer[value_start].isspace():
                value_start += 1
            if value_start == len(buffer):
                continue
            try:
                value, _ = decoder.raw_decode(buffer, value_start)
            except json.JSONDecodeError:
                continue
            for key in nested:
                value = value.get(key) if isinstance(value, dict) else None
            return _as_list(value)
    # Unusual key ordering or oversized metadata: preserve correctness even if
    # it costs one full parse, and expose the fallback count in skipped/report.
    payload = json.loads(path.read_text(encoding="utf-8"))
    value: Any = payload
    for key in prefix.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    return _as_list(value)


def build_dataset(
    roots: Iterable[Path],
    output: Path,
    *,
    teacher_team: str,
    dates: set[str],
    valid_dates: set[str],
    feature_config: PTCGFeatureConfig,
    storage_path: Path = DEFAULT_STORAGE_PATH,
    min_free_gib: float = DEFAULT_MIN_FREE_GIB,
    write_split_files: bool = False,
    require_train_validation: bool = True,
) -> dict[str, Any]:
    if feature_config.schema_version != UNIVERSAL_FEATURE_SCHEMA:
        raise ValueError("daily winner comparison requires ptcg_features_universal")
    if not dates or not valid_dates <= dates:
        raise ValueError("dates are required and valid_dates must be a subset")
    roots = [path.resolve() for path in roots]
    if not roots or any(not path.is_dir() for path in roots):
        raise FileNotFoundError("all daily Episode roots must exist")
    storage = assert_storage_safe(storage_path, min_free_gib)
    output.parent.mkdir(parents=True, exist_ok=True)
    teacher = _normalize_team(teacher_team)
    card_metadata = load_card_metadata()
    report: dict[str, Any] = {
        "schema_version": SOURCE_SCHEMA,
        "teacher_team": teacher_team,
        "run_dates": sorted(dates),
        "valid_dates": sorted(valid_dates),
        "episodes_seen": 0,
        "team_matches": 0,
        "teacher_wins": 0,
        "records": 0,
        "meta_scan": "ijson" if ijson is not None else "prefix_json",
    }
    skipped: Counter[str] = Counter()
    records_by_split: Counter[str] = Counter()
    records_by_selection: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    episodes: list[dict[str, Any]] = []
    deck_counts: Counter[str] = Counter()
    deck_values: dict[str, list[int]] = {}
    seen_decision_keys: set[tuple[int, int, int]] = set()
    split_paths = {
        split: output.with_name(f"{output.stem}.{split}{output.suffix}")
        for split in ("train", "validation", "test")
    }
    with ExitStack() as stack:
        handle = stack.enter_context(output.open("w", encoding="utf-8"))
        split_handles: dict[str, TextIO] = {}
        if write_split_files:
            split_handles = {
                split: stack.enter_context(path.open("w", encoding="utf-8"))
                for split, path in split_paths.items()
            }
        for path in _episode_files(roots, dates):
            report["episodes_seen"] += 1
            if report["episodes_seen"] == 1 or report["episodes_seen"] % 500 == 0:
                print(
                    json.dumps(
                        {
                            "event": "daily_scan_progress",
                            "episodes_seen": report["episodes_seen"],
                            "team_matches": report["team_matches"],
                            "teacher_wins": report["teacher_wins"],
                            "records": report["records"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            try:
                teams = [str(value) for value in _scan_list(path, "info.TeamNames")]
            except (OSError, json.JSONDecodeError, AttributeError, StopIteration):
                skipped["invalid_team_header"] += 1
                continue
            matching = [
                index for index, team_name in enumerate(teams)
                if _normalize_team(team_name) == teacher
            ]
            report["team_matches"] += len(matching)
            if not matching:
                continue
            try:
                rewards = _scan_list(path, "rewards")
            except (OSError, json.JSONDecodeError, AttributeError, StopIteration):
                skipped["invalid_reward_header"] += len(matching)
                continue
            winners = [index for index in matching if _is_winner(rewards, index)]
            if not winners:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                info = payload.get("info") or {}
                frames = _visual_frames(payload)
                decks = _decks(frames)
            except (OSError, json.JSONDecodeError, AttributeError, ValueError):
                skipped["missing_visualize_or_deck"] += len(winners)
                continue
            episode_id = _as_int(info.get("EpisodeId", path.stem), -1)
            if episode_id < 0:
                skipped["invalid_episode_id"] += len(winners)
                continue
            date = _episode_date(path)
            split = "validation" if date in valid_dates else "train"
            for player_index in winners:
                report["teacher_wins"] += 1
                deck = decks[player_index]
                deck_key = ",".join(str(value) for value in deck)
                deck_counts[deck_key] += 1
                deck_values[deck_key] = deck
                history: list[dict[str, int]] = []
                effect_steps: dict[int, int] = {}
                decision_count = 0
                for frame_index, frame in enumerate(frames):
                    observation = frame.get("obs")
                    if not isinstance(observation, dict):
                        continue
                    current = observation.get("current") or {}
                    if _as_int(current.get("yourIndex"), -1) != player_index:
                        continue
                    select = observation.get("select") or {}
                    if not isinstance(select, dict):
                        continue
                    action = _valid_action(frame.get("selected"), select)
                    if action is None:
                        skipped["invalid_action"] += 1
                        continue
                    options = _as_list(select.get("option"))
                    if (
                        not options
                        or len(options) > 128
                        or len(action) > 16
                        or _reference_entity_count(observation, player_index) > 192
                    ):
                        skipped["reference_codec_limits_or_empty_options"] += 1
                        continue
                    decision_key = (episode_id, player_index, frame_index)
                    if decision_key in seen_decision_keys:
                        raise ValueError(f"duplicate decision key: {decision_key}")
                    seen_decision_keys.add(decision_key)
                    model_observation = dict(observation)
                    model_observation["rl_history"] = list(history)
                    model_observation["rl_deck"] = list(deck)
                    model_observation["rl_expert_id"] = 1
                    model_observation["rl_card_metadata"] = card_metadata
                    effect = select.get("effect") or {}
                    effect_serial = _as_int(effect.get("serial"), -1)
                    model_observation["rl_effect_step"] = effect_steps.get(effect_serial, 0)
                    encoded = encode_observation(model_observation, feature_config)
                    if any(not encoded["action_mask"][index] for index in action):
                        raise ValueError(
                            f"reference action is masked in episode {episode_id}:{frame_index}"
                        )
                    option = options[action[0]] if action else {}
                    event = {
                        "type": _as_int(option.get("type"), 0)
                        if isinstance(option, dict) else 0,
                        "cardId": _as_int(option.get("cardId"), 0)
                        if isinstance(option, dict) else 0,
                        "attackId": _as_int(option.get("attackId"), 0)
                        if isinstance(option, dict) else 0,
                        "targetCount": len(action),
                        "selectionType": _as_int(select.get("type"), -1),
                        "selectionContext": _as_int(select.get("context"), -1),
                        "effectStep": effect_steps.get(effect_serial, 0),
                    }
                    history.append(event)
                    del history[:-32]
                    if effect_serial >= 0:
                        effect_steps[effect_serial] = effect_steps.get(effect_serial, 0) + 1
                    record = {
                        "dataset_version": UNIVERSAL_DATASET_VERSION,
                        "action_schema_version": ACTION_SCHEMA_VERSION,
                        "feature_schema_version": feature_schema_for_config(feature_config),
                        "source": str(path),
                        "submission_id": _submission_id(payload, player_index),
                        "expert_team_name": teacher_team,
                        "episode_id": episode_id,
                        "episode_step": frame_index,
                        "player_index": player_index,
                        "selection_type": event["selectionType"],
                        "selection_context": event["selectionContext"],
                        "selection_min_count": _as_int(select.get("minCount"), 0),
                        "selection_max_count": _as_int(
                            select.get("maxCount"), len(options)
                        ),
                        "targets": action,
                        "target_count": len(action),
                        "terminal_outcome": float(rewards[player_index]),
                        "potential_shaping": {"total": 0.0},
                        "split": split,
                        "encoded": encoded,
                    }
                    line = json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
                    handle.write(line)
                    if write_split_files:
                        split_handles[split].write(line)
                    decision_count += 1
                    report["records"] += 1
                    records_by_split[split] += 1
                    records_by_selection[
                        f"type={event['selectionType']},context={event['selectionContext']}"
                    ] += 1
                    target_counts[str(len(action))] += 1
                if decision_count:
                    episodes.append(
                        {
                            "episode_id": episode_id,
                            "date": date,
                            "split": split,
                            "file": str(path),
                            "sha256": _sha256(path),
                            "expert_players": [
                                {
                                    "player_index": player_index,
                                    "team_name": teacher_team,
                                    "submission_id": _submission_id(payload, player_index),
                                    "decision_records": decision_count,
                                }
                            ],
                        }
                    )
    if not report["records"]:
        raise ValueError("reference corpus produced no records")
    if require_train_validation and (
        not records_by_split["train"] or not records_by_split["validation"]
    ):
        raise ValueError("reference corpus produced no train or validation records")
    card_metadata_path = output.with_suffix(output.suffix + ".card_metadata.json")
    card_metadata_path.write_text(
        json.dumps(serialize_card_metadata(card_metadata), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dataset_hash = _sha256(output)
    dominant_deck = max(deck_counts, key=lambda key: (deck_counts[key], key))
    deck_path = output.with_suffix(output.suffix + ".dominant_deck.csv")
    deck_path.write_text(
        "\n".join(str(value) for value in deck_values[dominant_deck]) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "ptcg_bc_data_manifest_v1",
        "source_manifest_schema_version": SOURCE_SCHEMA,
        "source_identity": {
            "selection": "daily_team_name_winner_only",
            "team_name": teacher_team,
            "run_dates": sorted(dates),
            "valid_dates": sorted(valid_dates),
        },
        "feature_schema": feature_schema_for_config(feature_config),
        "action_contract": "full_action_set_v1",
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "dataset": str(output.resolve()),
        "dataset_sha256": dataset_hash,
        "card_metadata": str(card_metadata_path.resolve()),
        "card_metadata_sha256": _sha256(card_metadata_path),
        "raw_episode_count": len(episodes),
        "dataset_episode_count": len(episodes),
        "expert_trajectory_count": len(episodes),
        "dataset_trajectory_count": len(episodes),
        "zero_decision_trajectories": [],
        "records": report["records"],
        "records_by_split": dict(sorted(records_by_split.items())),
        "episodes": episodes,
    }
    if write_split_files:
        manifest["split_datasets"] = {
            split: {
                "path": str(path.resolve()),
                "records": records_by_split[split],
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for split, path in split_paths.items()
        }
    manifest_path = output.with_suffix(output.suffix + ".data_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report.update(
        {
            "feature_schema_version": feature_schema_for_config(feature_config),
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "dataset_sha256": dataset_hash,
            "records_by_split": dict(sorted(records_by_split.items())),
            "records_by_selection": dict(sorted(records_by_selection.items())),
            "target_count_distribution": dict(sorted(target_counts.items())),
            "skipped": dict(sorted(skipped.items())),
            "episodes": len(episodes),
            "distinct_decks": len(deck_counts),
            "dominant_deck_trajectories": deck_counts[dominant_deck],
            "dominant_deck": str(deck_path.resolve()),
            "dominant_deck_sha256": _sha256(deck_path),
            "data_manifest": str(manifest_path.resolve()),
            "data_manifest_sha256": _sha256(manifest_path),
            "split_datasets": manifest.get("split_datasets", {}),
            "storage_path": storage.path,
            "storage_free_gib": round(storage.free_gib, 2),
        }
    )
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-team", default="Yushin Ito")
    parser.add_argument("--run-dates", default=DEFAULT_DATES)
    parser.add_argument("--valid-dates", default="")
    parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE_PATH)
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    parser.add_argument(
        "--write-split-files",
        action="store_true",
        help="also write dataset.train/validation/test.jsonl for bounded-memory training",
    )
    parser.add_argument(
        "--allow-single-split",
        action="store_true",
        help="allow one daily shard to contain only train or only validation records",
    )
    args = parser.parse_args()
    dates = {value.strip() for value in args.run_dates.split(",") if value.strip()}
    valid_dates = {
        value.strip() for value in args.valid_dates.split(",") if value.strip()
    }
    if not valid_dates and not args.allow_single_split:
        valid_dates = {max(dates)}
    result = build_dataset(
        args.input_root,
        args.output,
        teacher_team=args.teacher_team,
        dates=dates,
        valid_dates=valid_dates,
        feature_config=feature_config_for_schema(UNIVERSAL_FEATURE_SCHEMA),
        storage_path=args.storage_path,
        min_free_gib=args.min_free_gib,
        write_split_files=args.write_split_files,
        require_train_validation=not args.allow_single_split,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
