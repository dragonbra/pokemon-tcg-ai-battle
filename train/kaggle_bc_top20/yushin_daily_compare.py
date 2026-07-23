"""Build paired model views of the Yushin Ito daily winner-only corpus.

The source selection intentionally mirrors the reference Kaggle Notebook:
scan official daily Episode JSON files, match ``info.TeamNames`` exactly after
case/whitespace normalization, and keep only Episodes won by Yushin Ito.  A
decision is admitted only when both the reference ID-only codec and the
repository universal codec can represent it.  Both output views therefore
share the exact same ``decision_id`` set and train/validation/test split.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import ijson  # type: ignore
except ImportError:  # pragma: no cover - exercised in minimal local environments
    ijson = None

from train.alakazam_bc_rl.card_metadata import load_card_metadata, serialize_card_metadata
from train.alakazam_bc_rl.features import (
    UNIVERSAL_FEATURE_SCHEMA,
    encode_observation,
    feature_config_for_schema,
)
from train.alakazam_bc_rl.id_only_pointer import IDOnlyCodec, IDOnlyConfig, valid_reference_action


SCHEMA_VERSION = "ptcg_yushin_daily_common_decisions_v1"
TEACHER_TEAM = "Yushin Ito"
TRAIN_DATES = tuple(f"2026-07-{day:02d}" for day in range(13, 21))
VALIDATION_DATES = ("2026-07-21",)
TEST_DATES = ("2026-07-22",)
ALL_DATES = frozenset((*TRAIN_DATES, *VALIDATION_DATES, *TEST_DATES))
DATE_RE = re.compile(r"2026-\d{2}-\d{2}")


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_team(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def is_winner(rewards: list[Any], index: int) -> bool:
    if not 0 <= index < len(rewards) or not isinstance(rewards[index], (int, float)):
        return False
    numeric = [float(value) for value in rewards if isinstance(value, (int, float))]
    return bool(numeric) and float(rewards[index]) == max(numeric)


def episode_date(path: Path) -> str:
    match = DATE_RE.search(path.as_posix())
    return match.group(0) if match else "unknown"


def split_for_date(date: str) -> str:
    if date in TRAIN_DATES:
        return "train"
    if date in VALIDATION_DATES:
        return "validation"
    if date in TEST_DATES:
        return "test"
    raise ValueError(f"date is outside the frozen split: {date}")


def _scan_array(path: Path, key: str) -> list[Any]:
    if ijson is not None:
        with path.open("rb") as handle:
            return _as_list(next(ijson.items(handle, key), []))
    payload = json.loads(path.read_text(encoding="utf-8"))
    value: Any = payload
    for part in key.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return _as_list(value)


def scan_episode_teams(path: Path) -> list[str]:
    return [str(value) for value in _scan_array(path, "info.TeamNames")]


def scan_episode_rewards(path: Path) -> list[Any]:
    return _scan_array(path, "rewards")


def _visual_trace(payload: dict[str, Any]) -> list[dict[str, Any]]:
    traces: list[list[Any]] = []
    singletons: list[Any] = []
    for step in _as_list(payload.get("steps")):
        for record in _as_list(step):
            raw = record.get("visualize") if isinstance(record, dict) else None
            if raw is None and isinstance(record, dict):
                raw = record.get("visual")
            if isinstance(raw, list) and raw:
                traces.append(raw)
            elif isinstance(raw, dict):
                singletons.append(raw)
    values = max(traces, key=len) if traces else singletons
    return [frame for frame in values if isinstance(frame, dict)]


def _episode_deck(frames: list[dict[str, Any]], player_index: int) -> list[int]:
    for frame in frames:
        action = frame.get("action")
        if (
            isinstance(action, list)
            and len(action) > player_index
            and isinstance(action[player_index], list)
            and len(action[player_index]) == 60
        ):
            return [int(card_id) for card_id in action[player_index]]
    raise ValueError("visual trace has no public 60-card deck frame")


def _episode_files(roots: Iterable[Path]) -> Iterator[Path]:
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"daily Episode root does not exist: {resolved}")
        for path in resolved.rglob("*.json"):
            if episode_date(path) not in ALL_DATES or path in seen:
                continue
            seen.add(path)
            yield path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decision_fingerprint(episode_id: int, frame_index: int, action: list[int]) -> str:
    raw = json.dumps(
        [episode_id, frame_index, action],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _history_event(
    observation: dict[str, Any],
    action: list[int],
    effect_step: int,
) -> dict[str, int]:
    select = observation.get("select") or {}
    options = select.get("option") or []
    option = options[action[0]] if action and action[0] < len(options) else {}
    if not isinstance(option, dict):
        option = {}
    return {
        "type": _as_int(option.get("type"), 0),
        "cardId": _as_int(option.get("cardId"), 0),
        "attackId": _as_int(option.get("attackId"), 0),
        "targetCount": len(action),
        "selectionType": _as_int(select.get("type"), -1),
        "selectionContext": _as_int(select.get("context"), -1),
        "effectStep": effect_step,
    }


def _universal_record(
    *,
    observation: dict[str, Any],
    action: list[int],
    deck: list[int],
    history: list[dict[str, int]],
    effect_steps: dict[int, int],
    episode_id: int,
    frame_index: int,
    player_index: int,
    split: str,
    date: str,
    metadata: dict[int, list[float]],
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    select = observation.get("select") or {}
    options = select.get("option") or []
    # The current repository contract has 64 candidate slots.  Keeping a row
    # with additional legal choices would give it a different action space
    # from the reference model, so the common corpus excludes it for both.
    if len(options) > 64:
        return None, _history_event(observation, action, 0)
    effect = select.get("effect") or {}
    effect_serial = _as_int(effect.get("serial"), -1)
    effect_step = effect_steps.get(effect_serial, 0)
    model_observation = dict(observation)
    model_observation["rl_history"] = list(history)
    model_observation["rl_deck"] = list(deck)
    model_observation["rl_expert_id"] = 1
    model_observation["rl_effect_step"] = effect_step
    model_observation["rl_card_metadata"] = metadata
    encoded = encode_observation(
        model_observation,
        feature_config_for_schema(UNIVERSAL_FEATURE_SCHEMA),
    )
    mask = encoded.get("action_mask") or []
    if any(target >= len(mask) or not mask[target] for target in action):
        return None, _history_event(observation, action, effect_step)
    event = _history_event(observation, action, effect_step)
    return (
        {
            "dataset_version": "ptcg_kaggle_bc_universal",
            "action_schema_version": "ptcg_action_set_v1",
            "feature_schema_version": UNIVERSAL_FEATURE_SCHEMA,
            "source_selection_schema": SCHEMA_VERSION,
            "expert_team_name": TEACHER_TEAM,
            "submission_id": 0,
            "episode_id": episode_id,
            "episode_step": frame_index,
            "decision_id": f"{episode_id}:{frame_index}",
            "decision_fingerprint": _decision_fingerprint(episode_id, frame_index, action),
            "date": date,
            "player_index": player_index,
            "selection_type": _as_int(select.get("type"), -1),
            "selection_context": _as_int(select.get("context"), -1),
            "selection_min_count": max(0, _as_int(select.get("minCount"), 0)),
            "selection_max_count": max(0, _as_int(select.get("maxCount"), len(action))),
            "targets": list(action),
            "target_count": len(action),
            "terminal_outcome": 1.0,
            "potential_shaping": {"total": 0.0},
            "split": split,
            "encoded": encoded,
        },
        event,
    )


def build_paired_dataset(
    roots: Iterable[Path],
    output: Path,
    *,
    teacher_team: str = TEACHER_TEAM,
) -> dict[str, Any]:
    """Create reference and universal views with identical accepted decisions."""
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to reuse dataset directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    reference_root = output / "reference_id_only"
    universal_root = output / "universal"
    reference_root.mkdir()
    universal_root.mkdir()
    reference_paths = {
        split: reference_root / f"{split}.jsonl.gz"
        for split in ("train", "validation", "test")
    }
    reference_handles = {
        split: gzip.open(path, "wt", encoding="utf-8")
        for split, path in reference_paths.items()
    }
    universal_path = universal_root / "dataset.jsonl"
    reference_codec = IDOnlyCodec(IDOnlyConfig())
    card_metadata = load_card_metadata()
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    episodes: list[dict[str, Any]] = []
    decision_ids_by_split: dict[str, list[str]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    teacher = normalize_team(teacher_team)
    try:
        with universal_path.open("w", encoding="utf-8") as universal_handle:
            for file_index, path in enumerate(_episode_files(roots), 1):
                counts["episodes_seen"] += 1
                try:
                    teams = scan_episode_teams(path)
                except Exception as exc:  # noqa: BLE001 - audit exact source failures
                    skipped[f"team_scan_{type(exc).__name__}"] += 1
                    continue
                matching = [
                    index for index, name in enumerate(teams) if normalize_team(name) == teacher
                ]
                if len(matching) != 1:
                    if matching:
                        skipped["non_unique_teacher"] += 1
                    continue
                player_index = matching[0]
                counts["team_matches"] += 1
                try:
                    rewards = scan_episode_rewards(path)
                except Exception as exc:  # noqa: BLE001
                    skipped[f"reward_scan_{type(exc).__name__}"] += 1
                    continue
                if not is_winner(rewards, player_index):
                    skipped["teacher_not_winner"] += 1
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    frames = _visual_trace(payload)
                    deck = _episode_deck(frames, player_index)
                except Exception as exc:  # noqa: BLE001
                    skipped[f"replay_{type(exc).__name__}"] += 1
                    continue
                episode_id = _as_int((payload.get("info") or {}).get("EpisodeId"), -1)
                if episode_id < 0:
                    episode_id = _as_int(path.stem, -1)
                if episode_id < 0:
                    skipped["missing_episode_id"] += 1
                    continue
                date = episode_date(path)
                split = split_for_date(date)
                accepted = 0
                history: list[dict[str, int]] = []
                effect_steps: dict[int, int] = {}
                for frame_index, frame in enumerate(frames):
                    observation = frame.get("obs")
                    action = frame.get("selected")
                    if not isinstance(observation, dict):
                        continue
                    current = observation.get("current") or {}
                    if _as_int(current.get("yourIndex"), -1) != player_index:
                        continue
                    select = observation.get("select") or {}
                    if not valid_reference_action(action, select):
                        skipped["invalid_or_missing_action"] += 1
                        continue
                    assert isinstance(action, list)
                    reference = reference_codec.encode(observation, action)
                    if reference is None:
                        skipped["reference_codec_limit"] += 1
                        continue
                    universal, history_event = _universal_record(
                        observation=observation,
                        action=action,
                        deck=deck,
                        history=history,
                        effect_steps=effect_steps,
                        episode_id=episode_id,
                        frame_index=frame_index,
                        player_index=player_index,
                        split=split,
                        date=date,
                        metadata=card_metadata,
                    )
                    effect = select.get("effect") or {}
                    serial = _as_int(effect.get("serial"), -1)
                    history.append(history_event)
                    del history[:-32]
                    if serial >= 0:
                        effect_steps[serial] = effect_steps.get(serial, 0) + 1
                    if universal is None:
                        skipped["universal_codec_limit"] += 1
                        continue
                    decision_id = str(universal["decision_id"])
                    reference.update(
                        {
                            "source_selection_schema": SCHEMA_VERSION,
                            "expert_team_name": teacher_team,
                            "episode_id": episode_id,
                            "episode_step": frame_index,
                            "decision_id": decision_id,
                            "decision_fingerprint": universal["decision_fingerprint"],
                            "date": date,
                            "split": split,
                        }
                    )
                    reference_handles[split].write(
                        json.dumps(reference, separators=(",", ":")) + "\n"
                    )
                    universal_handle.write(
                        json.dumps(universal, ensure_ascii=True, sort_keys=True) + "\n"
                    )
                    decision_ids_by_split[split].append(decision_id)
                    counts[f"decisions_{split}"] += 1
                    accepted += 1
                if accepted:
                    episodes.append(
                        {
                            "episode_id": episode_id,
                            "date": date,
                            "split": split,
                            "player_index": player_index,
                            "teams": teams,
                            "rewards": rewards,
                            "accepted_decisions": accepted,
                            "source_file": path.name,
                            "source_bytes": path.stat().st_size,
                        }
                    )
                    counts[f"episodes_{split}"] += 1
                if file_index == 1 or file_index % 250 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "extract_progress",
                                "episodes_seen": counts["episodes_seen"],
                                "team_matches": counts["team_matches"],
                                "decisions": sum(
                                    counts[f"decisions_{name}"]
                                    for name in ("train", "validation", "test")
                                ),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    finally:
        for handle in reference_handles.values():
            handle.close()

    missing_splits = [
        split for split, ids in decision_ids_by_split.items() if not ids
    ]
    if missing_splits:
        raise RuntimeError(f"paired dataset has empty splits: {missing_splits}")
    if len({item for values in decision_ids_by_split.values() for item in values}) != sum(
        len(values) for values in decision_ids_by_split.values()
    ):
        raise RuntimeError("decision IDs are not globally unique")

    card_metadata_path = universal_path.with_suffix(".jsonl.card_metadata.json")
    card_metadata_path.write_text(
        json.dumps(serialize_card_metadata(card_metadata), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paired_hashes = {
        split: hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
        for split, ids in decision_ids_by_split.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "teacher_team": teacher_team,
        "source_contract": "official daily Episodes; exact team; winner-only",
        "dates": {
            "train": list(TRAIN_DATES),
            "validation": list(VALIDATION_DATES),
            "test": list(TEST_DATES),
        },
        "split_policy": "calendar holdout; no Episode crosses splits",
        "common_decision_policy": (
            "emit only when both reference ID-only and universal codecs represent "
            "the same visual decision frame"
        ),
        "counts": dict(sorted(counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "decision_id_sha256_by_split": paired_hashes,
        "episodes": sorted(episodes, key=lambda row: (row["date"], row["episode_id"])),
        "outputs": {
            "reference": {
                split: {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for split, path in reference_paths.items()
            },
            "universal": {
                "path": str(universal_path),
                "sha256": _sha256(universal_path),
                "bytes": universal_path.stat().st_size,
                "card_metadata": str(card_metadata_path),
                "card_metadata_sha256": _sha256(card_metadata_path),
            },
        },
    }
    manifest_path = output / "paired_dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-team", default=TEACHER_TEAM)
    args = parser.parse_args()
    result = build_paired_dataset(
        args.input_root,
        args.output,
        teacher_team=args.teacher_team,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
