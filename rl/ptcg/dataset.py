from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from rl.core.storage import DEFAULT_MIN_FREE_GIB, DEFAULT_STORAGE_PATH, assert_storage_safe

from .features import FEATURE_SCHEMA_VERSION, PTCGFeatureConfig, encode_observation
from .rewards import potential_shaping


DATASET_VERSION = "ptcg_bc_v1"
SUPPORTED_FEATURE_SCHEMA_VERSIONS = frozenset(
    {"ptcg_features_v1", "ptcg_features_v2", FEATURE_SCHEMA_VERSION}
)
MAIN_SELECT_TYPE = 0
MAIN_SELECT_CONTEXT = 0


def _read_trace(path: str | Path) -> dict[str, Any]:
    trace_path = Path(path)
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("trace"), list):
        raise ValueError(f"trace is not a local battle trace: {trace_path}")
    return payload


def iter_behavior_cloning_records(
    path: str | Path,
    *,
    teacher_player_index: int | None = 0,
    feature_config: PTCGFeatureConfig = PTCGFeatureConfig(),
    include_effect_selections: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield masked-candidate BC records from a local battle trace.

    The local runner stores the observation before each ``battle_select``
    call and the teacher's returned list of option indices. The first phase
    deliberately keeps only the main-action contract (``type=0, context=0``)
    and single-index decisions. Effect selections can be enabled later, but
    multi-select effects still require a separate sequential target contract.
    """
    if teacher_player_index not in (None, 0, 1):
        raise ValueError("teacher_player_index must be 0 or 1")
    payload = _read_trace(path)
    source_path = str(Path(path).resolve())
    result = payload.get("result") or {}
    game_id = str(result.get("game_id") or result.get("gameId") or Path(path).stem)
    winner = result.get("winner")
    terminal_outcome = 1.0 if winner == 0 else -1.0 if winner == 1 else 0.0
    actual_teacher_index = teacher_player_index
    if actual_teacher_index is None:
        actual_teacher_index = int(result.get("candidate_physical_index", 0))
    if actual_teacher_index not in (0, 1):
        raise ValueError(f"invalid candidate physical index in {source_path}")

    trace_entries = payload["trace"]
    history: list[dict[str, int]] = []
    for entry_index, entry in enumerate(trace_entries):
        if not isinstance(entry, dict):
            continue
        observation = entry.get("observation")
        if not isinstance(observation, dict):
            continue
        current = observation.get("current") or {}
        select = observation.get("select") or {}
        if int(current.get("yourIndex", -1)) != actual_teacher_index:
            continue
        select_type_value = select.get("type", -1)
        context_value = select.get("context", -1)
        select_type = int(-1 if select_type_value is None else select_type_value)
        context = int(-1 if context_value is None else context_value)
        is_main = select_type == MAIN_SELECT_TYPE and context == MAIN_SELECT_CONTEXT
        if not is_main and not include_effect_selections:
            continue
        action = entry.get("action")
        if not isinstance(action, list) or len(action) != 1:
            # The candidate policy currently scores one option at a time.
            # Ignoring a multi-select record is safer than pretending its
            # first target was the complete teacher decision.
            continue
        try:
            target = int(action[0])
        except (TypeError, ValueError) as exc:
            location = f"{source_path}:{entry.get('step')}"
            raise ValueError(f"invalid teacher action at {location}") from exc
        options = select.get("option") or []
        if not isinstance(options, list) or not 0 <= target < len(options):
            raise ValueError(
                f"teacher action is outside legal options at {source_path}:{entry.get('step')}"
            )
        model_observation = dict(observation)
        model_observation["rl_history"] = list(history)
        encoded = encode_observation(model_observation, feature_config)
        if not encoded["action_mask"][target]:
            raise ValueError(
                f"teacher action is masked at {source_path}:{entry.get('step')}"
            )
        option = options[target]
        if is_main:
            history.append(
                {
                    "type": int(option.get("type", 0) or 0),
                    "cardId": int(option.get("cardId", 0) or 0),
                    "attackId": int(option.get("attackId", 0) or 0),
                }
            )
            del history[:-32]
        yield {
            "dataset_version": DATASET_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "source": source_path,
            "game_id": game_id,
            "step": int(entry.get("step", -1)),
            "player_index": actual_teacher_index,
            "selection_type": select_type,
            "selection_context": context,
            "target": target,
            "terminal_outcome": terminal_outcome,
            "potential_shaping": potential_shaping(
                observation,
                (
                    trace_entries[entry_index + 1].get("observation")
                    if entry_index + 1 < len(trace_entries)
                    and isinstance(trace_entries[entry_index + 1], dict)
                    else None
                ),
            ),
            "encoded": encoded,
        }


def write_behavior_cloning_dataset(
    traces: Iterable[str | Path],
    output: str | Path,
    *,
    teacher_player_index: int | None = 0,
    feature_config: PTCGFeatureConfig = PTCGFeatureConfig(),
    include_effect_selections: bool = False,
    storage_path: str | Path = DEFAULT_STORAGE_PATH,
    min_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> dict[str, int | str | float]:
    """Convert one or more local traces to a versioned JSONL dataset."""
    storage = assert_storage_safe(storage_path, min_free_gib)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    traces_read = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            traces_read += 1
            for record in iter_behavior_cloning_records(
                trace,
                teacher_player_index=teacher_player_index,
                feature_config=feature_config,
                include_effect_selections=include_effect_selections,
            ):
                handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
                records += 1
    return {
        "dataset_version": DATASET_VERSION,
        "traces": traces_read,
        "records": records,
        "output": str(output_path),
        "storage_path": storage.path,
        "storage_free_gib": round(storage.free_gib, 2),
    }


def load_behavior_cloning_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate the JSONL records used by the BC trainer."""
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid dataset JSON at line {line_number}") from exc
            if (
                not isinstance(record, dict)
                or record.get("dataset_version") != DATASET_VERSION
                or record.get("feature_schema_version") not in SUPPORTED_FEATURE_SCHEMA_VERSIONS
            ):
                raise ValueError(f"unsupported dataset record at line {line_number}")
            encoded = record.get("encoded")
            target = record.get("target")
            if not isinstance(encoded, dict) or not isinstance(target, int):
                raise ValueError(f"malformed dataset record at line {line_number}")
            if "terminal_outcome" not in record:
                record["terminal_outcome"] = 0.0
            if record["terminal_outcome"] not in (-1.0, 0.0, 1.0):
                raise ValueError(f"invalid terminal outcome at dataset line {line_number}")
            if "potential_shaping" not in record:
                record["potential_shaping"] = {"total": 0.0}
            if not isinstance(record["potential_shaping"], dict):
                raise ValueError(f"invalid potential shaping at dataset line {line_number}")
            mask = encoded.get("action_mask")
            if not isinstance(mask, list) or not 0 <= target < len(mask) or not mask[target]:
                raise ValueError(f"illegal target at dataset line {line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"dataset contains no records: {path}")
    return records
