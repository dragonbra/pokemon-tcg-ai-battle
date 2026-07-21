"""Relabel model-rollout observations with the deterministic rule teacher."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Iterator

from rl.core.storage import DEFAULT_MIN_FREE_GIB, DEFAULT_STORAGE_PATH, assert_storage_safe

from .dataset import DATASET_VERSION, MAIN_SELECT_CONTEXT, MAIN_SELECT_TYPE
from .features import feature_config_for_schema, encode_observation


def _load_teacher(root: Path) -> ModuleType:
    path = root.resolve() / "main.py"
    spec = importlib.util.spec_from_file_location("rl_dagger_teacher", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load teacher module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_trace(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("trace"), list):
        raise ValueError(f"not an evaluation trace: {path}")
    return payload


def _outcome(payload: dict[str, Any], player_index: int) -> float:
    winner = (payload.get("result") or {}).get("winner")
    return 1.0 if winner == player_index else -1.0 if winner in (0, 1) else 0.0


def iter_dagger_records(
    path: str | Path,
    teacher: ModuleType,
    *,
    teacher_player_index: int | None = None,
    feature_schema_version: str = "ptcg_features_v2",
) -> Iterator[dict[str, Any]]:
    """Yield expert labels for states visited by a model rollout.

    The teacher is called for every candidate-owned selection so its stateful
    turn memory remains coherent, but only single-option main actions become
    training records. The rollout's outcome is retained for audit and is not
    used by default as a BC weight.
    """
    config = feature_config_for_schema(feature_schema_version)
    trace_path = Path(path)
    payload = _read_trace(trace_path)
    result = payload.get("result") or {}
    game_id = str(result.get("game_id") or trace_path.stem)
    player_index = (
        teacher_player_index
        if teacher_player_index is not None
        else int(result.get("candidate_physical_index", 0))
    )
    if player_index not in (0, 1):
        raise ValueError(f"invalid teacher player index in {trace_path}")

    # The rule agent resets module-level turn state when it sees select=None.
    teacher.agent({"select": None})
    history: list[dict[str, int]] = []
    entries = payload["trace"]
    outcome = _outcome(payload, player_index)
    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        observation = entry.get("observation")
        if not isinstance(observation, dict):
            continue
        current = observation.get("current") or {}
        if int(current.get("yourIndex", -1)) != player_index:
            continue

        select = observation.get("select") or {}
        is_main = (
            int(select.get("type", -1)) == MAIN_SELECT_TYPE
            and int(select.get("context", -1)) == MAIN_SELECT_CONTEXT
        )
        # Call the expert on effects as well, so its stateful memory follows
        # the same sequence of observations as it would in a real battle.
        # A failed effect-only selection is not a usable main-action label;
        # keep the rest of the rollout instead of fabricating a target.
        try:
            expert_action = teacher.agent(observation)
        except Exception:
            if not is_main:
                continue
            raise
        if not is_main:
            continue
        if not isinstance(expert_action, list) or len(expert_action) != 1:
            continue
        target = int(expert_action[0])
        options = select.get("option") or []
        if not 0 <= target < len(options):
            raise ValueError(f"teacher returned an illegal option at {trace_path}:{entry.get('step')}")

        model_observation = dict(observation)
        model_observation["rl_history"] = list(history)
        encoded = encode_observation(model_observation, config)
        if not encoded["action_mask"][target]:
            raise ValueError(f"teacher target was masked at {trace_path}:{entry.get('step')}")
        option = options[target]
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
            "feature_schema_version": feature_schema_version,
            "source": str(trace_path.resolve()),
            "game_id": game_id,
            "step": int(entry.get("step", -1)),
            "player_index": player_index,
            "selection_type": int(select.get("type", -1)),
            "selection_context": int(select.get("context", -1)),
            "target": target,
            "terminal_outcome": outcome,
            "label_source": "rule_teacher_dagger",
            "rollout_action": entry.get("action"),
            "encoded": encoded,
        }


def write_dagger_dataset(
    traces: Iterable[str | Path],
    output: str | Path,
    *,
    teacher_root: str | Path,
    teacher_player_index: int | None = None,
    feature_schema_version: str = "ptcg_features_v2",
    storage_path: str | Path = DEFAULT_STORAGE_PATH,
    min_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> dict[str, int | str | float]:
    storage = assert_storage_safe(storage_path, min_free_gib)
    teacher = _load_teacher(Path(teacher_root))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traces_read = 0
    records = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            traces_read += 1
            for record in iter_dagger_records(
                trace,
                teacher,
                teacher_player_index=teacher_player_index,
                feature_schema_version=feature_schema_version,
            ):
                handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
                records += 1
    return {
        "dataset_version": DATASET_VERSION,
        "traces": traces_read,
        "records": records,
        "output": str(output_path),
        "feature_schema_version": feature_schema_version,
        "storage_path": storage.path,
        "storage_free_gib": round(storage.free_gib, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature-schema",
        default="ptcg_features_v2",
        choices=(
            "ptcg_features_v1",
            "ptcg_features_v2",
            "ptcg_features_v3",
            "ptcg_features_v4",
        ),
    )
    parser.add_argument("--teacher-player-index", type=int, choices=(0, 1))
    parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE_PATH)
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    args = parser.parse_args()
    result = write_dagger_dataset(
        args.traces,
        args.output,
        teacher_root=args.teacher,
        teacher_player_index=args.teacher_player_index,
        feature_schema_version=args.feature_schema,
        storage_path=args.storage_path,
        min_free_gib=args.min_free_gib,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
