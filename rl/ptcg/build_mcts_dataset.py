"""Build small policy targets from official counterfactual search expansions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import itertools
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence

from rl.core.storage import DEFAULT_MIN_FREE_GIB, DEFAULT_STORAGE_PATH, assert_storage_safe

from .dataset import DATASET_VERSION
from .features import encode_observation
from .inference import PTCGCandidatePolicy
from .mcts import PUCTSearch, Selection


def _load_deck(path: Path) -> list[int]:
    values = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(values) != 60:
        raise ValueError(f"deck must contain 60 card IDs: {path}")
    return values


def _load_trace(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("trace"), list):
        raise ValueError(f"not an evaluation trace: {path}")
    return payload


def _load_teacher(root: Path) -> ModuleType:
    path = root.resolve() / "main.py"
    spec = importlib.util.spec_from_file_location("rl_mcts_rollout_teacher", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load rollout teacher: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_cg(cg_root: Path):
    sys.path.insert(0, str(cg_root.resolve()))
    from cg.api import search_begin, search_end, search_step, to_observation_class

    return search_begin, search_end, search_step, to_observation_class


def _perspective_observation(observation: object, player_index: int) -> dict[str, Any]:
    result = asdict(observation)
    current = result.get("current") or {}
    if int(current.get("yourIndex", player_index)) == player_index:
        return result
    players = current.get("players") or []
    if len(players) == 2:
        current["players"] = [players[1], players[0]]
    current["yourIndex"] = player_index
    first_player = current.get("firstPlayer")
    if first_player in (0, 1):
        current["firstPlayer"] = 1 - int(first_player)
    return result


def _leaf_value(
    observation: object,
    player_index: int,
    policy: PTCGCandidatePolicy,
) -> float:
    state = getattr(observation, "current", None)
    if state is None:
        return 0.0
    result = getattr(state, "result", None)
    if isinstance(result, int) and result >= 0:
        if result == 2:
            return 0.0
        return 1.0 if result == player_index else -1.0
    observation_dict = _perspective_observation(observation, player_index)
    try:
        value, _probabilities = policy.score_candidates(observation_dict)
        return float(value)
    except Exception:
        return 0.0


def _current_player(observation: object) -> int:
    state = getattr(observation, "current", None)
    value = getattr(state, "yourIndex", -1) if state is not None else -1
    return int(value)


def _teacher_selection(
    observation: object,
    teacher: ModuleType | None,
) -> Selection | None:
    if teacher is None:
        return None
    payload = asdict(observation)
    select = payload.get("select") or {}
    options = select.get("option") or []
    minimum = int(select.get("minCount", 1) or 0)
    maximum = int(select.get("maxCount", 1) or 1)
    try:
        proposed = teacher.agent(payload)
    except Exception:
        return None
    if not isinstance(proposed, list):
        return None
    try:
        selection = tuple(int(index) for index in proposed)
    except (TypeError, ValueError):
        return None
    if not minimum <= len(selection) <= maximum:
        return None
    if len(set(selection)) != len(selection) or any(
        index < 0 or index >= len(options) for index in selection
    ):
        return None
    return selection


def _selection_candidates(
    observation: object,
    policy: PTCGCandidatePolicy,
    *,
    max_children: int,
    rollout_teacher: ModuleType | None,
) -> list[tuple[Selection, float]]:
    """Return bounded legal selections and model priors for one SearchState."""
    payload = asdict(observation)
    select = payload.get("select") or {}
    options = select.get("option") or []
    if not options:
        return []
    minimum = int(select.get("minCount", 1) or 0)
    maximum = int(select.get("maxCount", 1) or 1)
    minimum = max(0, minimum)
    maximum = min(maximum, len(options))
    if minimum > maximum:
        return []

    actor_index = _current_player(observation)
    policy_failed = False
    try:
        _value, probabilities = policy.score_candidates(
            _perspective_observation(observation, actor_index)
        )
    except Exception:
        policy_failed = True
        probabilities = [1.0 / len(options)] * len(options)

    if policy_failed:
        teacher_selection = _teacher_selection(observation, rollout_teacher)
        if teacher_selection is not None:
            return [(teacher_selection, 1.0)]

    def prior(selection: Selection) -> float:
        if not selection:
            return 1.0
        return math.prod(
            max(1e-8, float(probabilities[index]))
            if index < len(probabilities)
            else 1.0
            for index in selection
        )

    if minimum == maximum == 1:
        return [((index,), prior((index,))) for index in range(len(options))]

    counts = sum(math.comb(len(options), count) for count in range(minimum, maximum + 1))
    candidate_indices = list(range(len(options)))
    if counts > max_children:
        candidate_indices = sorted(
            candidate_indices,
            key=lambda index: float(probabilities[index]) if index < len(probabilities) else 0.0,
            reverse=True,
        )[: max(max_children, maximum)]

    selections: list[Selection] = []
    for count in range(minimum, maximum + 1):
        for combination in itertools.combinations(candidate_indices, count):
            selections.append(tuple(combination))
            if len(selections) >= max_children:
                break
        if len(selections) >= max_children:
            break
    if not selections:
        return []
    return [(selection, prior(selection)) for selection in selections]


def _start_search(
    observation: dict[str, Any],
    *,
    player_index: int,
    deck: list[int],
    rng: random.Random,
    search_begin: Any,
    to_observation_class: Any,
) -> Any:
    current = observation.get("current") or {}
    players = current.get("players") or []
    if len(players) != 2:
        raise ValueError("search observation must contain two players")
    own = players[player_index]
    opponent = players[1 - player_index]
    active = opponent.get("active") or []
    deck_count = int(own.get("deckCount", 0) or 0)
    prize_count = len(own.get("prize") or [])
    return search_begin(
        to_observation_class(observation),
        your_deck=rng.sample(deck, min(deck_count, len(deck))),
        your_prize=rng.sample(deck, min(prize_count, len(deck))),
        opponent_deck=[1072] * int(opponent.get("deckCount", 0) or 0),
        opponent_prize=[1] * len(opponent.get("prize") or []),
        opponent_hand=[1] * int(opponent.get("handCount", 0) or 0),
        opponent_active=[1072] if active and active[0] is None else [],
    )


def _run_search(
    observation: dict[str, Any],
    *,
    player_index: int,
    deck: list[int],
    policy: PTCGCandidatePolicy,
    search_begin: Any,
    search_end: Any,
    search_step: Any,
    to_observation_class: Any,
    rng: random.Random,
    simulations: int,
    cpuct: float,
    rollout_teacher: ModuleType | None,
) -> tuple[list[float | None], list[int], list[float], list[float | None]]:
    root_state = _start_search(
        observation,
        player_index=player_index,
        deck=deck,
        rng=rng,
        search_begin=search_begin,
        to_observation_class=to_observation_class,
    )
    try:
        search = PUCTSearch(
            root_player=player_index,
            simulations=simulations,
            cpuct=cpuct,
            step=lambda state, selection: search_step(state.searchId, selection),
            observation=lambda state: state.observation,
            player_index=_current_player,
            evaluate=lambda state_observation: _leaf_value(
                state_observation, player_index, policy
            ),
            expand=lambda state_observation: _selection_candidates(
                state_observation,
                policy,
                max_children=64,
                rollout_teacher=rollout_teacher,
            ),
            rng=rng,
        )
        root = search.run(root_state)
        values_by_index: list[float | None] = [None] * len(observation["select"]["option"])
        visits_by_index = [0] * len(values_by_index)
        policy_by_index = [0.0] * len(values_by_index)
        root_values = search.root_values(root)
        root_policy = search.root_policy(root)
        root_values_by_index: list[float | None] = [None] * len(values_by_index)
        for child, value, visits, target_probability in zip(
            root.children,
            root_values,
            [child.node.visits if child.node is not None else 0 for child in root.children],
            root_policy,
        ):
            if len(child.selection) != 1:
                continue
            index = child.selection[0]
            if 0 <= index < len(values_by_index):
                values_by_index[index] = float(value)
                root_values_by_index[index] = float(value)
                visits_by_index[index] = int(visits)
                policy_by_index[index] = float(target_probability)
        return values_by_index, visits_by_index, policy_by_index, root_values_by_index
    finally:
        search_end()


def _aggregate_search_results(
    results: Sequence[
        tuple[list[float | None], list[int], list[float], list[float | None]]
    ],
    option_count: int,
) -> tuple[list[float | None], list[int], list[float], list[float | None]]:
    """Aggregate root targets from several hidden-card determinizations."""
    if not results:
        raise ValueError("cannot aggregate an empty MCTS result set")
    value_sums = [0.0] * option_count
    value_counts = [0] * option_count
    visit_counts = [0] * option_count
    fallback_policy = [0.0] * option_count
    for values, visits, policy, _root_values in results:
        if not (len(values) == len(visits) == len(policy) == option_count):
            raise ValueError("MCTS result width does not match the root option count")
        for index in range(option_count):
            if values[index] is not None:
                value_sums[index] += float(values[index])
                value_counts[index] += 1
            visit_counts[index] += int(visits[index])
            fallback_policy[index] += float(policy[index])
    values = [
        value_sums[index] / value_counts[index] if value_counts[index] else None
        for index in range(option_count)
    ]
    total_visits = sum(visit_counts)
    if total_visits:
        policy = [count / total_visits for count in visit_counts]
    else:
        total_fallback = sum(fallback_policy)
        policy = (
            [value / total_fallback for value in fallback_policy]
            if total_fallback
            else [1.0 / option_count] * option_count
        )
    return values, visit_counts, policy, values.copy()


def _blend_teacher_policy(
    policy: list[float],
    action: object,
    weight: float,
) -> list[float]:
    """Optionally anchor a search target to a known teacher action."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("teacher policy weight must be in [0, 1]")
    if weight == 0.0 or not isinstance(action, list) or len(action) != 1:
        return policy
    try:
        target = int(action[0])
    except (TypeError, ValueError):
        return policy
    if not 0 <= target < len(policy):
        return policy
    return [
        (1.0 - weight) * value + (weight if index == target else 0.0)
        for index, value in enumerate(policy)
    ]


def build_records(
    traces: Iterable[Path],
    *,
    checkpoint: Path,
    deck: list[int],
    cg_root: Path,
    max_records: int,
    seed: int,
    rollout_steps: int = 0,
    rollout_teacher: ModuleType | None = None,
    simulations: int = 32,
    cpuct: float = 1.25,
    determinizations: int = 1,
    teacher_policy_weight: float = 0.0,
) -> list[dict[str, Any]]:
    if rollout_steps < 0:
        raise ValueError("rollout_steps cannot be negative")
    if simulations < 1:
        raise ValueError("simulations must be positive")
    if determinizations < 1:
        raise ValueError("determinizations must be positive")
    if not 0.0 <= teacher_policy_weight <= 1.0:
        raise ValueError("teacher_policy_weight must be in [0, 1]")
    policy = PTCGCandidatePolicy.from_checkpoint(str(checkpoint), map_location="cpu")
    schema_by_width = {24: "ptcg_features_v1", 32: "ptcg_features_v2", 36: "ptcg_features_v3"}
    feature_schema_version = schema_by_width[policy.feature_config.state_numeric_dim]
    search_begin, search_end, search_step, to_observation_class = _load_cg(cg_root)
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for trace_path in traces:
        payload = _load_trace(trace_path)
        result = payload.get("result") or {}
        player_index = int(result.get("candidate_physical_index", 0))
        winner = result.get("winner")
        outcome = 1.0 if winner == player_index else -1.0 if winner in (0, 1) else 0.0
        for entry in payload["trace"]:
            if len(records) >= max_records:
                return records
            if not isinstance(entry, dict):
                continue
            observation = entry.get("observation")
            if not isinstance(observation, dict):
                continue
            current = observation.get("current") or {}
            select = observation.get("select") or {}
            if int(current.get("yourIndex", -1)) != player_index:
                continue
            if int(select.get("type", -1)) != 0 or int(select.get("context", -1)) != 0:
                continue
            if int(select.get("minCount", 0) or 0) != 1 or int(select.get("maxCount", 0) or 0) != 1:
                continue
            if not select.get("option"):
                continue
            encoded = encode_observation(observation, policy.feature_config)
            search_results = []
            for _ in range(determinizations):
                try:
                    search_results.append(
                        _run_search(
                            observation,
                            player_index=player_index,
                            deck=deck,
                            policy=policy,
                            search_begin=search_begin,
                            search_end=search_end,
                            search_step=search_step,
                            to_observation_class=to_observation_class,
                            rng=rng,
                            simulations=simulations,
                            cpuct=cpuct,
                            rollout_teacher=rollout_teacher,
                        )
                    )
                except Exception:
                    continue
            if not search_results:
                continue
            values, visit_counts, mcts_policy, root_values = _aggregate_search_results(
                search_results, len(select.get("option") or [])
            )
            mcts_policy = _blend_teacher_policy(
                mcts_policy,
                entry.get("action"),
                teacher_policy_weight,
            )
            valid = [index for index, value in enumerate(values) if value is not None]
            if not valid:
                continue
            target = max(
                valid,
                key=lambda index: (float(mcts_policy[index]), -index),
            )
            if not encoded["action_mask"][target]:
                continue
            records.append(
                {
                    "dataset_version": DATASET_VERSION,
                    "feature_schema_version": feature_schema_version,
                    "source": str(trace_path.resolve()),
                    "game_id": str(result.get("game_id") or trace_path.stem),
                    "step": int(entry.get("step", -1)),
                    "player_index": player_index,
                    "selection_type": 0,
                    "selection_context": 0,
                    "target": target,
                    "terminal_outcome": outcome,
                    "mcts_action_values": values,
                    "mcts_visit_counts": visit_counts,
                    "mcts_policy": mcts_policy,
                    "mcts_root_values": root_values,
                    "mcts_simulations": simulations,
                    "mcts_cpuct": cpuct,
                    "mcts_determinizations": len(search_results),
                    "mcts_teacher_policy_weight": teacher_policy_weight,
                    "teacher_or_rollout_action": entry.get("action"),
                    "encoded": encoded,
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--cg-root", type=Path, required=True)
    parser.add_argument("--rollout-teacher", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=128)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--cpuct", type=float, default=1.25)
    parser.add_argument("--determinizations", type=int, default=1)
    parser.add_argument("--teacher-policy-weight", type=float, default=0.0)
    # Kept for command-line compatibility with the pre-PUCT collector. The
    # search budget is now controlled by --simulations.
    parser.add_argument("--rollout-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--storage-path", type=Path, default=DEFAULT_STORAGE_PATH)
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    args = parser.parse_args()
    if args.max_records < 1 or args.rollout_steps < 0:
        raise ValueError("max-records must be positive and rollout-steps cannot be negative")
    if (
        args.simulations < 1
        or args.cpuct <= 0
        or args.determinizations < 1
        or not 0.0 <= args.teacher_policy_weight <= 1.0
    ):
        raise ValueError(
            "simulations and determinizations must be positive, cpuct must be positive, "
            "and teacher-policy-weight must be in [0, 1]"
        )
    storage = assert_storage_safe(args.storage_path, args.min_free_gib)
    rollout_teacher = _load_teacher(args.rollout_teacher) if args.rollout_teacher else None
    records = build_records(
        args.traces,
        checkpoint=args.checkpoint,
        deck=_load_deck(args.deck),
        cg_root=args.cg_root,
        max_records=args.max_records,
        seed=args.seed,
        rollout_steps=args.rollout_steps,
        rollout_teacher=rollout_teacher,
        simulations=args.simulations,
        cpuct=args.cpuct,
        determinizations=args.determinizations,
        teacher_policy_weight=args.teacher_policy_weight,
    )
    if not records:
        raise ValueError("MCTS target collection produced no records")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "records": len(records),
                "output": str(args.output),
                "storage_path": storage.path,
                "storage_free_gib": round(storage.free_gib, 2),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
