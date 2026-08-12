"""Evaluate a foundation PPO checkpoint with the official CPU engine.

The evaluation contract mirrors the corrected CUDA fixed-matchup evaluator:
player0 is the Lucario candidate, player1 is the initial frozen foundation,
and every opponent receives the same fixed seed schedule for the initial and
candidate runs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "tools"))
sys.path.insert(0, str(WORKSPACE_ROOT / "tools"))

from ptcg_cuda_engine.legacy_codecs import (  # noqa: E402
    policy_codec_v1_to_idonly_codec_v1,
)
from ptcg_cuda_engine.policy_adapters import (  # noqa: E402
    FoundationR15DeviceAdapter,
    StaticBatchFieldsDeviceAdapter,
    foundation_r15_static_fields,
)
from run_official_seeded_multi_policy_loop import (  # noqa: E402
    load_foundation_0020_checkpoint,
    load_pool_rows,
    read_deck,
)
from seeded_cpp_shim import SeededCppLib  # noqa: E402


ONGOING = 0
PLAYER0_WIN = 1
PLAYER1_WIN = 2
DRAW = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--learner-deck",
        type=Path,
        help="Optional 60-card deck.csv for player0; defaults to pool row 0.",
    )
    parser.add_argument(
        "--opponent",
        action="append",
        default=[],
        help="Evaluate only this exact pool opponent name; may be repeated.",
    )
    parser.add_argument(
        "--candidate-only",
        action="store_true",
        help="Skip the initial-foundation result and evaluate only the candidate.",
    )
    parser.add_argument(
        "--diagnostic-option",
        action="append",
        default=[],
        metavar="TYPE:RESOLVED_ID",
        help=(
            "Count player0 decisions where this native option is available or "
            "selected; may be repeated."
        ),
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=WORKSPACE_ROOT / ".tmp" / "cuda_zero_shot_six_decks" / "pool.json",
    )
    parser.add_argument(
        "--lib",
        type=Path,
        default=WORKSPACE_ROOT / "tmp" / "seeded_cpp_shim_policy_codec_v1" / "libcg_seeded.so",
    )
    parser.add_argument("--games-per-opponent", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--seed-start", type=int, default=2026080201)
    parser.add_argument("--engine-threads", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=64)
    parser.add_argument("--max-select", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_counts(
    wins: int, losses: int, draws: int, timeouts: int
) -> dict[str, float | int]:
    games = wins + losses + draws
    win_rate = wins / games if games else 0.0
    effective = (wins + 0.5 * draws) / games if games else 0.0
    standard_error = (effective * (1.0 - effective) / games) ** 0.5 if games else 0.0
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "timeouts": timeouts,
        "win_rate": win_rate,
        "effective_win_rate": effective,
        "effective_win_rate_ci95_low": max(0.0, effective - 1.96 * standard_error),
        "effective_win_rate_ci95_high": min(1.0, effective + 1.96 * standard_error),
    }


def resolve_pool_path(deck_directory: Path, manifest_path: str) -> Path:
    path = Path(manifest_path)
    if path.is_absolute():
        return path.resolve()
    workspace_relative = (WORKSPACE_ROOT / path).resolve()
    if workspace_relative.is_file():
        return workspace_relative
    return (deck_directory / path).resolve()


def _subset(batch: Mapping[str, Any], indices: Any) -> dict[str, Any]:
    return {
        name: value.index_select(0, indices)
        for name, value in batch.items()
    }


def _actions_for_route(
    batch: Mapping[str, Any],
    route_indices: Any,
    adapter: Any,
) -> dict[int, list[int]]:
    if route_indices.numel() == 0:
        return {}
    sub_batch = _subset(batch, route_indices)
    actions, lengths = adapter.act_device(sub_batch)
    output: dict[int, list[int]] = {}
    for local_index, global_index in enumerate(route_indices.tolist()):
        length = int(lengths[local_index].item())
        row = actions[local_index, :length].tolist()
        output[int(global_index)] = [int(value) for value in row if int(value) >= 0]
    return output


def _play_batch(
    *,
    shim: SeededCppLib,
    learner_model: Any,
    opponent_model: Any,
    learner_deck: list[int],
    opponent_deck: list[int],
    opponent_id: int,
    game_start: int,
    games: int,
    seed_start: int,
    max_steps: int,
    max_select: int,
    diagnostic_options: tuple[tuple[int, int], ...],
) -> tuple[dict[str, int], dict[str, Any]]:
    import torch

    seeds = [seed_start + opponent_id * 100_003 + game_start + offset for offset in range(games)]
    pairs = [(learner_deck, opponent_deck) for _ in seeds]
    battles = shim.start_battles(pairs, seeds)
    steps = [0] * len(battles)
    results = [ONGOING] * len(battles)
    timeouts = [False] * len(battles)
    learner_adapter = StaticBatchFieldsDeviceAdapter(
        FoundationR15DeviceAdapter(learner_model, max_select=max_select),
        foundation_r15_static_fields(learner_deck, torch.device("cpu")),
    )
    opponent_adapter = StaticBatchFieldsDeviceAdapter(
        FoundationR15DeviceAdapter(opponent_model, max_select=max_select),
        foundation_r15_static_fields(opponent_deck, torch.device("cpu")),
    )
    decisions = 0
    diagnostic_counts = {
        f"{option_type}:{resolved_id}": {
            "available_decisions": 0,
            "selected_decisions": 0,
            "games_available": set(),
            "games_selected": set(),
        }
        for option_type, resolved_id in diagnostic_options
    }
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for _tick in range(max_steps):
                metas = shim.select_meta_many(battles)
                for index, meta in enumerate(metas):
                    result = int(meta.game_result)
                    if result in (PLAYER0_WIN, PLAYER1_WIN, DRAW):
                        results[index] = result
                active_indices = [
                    index
                    for index, result in enumerate(results)
                    if result == ONGOING and steps[index] < max_steps
                ]
                if not active_indices:
                    break
                active_battles = [battles[index] for index in active_indices]
                codec = shim.policy_codec_v1_many(active_battles).to_torch()
                for name in ("entity_mask", "option_mask"):
                    codec[name] = codec[name].bool()
                actors = (codec["global_cat"][:, 3].long() - 1).clamp(0, 1)
                idonly = policy_codec_v1_to_idonly_codec_v1(
                    codec,
                    max_card_id=2048,
                    max_action_steps=max_select,
                    target_entity_capacity=128,
                    target_option_capacity=80,
                )
                route_actions: dict[int, list[int]] = {}
                route_actions.update(
                    _actions_for_route(
                        idonly,
                        actors.eq(0).nonzero(as_tuple=False).flatten(),
                        learner_adapter,
                    )
                )
                route_actions.update(
                    _actions_for_route(
                        idonly,
                        actors.eq(1).nonzero(as_tuple=False).flatten(),
                        opponent_adapter,
                    )
                )
                actions = [route_actions[index] for index in range(len(active_battles))]
                if diagnostic_options:
                    option_rows = shim.option_records_many(active_battles)
                    for local_index, game_index in enumerate(active_indices):
                        if int(actors[local_index].item()) != 0:
                            continue
                        selected = set(actions[local_index])
                        records = option_rows[local_index]
                        for option_type, resolved_id in diagnostic_options:
                            key = f"{option_type}:{resolved_id}"
                            matching = {
                                int(record[0])
                                for record in records
                                if int(record[1]) == option_type
                                and int(record[7]) == resolved_id
                            }
                            if not matching:
                                continue
                            row = diagnostic_counts[key]
                            row["available_decisions"] += 1
                            row["games_available"].add(game_index)
                            if selected & matching:
                                row["selected_decisions"] += 1
                                row["games_selected"].add(game_index)
                errors = shim.select_many(active_battles, actions)
                if any(int(error) != 0 for error in errors):
                    raise RuntimeError(
                        f"CPU engine select failed: {errors[:16]}"
                    )
                for index in active_indices:
                    steps[index] += 1
                decisions += len(active_battles)
            final_metas = shim.select_meta_many(battles)
            for index, meta in enumerate(final_metas):
                result = int(meta.game_result)
                if result in (PLAYER0_WIN, PLAYER1_WIN, DRAW):
                    results[index] = result
            for index, result in enumerate(results):
                if result == ONGOING:
                    timeouts[index] = True
                    results[index] = DRAW
    finally:
        shim.finish_many(battles)

    counts = {
        "wins": sum(result == PLAYER0_WIN for result in results),
        "losses": sum(result == PLAYER1_WIN for result in results),
        "draws": sum(result == DRAW for result in results),
        "timeouts": sum(timeouts),
    }
    diagnostics = {
        key: {
            "available_decisions": int(row["available_decisions"]),
            "selected_decisions": int(row["selected_decisions"]),
            "games_available": len(row["games_available"]),
            "games_selected": len(row["games_selected"]),
        }
        for key, row in diagnostic_counts.items()
    }
    return counts, {
        "games": len(battles),
        "decisions": decisions,
        "max_steps": max(steps) if steps else 0,
        "wall_seconds": time.perf_counter() - started,
        "diagnostic_options": diagnostics,
    }


def load_candidate_model(
    checkpoint: Path,
    foundation_checkpoint: Path,
    ontology: Path,
    initial_model: Any,
) -> tuple[Any, int]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise KeyError(f"{checkpoint} is missing model_state_dict")
    candidate = copy.deepcopy(initial_model)
    policy_state = {
        name.removeprefix("policy."): value.detach().cpu()
        for name, value in model_state.items()
        if name.startswith("policy.")
    }
    candidate.load_state_dict(policy_state, strict=True)
    candidate.eval().requires_grad_(False)
    return candidate, int(payload.get("iteration", -1))


def evaluate_label(
    *,
    label: str,
    learner_model: Any,
    opponent_model: Any,
    iteration: int,
    shim: SeededCppLib,
    learner_deck: list[int],
    opponent_decks: list[list[int]],
    opponent_names: list[str],
    opponent_ids: list[int],
    games_per_opponent: int,
    batch_size: int,
    seed_start: int,
    max_steps: int,
    max_select: int,
    diagnostic_options: tuple[tuple[int, int], ...],
) -> dict[str, Any]:
    totals = {"wins": 0, "losses": 0, "draws": 0, "timeouts": 0}
    per_opponent: dict[str, Any] = {}
    diagnostic_totals = {
        f"{option_type}:{resolved_id}": {
            "available_decisions": 0,
            "selected_decisions": 0,
            "games_available": 0,
            "games_selected": 0,
        }
        for option_type, resolved_id in diagnostic_options
    }
    started = time.perf_counter()
    for opponent_id, opponent_name, opponent_deck in zip(
        opponent_ids, opponent_names, opponent_decks, strict=True
    ):
        counts = {"wins": 0, "losses": 0, "draws": 0, "timeouts": 0}
        decisions = 0
        opponent_started = time.perf_counter()
        for game_start in range(0, games_per_opponent, batch_size):
            games = min(batch_size, games_per_opponent - game_start)
            batch_counts, batch_stats = _play_batch(
                shim=shim,
                learner_model=learner_model,
                opponent_model=opponent_model,
                learner_deck=learner_deck,
                opponent_deck=opponent_deck,
                opponent_id=opponent_id,
                game_start=game_start,
                games=games,
                seed_start=seed_start,
                max_steps=max_steps,
                max_select=max_select,
                diagnostic_options=diagnostic_options,
            )
            for key in counts:
                counts[key] += batch_counts[key]
                totals[key] += batch_counts[key]
            decisions += int(batch_stats["decisions"])
            for key, row in batch_stats["diagnostic_options"].items():
                for metric, value in row.items():
                    diagnostic_totals[key][metric] += int(value)
            print(
                "CPU_EVAL_BATCH "
                + json.dumps(
                    {
                        "label": label,
                        "opponent": opponent_name,
                        "games_done": min(game_start + games, games_per_opponent),
                        "games_target": games_per_opponent,
                        **{
                            key: value
                            for key, value in batch_stats.items()
                            if key != "diagnostic_options"
                        },
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        per_opponent[opponent_name] = {
            **summarize_counts(**counts),
            "decisions": decisions,
            "wall_seconds": time.perf_counter() - opponent_started,
        }
    return {
        "label": label,
        "iteration": iteration,
        "overall": summarize_counts(**totals),
        "per_opponent": per_opponent,
        "decisions": sum(int(row["decisions"]) for row in per_opponent.values()),
        "diagnostic_options": diagnostic_totals,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> int:
    args = parse_args()
    if min(args.games_per_opponent, args.batch_size, args.max_steps, args.max_select) <= 0:
        raise ValueError("games, batch, max-steps, and max-select must be positive")
    import torch

    diagnostic_options: list[tuple[int, int]] = []
    for value in args.diagnostic_option:
        try:
            option_type_text, resolved_id_text = value.split(":", 1)
            option_type = int(option_type_text)
            resolved_id = int(resolved_id_text)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid --diagnostic-option {value!r}; expected TYPE:RESOLVED_ID"
            ) from error
        if option_type < 0 or resolved_id < 0:
            raise ValueError("diagnostic option values must be nonnegative")
        pair = (option_type, resolved_id)
        if pair not in diagnostic_options:
            diagnostic_options.append(pair)
    diagnostic_options_tuple = tuple(diagnostic_options)

    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device("cpu")
    rows, unsupported = load_pool_rows(args.pool.resolve(), [])
    if unsupported or len(rows) != 6:
        raise ValueError("fixed CPU evaluation requires exactly six supported deck rows")
    all_decks = [
        read_deck((WORKSPACE_ROOT / str(row["directory"])).resolve() / "deck.csv")
        for row in rows
    ]
    all_opponent_names = [str(row["name"]) for row in rows]
    if args.opponent:
        unknown = sorted(set(args.opponent) - set(all_opponent_names))
        if unknown:
            raise ValueError(f"unknown fixed-pool opponents: {unknown}")
        requested = set(args.opponent)
        opponent_ids = [
            index
            for index, name in enumerate(all_opponent_names)
            if name in requested
        ]
    else:
        opponent_ids = list(range(len(rows)))
    opponent_decks = [all_decks[index] for index in opponent_ids]
    opponent_names = [all_opponent_names[index] for index in opponent_ids]
    learner_deck_path = (
        args.learner_deck.resolve()
        if args.learner_deck is not None
        else (WORKSPACE_ROOT / str(rows[0]["directory"])).resolve() / "deck.csv"
    )
    learner_deck = read_deck(learner_deck_path)
    foundation_directory = (WORKSPACE_ROOT / str(rows[0]["directory"])).resolve()
    foundation_checkpoint = resolve_pool_path(
        foundation_directory, str(rows[0]["checkpoint"])
    )
    ontology = resolve_pool_path(foundation_directory, str(rows[0]["ontology"]))
    initial_model, _config = load_foundation_0020_checkpoint(
        foundation_checkpoint, ontology, device
    )
    initial_model.eval().requires_grad_(False)
    candidate_model, candidate_iteration = load_candidate_model(
        args.candidate_checkpoint.resolve(),
        foundation_checkpoint,
        ontology,
        initial_model,
    )
    shim = SeededCppLib(args.lib.resolve())
    shim.set_batch_threads(args.engine_threads)

    results = []
    model_rows = [("candidate", candidate_model, candidate_iteration)]
    if not args.candidate_only:
        model_rows.insert(0, ("initial", initial_model, 0))
    for label, model, iteration in model_rows:
        result = evaluate_label(
            label=label,
            learner_model=model,
            opponent_model=initial_model,
            iteration=iteration,
            shim=shim,
            learner_deck=learner_deck,
            opponent_decks=opponent_decks,
            opponent_names=opponent_names,
            opponent_ids=opponent_ids,
            games_per_opponent=args.games_per_opponent,
            batch_size=args.batch_size,
            seed_start=args.seed_start,
            max_steps=args.max_steps,
            max_select=args.max_select,
            diagnostic_options=diagnostic_options_tuple,
        )
        results.append(result)
        print("CPU_EVAL_RESULT " + json.dumps(result, sort_keys=True), flush=True)

    baseline = next((row for row in results if row["label"] == "initial"), None)
    for result in results:
        if baseline is None or result is baseline:
            continue
        result["effective_win_rate_delta_vs_initial"] = (
            result["overall"]["effective_win_rate"]
            - baseline["overall"]["effective_win_rate"]
        )
        result["per_opponent_effective_delta_vs_initial"] = {
            name: (
                result["per_opponent"][name]["effective_win_rate"]
                - baseline["per_opponent"][name]["effective_win_rate"]
            )
            for name in opponent_names
        }
    report = {
        "schema_version": "foundation_ppo_fixed_cpu_matchups_v2",
        "contract": {
            "engine": "official_seeded_cpp_cpu",
            "learner_player": 0,
            "learner_deck": str(learner_deck_path),
            "learner_deck_sha256": sha256_file(learner_deck_path),
            "opponents": "six_decks_with_frozen_initial_foundation",
            "opponent_player": 1,
            "decode": "greedy_argmax_fp32_cpu",
            "games_per_opponent": args.games_per_opponent,
            "total_games_per_result": args.games_per_opponent * len(opponent_names),
            "seed_start": args.seed_start,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "opponent_names": opponent_names,
            "opponent_ids": opponent_ids,
            "candidate_only": args.candidate_only,
            "diagnostic_options": [
                f"{option_type}:{resolved_id}"
                for option_type, resolved_id in diagnostic_options_tuple
            ],
        },
        "candidate_checkpoint": str(args.candidate_checkpoint.resolve()),
        "candidate_checkpoint_sha256": sha256_file(args.candidate_checkpoint.resolve()),
        "foundation_checkpoint": str(foundation_checkpoint),
        "foundation_checkpoint_sha256": sha256_file(foundation_checkpoint),
        "cpu_engine": str(args.lib.resolve()),
        "cpu_engine_sha256": sha256_file(args.lib.resolve()),
        "torch_threads": args.torch_threads,
        "engine_threads": args.engine_threads,
        "results": results,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("CPU_EVAL_COMPLETE " + json.dumps(report["results"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
