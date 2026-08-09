"""CUDA-engine seeded-2048 zero-shot evaluation for all Frozen-0806 decks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import os
import subprocess
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.frozen_0806_contract import (
    FROZEN_0806_CONTRACT_ID,
    FROZEN_0806_EVALUATION_GAMES,
    FROZEN_0806_EVALUATION_SEED,
    FROZEN_0806_EVALUATION_UNITS,
    FROZEN_0806_FIRST_PLAYER_CONTRACT,
    evaluation_coin_winner,
    evaluation_game_seed,
)
POOL_ROOT = ROOT / "evaluation/arena/frozen_pools/0806_kaggle_top100_plus_v1"
POLICY_ROOT = POOL_ROOT / "policies/policy_0806"
POLICY_MODEL = (
    ROOT / "archive/pretrained/0031_friend_0806_epoch11_best_validation_loss/model.pt"
)
BENCHMARK = ROOT / "engine_cuda/tools/benchmark_0037_cuda_resident_refill.py"
CANONICAL_OUTPUT_ROOT = (
    ROOT
    / "evaluation/arena/combat_mat/policy_0806"
    / "0806_kaggle_top100_plus_v1_cuda_seeded_2048_agent_choice_v3"
)
CANONICAL_TEMP_ROOT = (
    ROOT / ".tmp/evaluation/policy_0806_cuda_seeded2048_agent_choice_v3"
)
OUTPUT_ROOT = CANONICAL_OUTPUT_ROOT
TEMP_ROOT = CANONICAL_TEMP_ROOT
EVALUATION_SEED = FROZEN_0806_EVALUATION_SEED
EXPECTED_DECKS = 55
EXPECTED_GAMES = FROZEN_0806_EVALUATION_GAMES
CUDA_LANE_COUNT = 256
POLICY_SHA256 = "0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8"
ACTOR_MODEL = POLICY_MODEL
ACTOR_SHA256 = POLICY_SHA256
ACTOR_LABEL = "Policy-0806"
ACTOR_MODE = "0806"


def build_cuda_schedule(
    *,
    focal_deck_id: str,
    entries: Iterable[Any],
    evaluation_seed: int | None = None,
) -> dict[str, Any]:
    evaluation_seed = EVALUATION_SEED if evaluation_seed is None else int(evaluation_seed)
    if evaluation_seed < 0:
        raise ValueError("evaluation_seed cannot be negative")
    jobs: list[dict[str, Any]] = []
    entries = tuple(entries)
    if not entries or any(int(entry.games) < 1 for entry in entries):
        raise ValueError("Frozen-0806 schedule entries must contain positive slots")
    for replica in range(FROZEN_0806_EVALUATION_UNITS):
        for entry in entries:
            for slot in range(int(entry.games)):
                jobs.append(
                    {
                        "game_id": f"r{replica + 1:02d}-{entry.deck_id}-{slot + 1:03d}",
                        "opponent_id": str(entry.deck_id),
                        "replica": replica,
                        "slot": slot,
                        "engine_seed": evaluation_game_seed(
                            evaluation_seed=evaluation_seed,
                            focal_identity=focal_deck_id,
                            opponent_identity=str(entry.deck_id),
                            slot=slot,
                            replica=replica,
                        ),
                        "search_seed": evaluation_game_seed(
                            evaluation_seed=evaluation_seed,
                            focal_identity=focal_deck_id,
                            opponent_identity=str(entry.deck_id),
                            slot=slot,
                            replica=replica,
                            namespace="search",
                        ),
                        "focal_won_toss": evaluation_coin_winner(
                            evaluation_seed=evaluation_seed,
                            focal_identity=focal_deck_id,
                            opponent_identity=str(entry.deck_id),
                            slot=slot,
                            replica=replica,
                        ),
                    }
                )
    payload = {
        "schema": "policy_0806_cuda_seeded2048_agent_choice_v3",
        "contract_id": FROZEN_0806_CONTRACT_ID,
        "first_player_contract": FROZEN_0806_FIRST_PLAYER_CONTRACT,
        "evaluation_seed": evaluation_seed,
        "focal_deck_id": focal_deck_id,
        "jobs": jobs,
    }
    payload["schedule_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_output_roots(
    *, output_root: Path | None = None, temp_root: Path | None = None
) -> None:
    """Select an isolated evidence namespace without overwriting history."""

    global OUTPUT_ROOT, TEMP_ROOT
    selected_output = (output_root or CANONICAL_OUTPUT_ROOT).resolve()
    selected_temp = (temp_root or CANONICAL_TEMP_ROOT).resolve()
    allowed_output = (
        ROOT / "evaluation/arena/combat_mat/policy_0806"
    ).resolve()
    allowed_temp = (ROOT / ".tmp/evaluation").resolve()
    if not selected_output.is_relative_to(allowed_output):
        raise ValueError("CUDA evaluation output must stay under Policy-0806 combat_mat")
    if not selected_temp.is_relative_to(allowed_temp):
        raise ValueError("CUDA evaluation temp root must stay under .tmp/evaluation")
    OUTPUT_ROOT = selected_output
    TEMP_ROOT = selected_temp


def configure_actor(*, checkpoint: Path, label: str) -> None:
    """Configure a strict 0031 focal actor while keeping opponents at 0806."""

    global ACTOR_MODEL, ACTOR_SHA256, ACTOR_LABEL, ACTOR_MODE
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    ACTOR_MODEL = checkpoint
    ACTOR_SHA256 = _sha256(checkpoint)
    ACTOR_LABEL = str(label).strip()
    if not ACTOR_LABEL:
        raise ValueError("actor label cannot be empty")
    ACTOR_MODE = "0806" if ACTOR_SHA256 == POLICY_SHA256 else "pretrained"


def _catalog() -> tuple[Any, tuple[Any, ...]]:
    from evaluation.frozen_0806_runtime import load_frozen_0806_runtime_catalog

    catalog = load_frozen_0806_runtime_catalog(opponent_policy_label="0806")
    ordered = tuple(
        sorted(
            catalog.candidates,
            key=lambda item: int(item.package_manifest["frozen_deck_number"]),
        )
    )
    if len(ordered) != EXPECTED_DECKS:
        raise RuntimeError("Frozen-0806 catalog is not exactly 55 decks")
    return catalog, ordered


def _result_path(number: str) -> Path:
    return TEMP_ROOT / "results" / f"{number}.json"


def _schedule_path(number: str) -> Path:
    return TEMP_ROOT / "schedules" / f"{number}.json"


def resolve_candidate_deck_path(candidate: Any, deck_root: Path = POOL_ROOT / "decks") -> Path:
    from evaluation.frozen_0806 import exact_deck_sha256

    number = str(candidate.package_manifest["frozen_deck_number"])
    matches = sorted(path for path in deck_root.glob(f"{number}_*") if path.is_dir())
    if len(matches) != 1:
        raise RuntimeError(f"Frozen deck {number} does not resolve to exactly one directory")
    deck_path = matches[0] / "deck.csv"
    cards = tuple(
        int(value)
        for value in deck_path.read_text(encoding="ascii").splitlines()
        if value.strip()
    )
    expected_cards = tuple(int(value) for value in candidate.deck)
    expected_hash = str(candidate.package_manifest["exact_deck_sha256"])
    if cards != expected_cards or exact_deck_sha256(cards) != expected_hash:
        raise RuntimeError(f"Frozen deck {number} failed exact-deck identity validation")
    return deck_path


def merge_cuda_chunk_results(
    chunks: Iterable[dict[str, Any]], *, expected_games: int
) -> dict[str, Any]:
    ordered = sorted(chunks, key=lambda item: int(item["schedule"]["schedule_offset"]))
    if not ordered:
        raise ValueError("at least one CUDA chunk is required")
    cursor = 0
    game_results: list[int] = []
    terminal_hashes: list[str] = []
    forfeits: list[int] = []
    wall_seconds = gpu_seconds = 0.0
    routed_rows = completed_games = errors = 0
    peak_allocated = peak_reserved = 0
    terminal_turns: list[int] = []
    engine_selections: list[int] = []
    terminal_prize_counts: list[list[int]] = []
    first_player_choosers: list[int] = []
    first_player_actions: list[int] = []
    actual_first_players: list[int] = []
    diagnostic_chunks = 0
    for chunk in ordered:
        collector = chunk["collector"]
        schedule = chunk["schedule"]
        offset = int(schedule["schedule_offset"])
        games = int(collector["games"])
        results = list(chunk["determinism"]["game_results"])
        if (
            chunk.get("passed") is not True
            or offset != cursor
            or games != len(results)
            or int(collector["completed_games"]) != games
            or int(collector["errors"]) != 0
        ):
            raise ValueError("CUDA chunks are incomplete or non-contiguous")
        cursor += games
        completed_games += int(collector["completed_games"])
        errors += int(collector["errors"])
        wall_seconds += float(collector["wall_seconds"])
        gpu_seconds += float(collector["gpu_seconds"])
        routed_rows += int(collector["routed_ready_rows"])
        game_results.extend(int(value) for value in results)
        terminal_hashes.append(str(chunk["determinism"]["terminal_state_sha256"]))
        diagnostics = chunk.get("per_game_diagnostics")
        if diagnostics is not None:
            diagnostic_chunks += 1
            if (
                not isinstance(diagnostics, dict)
                or len(diagnostics.get("terminal_turns", [])) != games
                or len(diagnostics.get("engine_selections", [])) != games
                or len(diagnostics.get("terminal_prize_counts", [])) != games
                or len(diagnostics.get("first_player_choosers", [])) != games
                or len(diagnostics.get("first_player_actions", [])) != games
                or len(diagnostics.get("actual_first_players", [])) != games
            ):
                raise ValueError("CUDA per-game diagnostics do not match chunk size")
            terminal_turns.extend(int(value) for value in diagnostics["terminal_turns"])
            engine_selections.extend(
                int(value) for value in diagnostics["engine_selections"]
            )
            terminal_prize_counts.extend(
                [int(row[0]), int(row[1])]
                for row in diagnostics["terminal_prize_counts"]
            )
            first_player_choosers.extend(
                int(value) for value in diagnostics["first_player_choosers"]
            )
            first_player_actions.extend(
                int(value) for value in diagnostics["first_player_actions"]
            )
            actual_first_players.extend(
                int(value) for value in diagnostics["actual_first_players"]
            )
        forfeits.extend(
            int(value)
            for value in chunk["progress_guard"]["forfeit_schedule_indices"]
        )
        peak_allocated = max(
            peak_allocated, int(chunk["memory"]["torch_peak_allocated_bytes"])
        )
        peak_reserved = max(
            peak_reserved, int(chunk["memory"]["torch_peak_reserved_bytes"])
        )
    if cursor != expected_games:
        raise ValueError(f"CUDA chunks cover {cursor}, expected {expected_games}")
    if diagnostic_chunks not in {0, len(ordered)}:
        raise ValueError("CUDA chunks must either all include diagnostics or all omit them")
    merged = copy.deepcopy(ordered[0])
    merged["schema_version"] = "cuda_semantic0031_rollout_chunked_v1"
    merged["collector"].update(
        {
            "games": expected_games,
            "completed_games": completed_games,
            "errors": errors,
            "wall_seconds": wall_seconds,
            "gpu_seconds": gpu_seconds,
            "games_per_second_wall": expected_games / wall_seconds,
            "games_per_second_gpu": expected_games / gpu_seconds,
            "routed_ready_rows": routed_rows,
            "chunk_count": len(ordered),
            "cuda_batch_size": max(int(item["collector"]["games"]) for item in ordered),
        }
    )
    encoded_results = json.dumps(game_results, separators=(",", ":")).encode("ascii")
    encoded_terminal = json.dumps(terminal_hashes, separators=(",", ":")).encode(
        "ascii"
    )
    merged["determinism"].update(
        {
            "game_results": game_results,
            "game_results_sha256": hashlib.sha256(encoded_results).hexdigest(),
            "terminal_state_sha256": hashlib.sha256(encoded_terminal).hexdigest(),
            "terminal_state_chunk_sha256": terminal_hashes,
        }
    )
    merged["progress_guard"]["forfeit_schedule_indices"] = sorted(forfeits)
    merged["progress_guard"]["forfeit_count"] = len(forfeits)
    merged["schedule"].update(
        {"schedule_offset": 0, "used_jobs": expected_games}
    )
    merged["memory"].update(
        {
            "torch_peak_allocated_bytes": peak_allocated,
            "torch_peak_reserved_bytes": peak_reserved,
        }
    )
    if terminal_turns:
        merged["per_game_diagnostics"] = {
            "schema": "cuda_resident_terminal_diagnostics_v2_agent_first_player",
            "terminal_turns": terminal_turns,
            "engine_selections": engine_selections,
            "terminal_prize_counts": terminal_prize_counts,
            "first_player_choosers": first_player_choosers,
            "first_player_actions": first_player_actions,
            "actual_first_players": actual_first_players,
        }
    return merged


def build_cuda_game_records(
    schedule: dict[str, Any], result: dict[str, Any]
) -> list[dict[str, Any]]:
    """Materialize the immutable per-game evidence behind an aggregate report."""

    jobs = list(schedule.get("jobs", []))
    outcomes = list(result.get("determinism", {}).get("game_results", []))
    if len(jobs) != len(outcomes):
        raise ValueError("CUDA schedule/result lengths differ")
    progress_guard = result.get("progress_guard", {})
    forfeits = {
        int(index) for index in progress_guard.get("forfeit_schedule_indices", [])
    }
    turn_limit_draws = {
        int(index)
        for index in progress_guard.get("turn_limit_draw_schedule_indices", [])
    }
    diagnostics = result.get("per_game_diagnostics")
    collector = result.get("collector")
    successful_engine_run = (
        result.get("passed") is True
        and isinstance(collector, dict)
        and int(collector.get("completed_games", -1)) == len(jobs)
        and int(collector.get("errors", -1)) == 0
    )
    terminal_turns: list[Any] = []
    engine_selections: list[Any] = []
    terminal_prize_counts: list[Any] = []
    first_player_choosers: list[Any] = []
    first_player_actions: list[Any] = []
    actual_first_players: list[Any] = []
    if diagnostics is not None:
        if not isinstance(diagnostics, dict):
            raise ValueError("CUDA per-game diagnostics must be a mapping")
        terminal_turns = list(diagnostics.get("terminal_turns", []))
        engine_selections = list(diagnostics.get("engine_selections", []))
        terminal_prize_counts = list(diagnostics.get("terminal_prize_counts", []))
        first_player_choosers = list(diagnostics.get("first_player_choosers", []))
        first_player_actions = list(diagnostics.get("first_player_actions", []))
        actual_first_players = list(diagnostics.get("actual_first_players", []))
        if not (
            len(terminal_turns)
            == len(engine_selections)
            == len(terminal_prize_counts)
            == len(first_player_choosers)
            == len(first_player_actions)
            == len(actual_first_players)
            == len(jobs)
        ):
            raise ValueError("CUDA per-game diagnostics/result lengths differ")
    records: list[dict[str, Any]] = []
    for index, (job, raw_result) in enumerate(zip(jobs, outcomes, strict=True)):
        focal_player = 0 if bool(job["focal_won_toss"]) else 1
        won = (focal_player == 0 and raw_result == 1) or (
            focal_player == 1 and raw_result == 2
        )
        lost = (focal_player == 0 and raw_result == 2) or (
            focal_player == 1 and raw_result == 1
        )
        record = {
            **job,
            "schedule_index": index,
            "raw_engine_result": int(raw_result),
            "focal_outcome": "win" if won else "loss" if lost else "draw",
            "repeat_forfeit": index in forfeits,
            "turn_limit_draw": index in turn_limit_draws,
        }
        if successful_engine_run:
            # This is derived from the fail-closed benchmark contract: any
            # engine status=error aborts without publishing a passed result.
            record["error"] = False
            record["continuation_error"] = False
        if diagnostics is not None:
            raw_turn = int(terminal_turns[index])
            raw_prizes = terminal_prize_counts[index]
            if (
                not isinstance(raw_prizes, (list, tuple))
                or len(raw_prizes) != 2
            ):
                raise ValueError(f"invalid terminal prize counts at game {index}")
            prize_counts = (int(raw_prizes[0]), int(raw_prizes[1]))
            focal_prizes = prize_counts[focal_player]
            opponent_prizes = prize_counts[1 - focal_player]
            chooser = int(first_player_choosers[index])
            first_player_action = int(first_player_actions[index])
            actual_first = int(actual_first_players[index])
            if (
                chooser not in (0, 1)
                or first_player_action not in (0, 1)
                or actual_first not in (0, 1)
                or chooser != 0
                or actual_first
                != (chooser if first_player_action == 0 else 1 - chooser)
            ):
                raise ValueError(f"invalid Agent first-player choice at game {index}")
            record.update(
                {
                    "terminal_turn": raw_turn,
                    "game_length": (raw_turn + 1) // 2,
                    "engine_selections": int(engine_selections[index]),
                    "terminal_prize_counts": list(prize_counts),
                    # (focal prizes taken - opponent prizes taken); the common
                    # six-Prize initial count cancels.
                    "prize_differential": opponent_prizes - focal_prizes,
                    "first_player_chooser": chooser,
                    "first_player_action": first_player_action,
                    "winner_chose_first": first_player_action == 0,
                    "actual_first_player": actual_first,
                    "focal_first": actual_first == focal_player,
                }
            )
        records.append(record)
    return records


def _game_evidence(candidate: Any, result: dict[str, Any]) -> dict[str, Any]:
    number = str(candidate.package_manifest["frozen_deck_number"])
    schedule = json.loads(_schedule_path(number).read_text(encoding="utf-8"))
    return {
        "schema": "policy_0806_cuda_seeded2048_games_v3_agent_first_player",
        "contract_id": (
            FROZEN_0806_CONTRACT_ID
            if EVALUATION_SEED == FROZEN_0806_EVALUATION_SEED
            else "gate_g_independent_seed_agent_first_player_v2"
        ),
        "first_player_contract": FROZEN_0806_FIRST_PLAYER_CONTRACT,
        "evaluation_seed": EVALUATION_SEED,
        "deck_number": number,
        "deck_id": candidate.name,
        "exact_deck_sha256": candidate.package_manifest["exact_deck_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "game_results_sha256": result["determinism"]["game_results_sha256"],
        "games": build_cuda_game_records(schedule, result),
    }


def _cached_result_is_reusable(
    cached: dict[str, Any], *, schedule_file_sha256: str
) -> bool:
    """Fail closed on stale schema, diagnostics, seed schedule, or model state."""

    device = cached.get("device", {})
    diagnostics = cached.get("per_game_diagnostics", {})
    return (
        cached.get("passed") is True
        and cached.get("schema_version")
        == "cuda_semantic0031_resident_refill_strict_fp32_v3_agent_first_player"
        and cached.get("collector", {}).get("completed_games") == EXPECTED_GAMES
        and cached.get("collector", {}).get("errors") == 0
        and cached.get("models", {}).get("actor_checkpoint_sha256") == ACTOR_SHA256
        and cached.get("models", {}).get("opponent_checkpoint_sha256") == POLICY_SHA256
        and cached.get("schedule", {}).get("sha256") == schedule_file_sha256
        and device.get("float32_matmul_precision") == "highest"
        and device.get("matmul_allow_tf32") is False
        and device.get("cudnn_allow_tf32") is False
        and diagnostics.get("schema")
        == "cuda_resident_terminal_diagnostics_v2_agent_first_player"
        and len(diagnostics.get("terminal_turns", [])) == EXPECTED_GAMES
        and len(diagnostics.get("engine_selections", [])) == EXPECTED_GAMES
        and len(diagnostics.get("terminal_prize_counts", [])) == EXPECTED_GAMES
        and len(diagnostics.get("first_player_choosers", [])) == EXPECTED_GAMES
        and len(diagnostics.get("first_player_actions", [])) == EXPECTED_GAMES
        and len(diagnostics.get("actual_first_players", [])) == EXPECTED_GAMES
        and all(value in (0, 1) for value in diagnostics.get("first_player_choosers", []))
        and all(value in (0, 1) for value in diagnostics.get("first_player_actions", []))
        and all(value in (0, 1) for value in diagnostics.get("actual_first_players", []))
    )


def _run_one(catalog: Any, candidate: Any) -> dict[str, Any]:
    number = str(candidate.package_manifest["frozen_deck_number"])
    focal_deck_path = resolve_candidate_deck_path(candidate)
    result_path = _result_path(number)
    schedule_path = _schedule_path(number)
    schedule = build_cuda_schedule(
        focal_deck_id=candidate.name,
        entries=catalog.pool.schedule,
    )
    if len(schedule["jobs"]) != EXPECTED_GAMES:
        raise RuntimeError(f"deck {number} schedule is not 2048 games")
    _atomic_json(schedule_path, schedule)
    if result_path.is_file():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if not _cached_result_is_reusable(
            cached, schedule_file_sha256=_sha256(schedule_path)
        ):
            result_path.unlink()
    if not result_path.is_file():
        log_path = TEMP_ROOT / "logs" / f"{number}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            command = [
                sys.executable,
                str(BENCHMARK),
                "--actor-mode",
                ACTOR_MODE,
                "--actor-package",
                str(POLICY_ROOT),
                "--actor-checkpoint",
                str(ACTOR_MODEL),
                "--opponent-model",
                str(POLICY_MODEL),
                "--focal-deck",
                str(focal_deck_path),
                "--deck-root",
                str(POOL_ROOT / "decks"),
                "--schedule",
                str(schedule_path),
                "--game-limit",
                str(EXPECTED_GAMES),
                "--lane-count",
                str(CUDA_LANE_COUNT),
                "--check-interval",
                "8",
                "--ability-repeat-limit",
                "20",
                "--output",
                str(result_path),
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if completed.returncode != 0:
                result_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"resident CUDA evaluation failed for deck {number}; see {log_path}"
                )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    diagnostics = result.get("per_game_diagnostics", {})
    if (
        result.get("passed") is not True
        or result.get("schema_version")
        != "cuda_semantic0031_resident_refill_strict_fp32_v3_agent_first_player"
        or result.get("collector", {}).get("completed_games") != EXPECTED_GAMES
        or result.get("collector", {}).get("errors") != 0
        or len(result.get("determinism", {}).get("game_results", []))
        != EXPECTED_GAMES
        or result.get("models", {}).get("actor_checkpoint_sha256") != ACTOR_SHA256
        or result.get("models", {}).get("opponent_checkpoint_sha256") != POLICY_SHA256
        or result.get("schedule", {}).get("sha256") != _sha256(schedule_path)
        or result.get("device", {}).get("float32_matmul_precision") != "highest"
        or result.get("device", {}).get("matmul_allow_tf32") is not False
        or result.get("device", {}).get("cudnn_allow_tf32") is not False
        or diagnostics.get("schema")
        != "cuda_resident_terminal_diagnostics_v2_agent_first_player"
        or len(diagnostics.get("terminal_turns", [])) != EXPECTED_GAMES
        or len(diagnostics.get("engine_selections", [])) != EXPECTED_GAMES
        or len(diagnostics.get("terminal_prize_counts", [])) != EXPECTED_GAMES
        or len(diagnostics.get("first_player_choosers", [])) != EXPECTED_GAMES
        or len(diagnostics.get("first_player_actions", [])) != EXPECTED_GAMES
        or len(diagnostics.get("actual_first_players", [])) != EXPECTED_GAMES
        or any(value not in (0, 1) for value in diagnostics.get("first_player_choosers", []))
        or any(value not in (0, 1) for value in diagnostics.get("first_player_actions", []))
        or any(value not in (0, 1) for value in diagnostics.get("actual_first_players", []))
    ):
        raise RuntimeError(f"CUDA result failed the seeded-2048 contract: deck {number}")
    return result


def _summary(catalog: Any, candidate: Any, result: dict[str, Any]) -> dict[str, Any]:
    number = str(candidate.package_manifest["frozen_deck_number"])
    schedule = json.loads(_schedule_path(number).read_text(encoding="utf-8"))
    game_records = build_cuda_game_records(schedule, result)
    by_opponent: dict[str, dict[str, int]] = {}
    first = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    second = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    wins = losses = draws = 0
    toss = {"focal_won": 0, "opponent_won": 0}
    choices = {"first": 0, "second": 0}
    for game in game_records:
        won = game["focal_outcome"] == "win"
        lost = game["focal_outcome"] == "loss"
        outcome = "wins" if won else "losses" if lost else "draws"
        wins += int(won)
        losses += int(lost)
        draws += int(not won and not lost)
        seat = first if game["focal_first"] else second
        seat["games"] += 1
        seat[outcome] += 1
        toss["focal_won" if game["focal_won_toss"] else "opponent_won"] += 1
        choices["first" if game["winner_chose_first"] else "second"] += 1
        row = by_opponent.setdefault(
            game["opponent_id"],
            {"games": 0, "wins": 0, "losses": 0, "draws": 0},
        )
        row["games"] += 1
        row[outcome] += 1
    if wins + losses + draws != EXPECTED_GAMES:
        raise RuntimeError(f"deck {number} outcome total is invalid")
    schedule_entry = next(
        entry for entry in catalog.pool.schedule if entry.deck_id == candidate.name
    )
    return {
        "deck_number": number,
        "deck_id": candidate.name,
        "display_name": candidate.display_name,
        "representative_cards": list(candidate.representative_cards),
        "schedule_games": int(schedule_entry.games),
        "best_rank": int(schedule_entry.best_rank),
        "observed_players": int(schedule_entry.observed_players),
        "segment": str(schedule_entry.segment),
        "exact_deck_sha256": candidate.package_manifest["exact_deck_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "games": EXPECTED_GAMES,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / EXPECTED_GAMES,
        "first": first,
        "second": second,
        "toss": toss,
        "first_player_choices": choices,
        "by_opponent": by_opponent,
        "progress_guard_forfeits": result["progress_guard"][
            "forfeit_schedule_indices"
        ],
        "turn_limit_draws": result["progress_guard"].get(
            "turn_limit_draw_schedule_indices", []
        ),
        "turn_limit_draw_contract": (
            result["progress_guard"].get("engine_turn_draw_limit") == 100
        ),
        "wall_seconds": result["collector"]["wall_seconds"],
        "games_per_second": result["collector"]["games_per_second_wall"],
        "lane_count": int(result["collector"].get("lane_count", 0)),
        "refill_events": int(result["collector"].get("refill_events", 0)),
        "peak_reserved_bytes": result["memory"]["torch_peak_reserved_bytes"],
        "game_results_sha256": result["determinism"]["game_results_sha256"],
        "terminal_state_sha256": result["determinism"]["terminal_state_sha256"],
    }


def _baseline_comparison(number: str, games: list[dict[str, Any]]) -> dict[str, Any] | None:
    if ACTOR_SHA256 == POLICY_SHA256:
        return None
    baseline_root = CANONICAL_OUTPUT_ROOT
    baseline_manifest_path = baseline_root / "manifest.json"
    baseline_games_path = baseline_root / "games" / f"{number}.json"
    if not baseline_manifest_path.is_file() or not baseline_games_path.is_file():
        return None
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    baseline_reports = {
        row["deck_number"]: row for row in baseline_manifest.get("reports", [])
    }
    baseline_summary = baseline_reports.get(number)
    if baseline_summary is None:
        return None
    baseline_games = json.loads(
        baseline_games_path.read_text(encoding="utf-8")
    ).get("games", [])
    if len(baseline_games) != len(games):
        raise RuntimeError(f"deck {number} baseline/current game counts differ")
    identity_fields = (
        "game_id", "opponent_id", "replica", "slot", "engine_seed",
        "search_seed", "focal_won_toss",
    )
    for index, (baseline, current) in enumerate(
        zip(baseline_games, games, strict=True)
    ):
        if any(baseline.get(key) != current.get(key) for key in identity_fields):
            raise RuntimeError(
                f"deck {number} baseline/current schedule differs at game {index}"
            )
    flips = Counter(
        (baseline["focal_outcome"], current["focal_outcome"])
        for baseline, current in zip(baseline_games, games, strict=True)
    )
    current_win_rate = sum(game["focal_outcome"] == "win" for game in games) / len(games)
    baseline_win_rate = float(baseline_summary["win_rate"])
    return {
        "baseline_actor_label": "Policy-0806",
        "baseline_actor_sha256": POLICY_SHA256,
        "baseline_win_rate": baseline_win_rate,
        "current_win_rate": current_win_rate,
        "delta_win_rate": current_win_rate - baseline_win_rate,
        "paired_same_outcome": sum(
            count for (before, after), count in flips.items() if before == after
        ),
        "baseline_loss_to_current_win": flips[("loss", "win")],
        "baseline_win_to_current_loss": flips[("win", "loss")],
        "baseline_draw_to_current_win": flips[("draw", "win")],
        "baseline_win_to_current_draw": flips[("win", "draw")],
        "all_outcome_pairs": {
            f"{before}_to_{after}": count
            for (before, after), count in sorted(flips.items())
        },
    }


def _deck_overview(candidate: Any, summary: dict[str, Any]) -> str:
    from evaluation.cards import card_image_url, load_card_catalog

    cards = load_card_catalog(ROOT / "data/official/EN_Card_Data.csv")
    groups: dict[str, list[str]] = {"Pokémon": [], "Trainer": [], "Energy": []}
    group_counts = Counter()
    for card_id, count in sorted(
        Counter(candidate.deck).items(),
        key=lambda item: (cards[item[0]]["name"], item[0]),
    ):
        card = cards[card_id]
        stage_or_type = str(card.get("stage_or_type", ""))
        if "Pokémon" in stage_or_type:
            group = "Pokémon"
        elif "Energy" in stage_or_type or "Energy" in str(card["name"]):
            group = "Energy"
        else:
            group = "Trainer"
        image_url = card_image_url(card["expansion"], card["collection_number"])
        image = (
            f'<img src="{html.escape(image_url, quote=True)}" '
            f'alt="{html.escape(card["name"], quote=True)}" loading="lazy" '
            f'onerror="this.hidden=true">'
            if image_url
            else ""
        )
        groups[group].append(
            f'<div class="deck-entry">{image}<span>'
            f'<span class="deck-card-name">{html.escape(card["name"])}</span>'
            f'<span class="deck-card-set">{html.escape(card["expansion"])} '
            f'{html.escape(card["collection_number"])} · ID {card_id}</span></span>'
            f'<span class="deck-card-count">×{count}</span></div>'
        )
        group_counts[group] += count
    representative_images = "".join(
        f'<img src="{html.escape(str(card["image_url"]), quote=True)}" '
        f'alt="{html.escape(str(card["name"]), quote=True)}" loading="eager" '
        f'onerror="this.hidden=true">'
        for card in candidate.representative_cards
    )
    group_html = "".join(
        f'<div class="deck-group"><div class="deck-group-head"><h3>{group}</h3>'
        f'<span class="deck-count">{group_counts[group]} 张</span></div>'
        f'<div class="deck-list">{"".join(groups[group])}</div></div>'
        for group in ("Pokémon", "Trainer", "Energy")
    )
    return (
        '<section class="candidate-overview"><div class="candidate-lead"><div>'
        '<p class="candidate-kicker">主视角卡组</p>'
        f'<h2 class="candidate-title">{html.escape(summary["deck_number"])} · '
        f'{html.escape(summary["display_name"])}</h2>'
        f'<div class="candidate-meta"><span>{html.escape(ACTOR_LABEL)}</span><span>Exact 60 cards</span>'
        '<span>Frozen Arena · 0806_kaggle_top100_plus_v1</span></div></div>'
        f'<div class="candidate-representatives">{representative_images}</div></div>'
        f'<div class="deck-groups">{group_html}</div></section>'
    )


def _report_html(
    catalog: Any,
    candidate: Any,
    summary: dict[str, Any],
    *,
    report_label: str = "Policy-0806 CUDA Seeded-2048",
    actor_label: str | None = None,
    actor_sha256: str | None = None,
    back_href: str = "../index.html",
    evaluation_seed: int = EVALUATION_SEED,
) -> str:
    actor_label = ACTOR_LABEL if actor_label is None else actor_label
    actor_sha256 = ACTOR_SHA256 if actor_sha256 is None else actor_sha256
    matchup_rows = []
    matchup_bars = []
    hero_images = "".join(
        f'<img src="{html.escape(str(card["image_url"]), quote=True)}" '
        f'alt="{html.escape(str(card["name"]), quote=True)}" loading="eager">'
        for card in candidate.representative_cards
    )
    for opponent in sorted(
        catalog.opponents,
        key=lambda item: int(item.package_manifest["frozen_deck_number"]),
    ):
        row = summary["by_opponent"][opponent.name]
        rate = row["wins"] / row["games"] if row["games"] else 0.0
        images = "".join(
            f'<img class="opponent-thumb" src="{html.escape(str(card["image_url"]), quote=True)}" '
            f'alt="{html.escape(str(card["name"]), quote=True)}" loading="lazy">'
            for card in opponent.representative_cards
        )
        identity = (
            f'<span class="deck-number">{opponent.package_manifest["frozen_deck_number"]}</span>'
            f'<span class="opponent-thumbnails">{images}</span>'
            f'<span class="opponent-name">{html.escape(opponent.display_name or opponent.name)}</span>'
        )
        matchup_bars.append(
            f'<div class="chart-row"><div class="opponent-identity">{identity}</div>'
            f'<div class="bar-track"><span class="bar" style="width:{rate:.4%}"></span></div>'
            f'<div class="chart-result"><span class="chart-rate">{rate:.1%}</span>'
            f'<span class="chart-record">{row["wins"]}-{row["losses"]}-{row["draws"]}</span></div></div>'
        )
        matchup_rows.append(
            f'<tr><td>{opponent.package_manifest["frozen_deck_number"]}</td>'
            f'<td><div class="opponent-identity"><span class="opponent-thumbnails">{images}</span>'
            f'<span>{html.escape(opponent.display_name or opponent.name)}</span></div></td>'
            f"<td>{row['games']}</td><td>{row['wins']}</td>"
            f"<td>{row['losses']}</td><td>{row['draws']}</td>"
            f"<td>{rate:.2%}</td></tr>"
        )
    embedded = json.dumps(
        {
            "schema": "policy_0806_cuda_seeded2048_report_v3_agent_first_player",
            "contract_id": FROZEN_0806_CONTRACT_ID,
            "first_player_contract": FROZEN_0806_FIRST_PLAYER_CONTRACT,
            "evidence_boundary": "CUDA Frozen strength contract; official CPU transfer remains a separate deployment check",
            "evaluation_seed": evaluation_seed,
            "actor_label": actor_label,
            "actor_sha256": actor_sha256,
            "opponent_policy_sha256": POLICY_SHA256,
            "candidate": {
                "deck_number": summary["deck_number"],
                "deck_id": summary["deck_id"],
                "display_name": summary["display_name"],
                "exact_deck_sha256": summary["exact_deck_sha256"],
                "deck": list(candidate.deck),
            },
            "summary": summary,
        },
        ensure_ascii=False,
    ).replace("</", "<\\/")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{summary['deck_number']} · {html.escape(summary['display_name'])} · {html.escape(report_label)}</title><style>
:root{{--bg:#f3f7f5;--surface:#fff;--soft:#f7faf8;--ink:#172b25;--muted:#60736c;--line:#dce7e2;--brand:#217a58;--dark:#14563d;--warn:#a66a18}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 system-ui,"PingFang SC",sans-serif}}main{{max-width:1240px;margin:auto;padding:36px 28px 64px}}
.hero{{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;margin-bottom:22px;padding:30px 32px;border-radius:8px;color:#fff;background:var(--dark)}}.hero-main{{display:flex;align-items:center;gap:18px}}.hero-art{{display:flex;min-width:78px}}.hero-art img{{width:58px;height:80px;margin-right:-13px;border:2px solid #d7eee4;border-radius:7px;object-fit:cover;box-shadow:0 8px 20px #082d2066}}.eyebrow{{margin:0 0 6px;color:#c8eadb;font-size:12px;font-weight:700;letter-spacing:.12em}}.back{{color:#d8eee5;text-decoration:none;font-weight:700}}h1{{margin:4px 0 0;font-size:32px;line-height:1.2}}h2{{margin:0 0 6px;font-size:20px}}section{{margin:18px 0;padding:22px;overflow:auto;border:1px solid var(--line);border-radius:8px;background:#fff}}
.candidate-overview{{padding:0;overflow:hidden}}.candidate-lead{{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:28px;padding:24px;border-bottom:1px solid var(--line);background:#fff}}.candidate-kicker{{margin:0 0 5px;color:var(--brand);font-size:12px;font-weight:750;text-transform:uppercase}}.candidate-title{{font-size:26px}}.candidate-meta{{display:flex;flex-wrap:wrap;gap:8px 16px;margin-top:12px;color:var(--muted);font-size:12px}}.candidate-representatives{{display:flex;align-items:center;gap:10px}}.candidate-representatives img{{width:112px;aspect-ratio:2.5/3.5;object-fit:cover;border:1px solid #cbd9d2;border-radius:6px;background:#e7eeea}}.deck-groups{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0;padding:18px 24px 24px}}.deck-group{{min-width:0;padding:0 20px;border-left:1px solid var(--line)}}.deck-group:first-child{{padding-left:0;border-left:0}}.deck-group:last-child{{padding-right:0}}.deck-group-head{{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin-bottom:10px}}.deck-group-head h3{{margin:0;font-size:15px}}.deck-count{{color:var(--muted);font-size:12px}}.deck-list{{display:grid;gap:3px}}.deck-entry{{display:grid;grid-template-columns:42px minmax(0,1fr) auto;align-items:center;gap:8px;min-height:48px;padding:4px 0;border-top:1px solid #edf2ef}}.deck-entry:first-child{{border-top:0}}.deck-entry img{{width:38px;height:52px;object-fit:cover;border:1px solid #d2ddd7;border-radius:3px;background:#e7eeea}}.deck-card-name{{display:block;font-size:12px;font-weight:650;line-height:1.25;overflow-wrap:anywhere}}.deck-card-set{{display:block;color:var(--muted);font-size:10px}}.deck-card-count{{font-size:14px;font-weight:750;font-variant-numeric:tabular-nums}}
.summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}.card{{min-height:92px;padding:15px 16px;border:1px solid var(--line);border-radius:11px;background:#fff}}.label,small{{display:block;color:var(--muted);font-size:12px}}.value{{margin-top:5px;font-size:23px;font-weight:750}}.warn{{border-left:4px solid var(--warn)}}
.matchup-chart{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:2px 20px;margin-top:14px;font-size:12px}}.chart-row{{display:grid;grid-template-columns:minmax(210px,270px) 1fr 125px;gap:8px;align-items:center;padding:4px 5px;border-radius:6px}}.chart-row:hover{{background:var(--soft)}}.opponent-identity{{display:flex;align-items:center;min-width:0;gap:7px}}.deck-number{{min-width:34px;padding:2px 5px;border:1px solid #b8cec4;border-radius:4px;background:#edf6f1;color:var(--dark);font-weight:800;text-align:center}}.opponent-thumbnails{{display:flex;padding-left:3px}}.opponent-thumb{{width:29px;height:38px;margin-left:-3px;object-fit:cover;border:1px solid #bdccc5;border-radius:4px;background:#e6eee9}}.opponent-name{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.bar-track{{height:7px;overflow:hidden;border-radius:999px;background:#e4ece8}}.bar{{display:block;height:100%;border-radius:inherit;background:linear-gradient(90deg,#2a8d65,#54b184)}}.chart-result{{text-align:right}}.chart-rate{{font-weight:800}}.chart-record{{margin-left:5px;color:var(--muted);font-size:11px}}
table{{width:100%;min-width:820px;margin-top:14px;border-collapse:collapse}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right;vertical-align:middle}}th{{background:var(--soft);color:#496159;font-size:12px}}th:nth-child(2),td:nth-child(2){{text-align:left}}tr:hover td{{background:#fbfdfc}}img{{cursor:zoom-in}}code{{font-size:11px;overflow-wrap:anywhere}}.lightbox{{position:fixed;inset:0;z-index:10;display:none;place-items:center;padding:24px;background:#071c16dd}}.lightbox.open{{display:grid}}.lightbox img{{max-width:min(90vw,520px);max-height:88vh;border-radius:12px;box-shadow:0 24px 80px #0009;cursor:zoom-out}}@media(max-width:780px){{main{{padding:18px 12px}}.hero{{padding:22px 18px;align-items:flex-start}}.hero-main{{align-items:flex-start}}.hero-art img{{width:45px;height:63px}}.matchup-chart{{grid-template-columns:1fr}}.candidate-lead{{grid-template-columns:1fr;padding:18px}}.candidate-representatives img{{width:88px}}.deck-groups{{grid-template-columns:1fr;padding:10px 18px 18px}}.deck-group,.deck-group:first-child,.deck-group:last-child{{padding:14px 0;border-left:0;border-top:1px solid var(--line)}}.deck-group:first-child{{border-top:0}}}}
</style></head><body><main><div class="hero"><div class="hero-main"><span class="hero-art">{hero_images}</span><div><a class="back" href="{html.escape(back_href, quote=True)}">← 返回评测总览</a><p class="eyebrow">{html.escape(report_label)} · DECK {summary['deck_number']}</p><h1>{html.escape(summary['display_name'])}</h1></div></div><div>{summary['lane_count']} resident CUDA lanes<br>{summary['games_per_second']:.2f} games/s</div></div>
{_deck_overview(candidate, summary)}
<section class="warn"><strong>评测合同：</strong>只固定 seed；抛硬币赢家由对应 Agent 处理 context 41 并自主选择先后手，不再由 harness 分配坐席。Actor：{html.escape(actor_label)}；Opponent：immutable Policy-0806。</section>
<section class="summary"><div class="card"><span class="label">W-L-D</span><div class="value">{summary['wins']}-{summary['losses']}-{summary['draws']}</div></div><div class="card"><span class="label">总体胜率</span><div class="value">{summary['win_rate']:.2%}</div></div><div class="card"><span class="label">先攻</span><div class="value">{summary['first']['wins']}/{summary['first']['games']}</div><small>{summary['first']['wins']/summary['first']['games']:.2%}</small></div><div class="card"><span class="label">后攻</span><div class="value">{summary['second']['wins']}/{summary['second']['games']}</div><small>{summary['second']['wins']/summary['second']['games']:.2%}</small></div><div class="card"><span class="label">耗时</span><div class="value">{summary['wall_seconds']:.1f}s</div><small>{summary['games_per_second']:.2f} games/s</small></div><div class="card"><span class="label">50 回合平局</span><div class="value">{len(summary.get('turn_limit_draws', [])) if summary.get('turn_limit_draw_contract', True) else '未记录'}</div><small>{'engine turn 100' if summary.get('turn_limit_draw_contract', True) else '历史结果未启用该护栏'}</small></div><div class="card"><span class="label">循环判负</span><div class="value">{len(summary['progress_guard_forfeits'])}</div><small>同一方同 Ability 第 20 次</small></div></section>
<section><h2>对 001–055 的表现</h2><p class="label">实际对局数按 Frozen-0806 频率分布；每个固定 slot 使用 8 个独立 seed replica。先后手来自 seeded toss 后赢家 Agent 的真实选择，不做人为平衡。</p><div class="matchup-chart">{''.join(matchup_bars)}</div><table><thead><tr><th>编号</th><th>对手卡组</th><th>对局</th><th>胜</th><th>负</th><th>平</th><th>胜率</th></tr></thead><tbody>{''.join(matchup_rows)}</tbody></table></section>
<section><h2>可复现合同</h2><p>Seed <code>{evaluation_seed}</code> · Actor <code>{actor_sha256}</code> · Opponent <code>{POLICY_SHA256}</code> · schedule <code>{summary['schedule_sha256']}</code> · result <code>{summary['game_results_sha256']}</code></p></section>
<script type="application/json" id="report-data">{embedded}</script></main><div class="lightbox" id="lightbox"><img alt="卡图大图预览"></div><script>const box=document.querySelector('#lightbox'),large=box.querySelector('img');for(const image of document.querySelectorAll('main img'))image.addEventListener('click',()=>{{large.src=image.src;large.alt=image.alt;box.classList.add('open')}});box.addEventListener('click',()=>box.classList.remove('open'));addEventListener('keydown',event=>{{if(event.key==='Escape')box.classList.remove('open')}});</script></body></html>"""


def _refresh(catalog: Any, candidates: tuple[Any, ...]) -> list[dict[str, Any]]:
    records = []
    for candidate in candidates:
        number = str(candidate.package_manifest["frozen_deck_number"])
        result_path = _result_path(number)
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        summary = _summary(catalog, candidate, result)
        game_records = build_cuda_game_records(
            json.loads(_schedule_path(number).read_text(encoding="utf-8")), result
        )
        comparison = _baseline_comparison(number, game_records)
        if comparison is not None:
            summary["baseline_comparison"] = comparison
        game_evidence_path = OUTPUT_ROOT / "games" / f"{number}.json"
        _atomic_json(game_evidence_path, _game_evidence(candidate, result))
        summary["game_records"] = f"games/{number}.json"
        summary["game_records_sha256"] = _sha256(game_evidence_path)
        report_name = str(candidate.package_manifest["frozen_report_href"])
        report_path = OUTPUT_ROOT / "reports" / report_name
        _atomic_text(report_path, _report_html(catalog, candidate, summary))
        summary["report"] = f"reports/{report_name}"
        summary["report_sha256"] = _sha256(report_path)
        records.append(summary)
    by_number = {record["deck_number"]: record for record in records}
    rows = []
    for candidate in candidates:
        number = str(candidate.package_manifest["frozen_deck_number"])
        record = by_number.get(number)
        schedule_entry = next(
            entry for entry in catalog.pool.schedule if entry.deck_id == candidate.name
        )
        images = "".join(
            f'<img src="{html.escape(str(card["image_url"]), quote=True)}" '
            f'alt="{html.escape(str(card["name"]), quote=True)}" loading="lazy">'
            for card in candidate.representative_cards
        )
        if record is None:
            result_cells = "<td>-</td><td>-</td><td>-</td><td>-</td><td>待评测</td>"
            title = f'<span>{html.escape(candidate.display_name or candidate.name)}</span>'
        else:
            first_rate = record["first"]["wins"] / record["first"]["games"]
            second_rate = record["second"]["wins"] / record["second"]["games"]
            result_cells = (
                f"<td>{record['wins']}-{record['losses']}-{record['draws']}</td>"
                f"<td class=\"strong\">{record['win_rate']:.2%}"
                + (
                    f"<small>vs 0806 {record['baseline_comparison']['delta_win_rate']:+.2%}</small>"
                    if "baseline_comparison" in record else ""
                )
                + "</td>"
                f"<td>{first_rate:.2%}<small>{record['first']['wins']}-{record['first']['losses']}-{record['first']['draws']}</small></td>"
                f"<td>{second_rate:.2%}<small>{record['second']['wins']}-{record['second']['losses']}-{record['second']['draws']}</small></td>"
                f"<td>{record['wall_seconds']:.1f}s<small>{record['games_per_second']:.2f} games/s</small></td>"
            )
            title = (
                f"<a href=\"{record['report']}\">"
                f"{html.escape(candidate.display_name or candidate.name)}</a>"
            )
        rows.append(
            f'<tr data-number="{int(number)}" data-win-rate="{record["win_rate"] if record else -1}" '
            f'data-wall="{record["wall_seconds"] if record else -1}"><td class="number">{number}</td><td><div class="deck"><span class="art">{images}</span>'
            f'<span>{title}<small>频率 {schedule_entry.games}/256 · 最佳名次 {schedule_entry.best_rank} · '
            f'{schedule_entry.observed_players} 人</small></span></div></td>{result_cells}</tr>'
        )
    total_games = sum(record["games"] for record in records)
    total_seconds = sum(record["wall_seconds"] for record in records)
    index = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(ACTOR_LABEL)} CUDA Seeded-2048 · Frozen-0806 卡组强度</title><style>
:root{{--bg:#f3f6f4;--paper:#fff;--ink:#17231f;--muted:#66766f;--line:#d9e3de;--green:#176b4d}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"PingFang SC",sans-serif}}header{{padding:28px max(20px,calc((100vw - 1500px)/2));background:#18382d;color:#fff}}h1{{margin:0;font-size:30px}}header p{{max-width:1050px;margin:7px 0 0;color:#cfe1da}}main{{max-width:1500px;margin:auto;padding:20px}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);background:#fff}}.stat{{padding:15px 18px;border-right:1px solid var(--line)}}.stat:last-child{{border:0}}.stat b{{display:block;font-size:23px}}.stat span,small{{display:block;color:var(--muted)}}.tools{{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0}}input{{width:min(420px,100%);padding:9px 11px;border:1px solid #b9c9c1;border-radius:4px}}button{{padding:9px 13px;border:1px solid #a9c2b7;border-radius:4px;background:#fff;color:#245c47;cursor:pointer}}button:hover{{background:#e9f3ee}}.table{{overflow:auto;border:1px solid var(--line);background:#fff}}table{{width:100%;min-width:1100px;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th{{position:sticky;top:0;background:#e9f0ec;color:#486158;font-size:12px}}th:nth-child(2),td:nth-child(2){{text-align:left}}tr:hover td{{background:#f8fbf9}}a{{color:var(--green);font-weight:700;text-decoration:none}}.deck{{display:flex;align-items:center;gap:10px}}.art{{display:flex;width:68px}}.art img{{width:38px;height:53px;margin-right:-8px;border:1px solid #c9d5cf;border-radius:3px;object-fit:cover;background:#e4ebe7;cursor:zoom-in}}.strong{{font-size:16px;font-weight:750}}.number{{font-size:16px;font-weight:850;color:var(--green)}}.contract{{margin-top:16px;padding:14px 16px;border-left:4px solid var(--green);background:#fff;color:var(--muted)}}.lightbox{{position:fixed;inset:0;z-index:10;display:none;place-items:center;padding:24px;background:#071c16dd}}.lightbox.open{{display:grid}}.lightbox img{{max-width:min(90vw,520px);max-height:88vh;border-radius:12px;box-shadow:0 24px 80px #0009}}@media(max-width:760px){{.stats{{grid-template-columns:1fr 1fr}}header{{padding:22px 16px}}main{{padding:14px}}}}</style></head><body>
<header><h1>{html.escape(ACTOR_LABEL)} · CUDA Seeded-2048 · Agent 选择先后手</h1><p>55 套 exact deck 均加载同一 {html.escape(ACTOR_LABEL)} checkpoint；对手固定为 Policy-0806。8 个独立 256 局 seed replica。只固定 seed，抛硬币赢家由 Agent 真实选择先后手，不做固定坐席。</p></header><main><div class="stats"><div class="stat"><b>{len(records)}/55</b><span>完成卡组</span></div><div class="stat"><b>{total_games:,}</b><span>CUDA 对局</span></div><div class="stat"><b>{sum(r['wins'] for r in records):,}</b><span>{html.escape(ACTOR_LABEL)} 胜局</span></div><div class="stat"><b>{total_seconds/60:.1f}m</b><span>累计 wall time</span></div></div>
<div class="tools"><input id="search" type="search" placeholder="筛选编号或牌型"><button data-sort="number">按编号</button><button data-sort="winRate">按胜率</button><button data-sort="wall">按耗时</button></div><div class="table"><table><thead><tr><th>编号</th><th>卡组 / 2048 局报告</th><th>W-L-D</th><th>胜率</th><th>实际先攻</th><th>实际后攻</th><th>耗时</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><div class="contract">8×256 independent seeds · seeded toss + winning Agent context-41 choice · strict FP32 · 256 resident CUDA lanes + masked refill · repeat-forfeit 20 · Actor <code>{ACTOR_SHA256}</code> · Opponent <code>{POLICY_SHA256}</code></div></main><div class="lightbox" id="lightbox"><img alt="卡图大图预览"></div><script>const q=document.querySelector('#search'),b=document.querySelector('tbody'),box=document.querySelector('#lightbox'),large=box.querySelector('img');q.addEventListener('input',()=>{{const s=q.value.toLowerCase();for(const r of b.rows)r.hidden=!r.innerText.toLowerCase().includes(s)}});for(const button of document.querySelectorAll('[data-sort]'))button.addEventListener('click',()=>{{const key=button.dataset.sort,rows=[...b.rows];rows.sort((a,c)=>key==='number'?+a.dataset.number-+c.dataset.number:+c.dataset[key]-+a.dataset[key]);b.append(...rows)}});for(const image of document.querySelectorAll('.art img'))image.addEventListener('click',()=>{{large.src=image.src;large.alt=image.alt;box.classList.add('open')}});box.addEventListener('click',()=>box.classList.remove('open'));addEventListener('keydown',event=>{{if(event.key==='Escape')box.classList.remove('open')}});</script></body></html>"""
    manifest = {
        "schema": "policy_0806_cuda_seeded2048_index_v3_agent_first_player",
        "contract_id": FROZEN_0806_CONTRACT_ID,
        "first_player_contract": FROZEN_0806_FIRST_PLAYER_CONTRACT,
        "evidence_boundary": "cuda_frozen_strength_contract_official_cpu_transfer_separate",
        "evaluation_seed": EVALUATION_SEED,
        "actor_label": ACTOR_LABEL,
        "actor_sha256": ACTOR_SHA256,
        "opponent_policy_sha256": POLICY_SHA256,
        "expected_decks": EXPECTED_DECKS,
        "expected_games": EXPECTED_DECKS * EXPECTED_GAMES,
        "published_decks": len(records),
        "published_games": total_games,
        "complete": len(records) == EXPECTED_DECKS,
        "lane_topology": "256_resident_cuda_masked_refill",
        "progress_guard_repeat_limit": 20,
        "reports": records,
    }
    _atomic_json(OUTPUT_ROOT / "manifest.json", manifest)
    _atomic_text(OUTPUT_ROOT / "index.html", index)
    _refresh_policy_root_index()
    return records


def _refresh_policy_root_index() -> None:
    policy_root = ROOT / "evaluation/arena/combat_mat/policy_0806"
    cards: list[str] = []
    for child in sorted(path for path in policy_root.iterdir() if path.is_dir()):
        manifest_path = child / "manifest.json"
        if not manifest_path.is_file():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("contract_id") != FROZEN_0806_CONTRACT_ID:
            continue
        label = payload.get("actor_label")
        if not isinstance(label, str):
            label = "Policy-0806" if child.name.startswith("0806_") else child.name
        completed = int(payload.get("published_decks", 0))
        games = int(payload.get("published_games", 0))
        cards.append(
            f'<a class="current" href="{html.escape(child.name, quote=True)}/index.html">'
            f'<b>{html.escape(label)} →</b><span>{completed}/55 套完成 · '
            f'{games:,}/112,640 局 · 对手 Policy-0806</span></a>'
        )
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Frozen-0806 Zero-Shot Comparison</title><style>
:root{{--bg:#f3f6f4;--ink:#17231f;--muted:#66766f;--green:#176b4d}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,"PingFang SC",sans-serif}}header{{padding:42px max(22px,calc((100vw - 1050px)/2));background:#18382d;color:#fff}}h1{{margin:0;font-size:34px}}header p{{margin:8px 0 0;color:#cfe1da}}main{{display:grid;gap:14px;max-width:1050px;margin:auto;padding:26px 20px}}.current{{display:block;padding:25px 28px;border:1px solid #9fc9b7;border-left:6px solid var(--green);border-radius:8px;background:#fff;color:inherit;text-decoration:none}}.current b{{display:block;color:var(--green);font-size:23px}}.current span{{color:var(--muted)}}</style></head><body><header><h1>Frozen-0806 · Zero-Shot Comparison</h1><p>统一合同：CUDA resident · Seeded-2048 · strict FP32 · 只固定 seed，抛硬币赢家 Agent 自主选择先后手。</p></header><main>{''.join(cards)}</main></body></html>\n'''
    _atomic_text(policy_root / "index.html", page)


def main(argv: list[str] | None = None) -> int:
    global EVALUATION_SEED

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--deck-number", action="append", default=[])
    parser.add_argument("--reproduce-first", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--temp-root", type=Path)
    parser.add_argument("--actor-checkpoint", type=Path, default=POLICY_MODEL)
    parser.add_argument("--actor-label", default="Policy-0806")
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=FROZEN_0806_EVALUATION_SEED,
        help="root seed; use the canonical default only for formal Frozen-0806",
    )
    args = parser.parse_args(argv)
    if args.evaluation_seed < 0:
        parser.error("--evaluation-seed cannot be negative")
    EVALUATION_SEED = int(args.evaluation_seed)
    configure_actor(checkpoint=args.actor_checkpoint, label=args.actor_label)
    configure_output_roots(output_root=args.output_root, temp_root=args.temp_root)
    catalog, candidates = _catalog()
    requested = (
        candidates
        if args.all
        else tuple(
            candidate
            for candidate in candidates
            if candidate.package_manifest["frozen_deck_number"]
            in set(args.deck_number)
        )
    )
    if not requested:
        parser.error("choose --all or at least one --deck-number NNN")
    _refresh(catalog, candidates)
    for candidate in requested:
        number = str(candidate.package_manifest["frozen_deck_number"])
        result = _run_one(catalog, candidate)
        if args.reproduce_first and number == "001":
            primary = _result_path(number)
            replay = TEMP_ROOT / "repro" / "001.json"
            replay.parent.mkdir(parents=True, exist_ok=True)
            replay.unlink(missing_ok=True)
            saved = _result_path(number)
            saved.replace(replay)
            try:
                rerun = _run_one(catalog, candidate)
                original = json.loads(replay.read_text(encoding="utf-8"))
                if (
                    original["determinism"]["game_results"]
                    != rerun["determinism"]["game_results"]
                    or original["progress_guard"]["forfeit_schedule_indices"]
                    != rerun["progress_guard"]["forfeit_schedule_indices"]
                ):
                    raise RuntimeError("deck 001 CUDA reproducibility check failed")
                _atomic_json(
                    TEMP_ROOT / "repro/001_diagnostic.json",
                    {
                        "game_results_exact": True,
                        "progress_guard_exact": True,
                        "raw_terminal_state_exact": (
                            original["determinism"]["terminal_state_sha256"]
                            == rerun["determinism"]["terminal_state_sha256"]
                        ),
                        "first_terminal_state_sha256": original["determinism"][
                            "terminal_state_sha256"
                        ],
                        "second_terminal_state_sha256": rerun["determinism"][
                            "terminal_state_sha256"
                        ],
                        "note": (
                            "Raw POD drift is retained as a diagnostic because TF32 near-ties "
                            "can change an internal greedy trajectory without changing outcomes."
                        ),
                    },
                )
            finally:
                replay.replace(saved)
            result = json.loads(saved.read_text(encoding="utf-8"))
        records = _refresh(catalog, candidates)
        summary = next(row for row in records if row["deck_number"] == number)
        print(
            f"CUDA_FROZEN_0806_COMPLETE actor={ACTOR_LABEL!r} deck={number} "
            f"record={summary['wins']}-{summary['losses']}-{summary['draws']} "
            f"wall={summary['wall_seconds']:.3f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
