"""Compare official-CPU and CUDA results on the shared Frozen-0806 panel.

This is deliberately *not* Gate G evidence.  The two inputs use the same
committed 8 x 256 schedule and are useful as a release diagnostic, but they do
not satisfy the independent-seed distribution-equivalence contract.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from evaluation.frozen_0806_contract import (
    FROZEN_0806_LEGACY_BALANCED_SEAT_CONTRACT_ID,
    FROZEN_0806_EVALUATION_GAMES,
    FROZEN_0806_EVALUATION_SEED,
    FROZEN_0806_EVALUATION_UNITS,
    FROZEN_0806_UNIT_GAMES,
    evaluation_game_seed,
)


EVIDENCE_TYPE = "shared_standard_panel_reference_not_independent_gate_g"
SCHEMA = "0038_standard_panel_cpu_cuda_reference_v1"
_REPORT_DATA = re.compile(
    r'<script\s+id=["\']report-data["\']\s+type=["\']application/json["\']>(.*?)</script>',
    re.DOTALL,
)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_Z95 = 1.959963984540054
_FROZEN_0806_BASE_SCHEDULE_SHA256 = (
    "16dbd18ce417405571c88997c9e97f9b2ec2adf96db544d1a9af988bb3c3cc3c"
)
_FROZEN_0806_SCHEDULE_ID = (
    "673fc18946281060e526fd49bee9a73b4240a5c72c96a2c73a1174ba2115a29d"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cpu_report(path: Path) -> dict[str, Any]:
    match = _REPORT_DATA.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"official CPU report lacks embedded report-data: {path}")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError("official CPU report-data must be a JSON object")
    return payload


def _load_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _outcome_cpu(game: Mapping[str, Any]) -> str:
    winner = game.get("winner")
    if winner == 0:
        return "win"
    if winner == 1:
        return "loss"
    if winner is None:
        return "draw"
    raise ValueError(f"invalid official CPU winner: {winner!r}")


def _outcome_cuda(game: Mapping[str, Any]) -> str:
    outcome = game.get("focal_outcome")
    if outcome not in {"win", "loss", "draw"}:
        raise ValueError(f"invalid CUDA focal_outcome: {outcome!r}")
    return str(outcome)


def _wilson(wins: int, games: int) -> list[float]:
    if games <= 0 or not 0 <= wins <= games:
        raise ValueError("invalid Wilson counts")
    probability = wins / games
    denominator = 1.0 + _Z95 * _Z95 / games
    centre = (probability + _Z95 * _Z95 / (2.0 * games)) / denominator
    half = (
        _Z95
        * math.sqrt(
            probability * (1.0 - probability) / games
            + _Z95 * _Z95 / (4.0 * games * games)
        )
        / denominator
    )
    return [centre - half, centre + half]


def _summary(outcomes: Iterable[str]) -> dict[str, Any]:
    counts = Counter(outcomes)
    games = sum(counts.values())
    wins = counts["win"]
    return {
        "games": games,
        "wins": wins,
        "losses": counts["loss"],
        "draws": counts["draw"],
        "win_rate": wins / games,
        "wilson_95": _wilson(wins, games),
    }


def _paired(cpu: Sequence[str], cuda: Sequence[str]) -> dict[str, Any]:
    if len(cpu) != len(cuda) or not cpu:
        raise ValueError("paired outcomes must be nonempty and equal length")
    score = {"win": 1.0, "draw": 0.5, "loss": 0.0}
    differences = [score[right] - score[left] for left, right in zip(cpu, cuda, strict=True)]
    estimate = sum(differences) / len(differences)
    variance = sum((value - estimate) ** 2 for value in differences) / (len(differences) - 1)
    half = _Z95 * math.sqrt(variance / len(differences))
    transitions = Counter(zip(cpu, cuda, strict=True))
    return {
        "games": len(cpu),
        "agreement": sum(left == right for left, right in zip(cpu, cuda, strict=True)),
        "cpu_loss_to_cuda_win": transitions[("loss", "win")],
        "cpu_win_to_cuda_loss": transitions[("win", "loss")],
        "cuda_minus_cpu_score": estimate,
        "paired_normal_95": [estimate - half, estimate + half],
        "score_definition": {"win": 1.0, "draw": 0.5, "loss": 0.0},
        "transition_matrix": {
            left: {right: transitions[(left, right)] for right in ("win", "draw", "loss")}
            for left in ("win", "draw", "loss")
        },
    }


def _ordered_cpu_blocks(games: Sequence[Mapping[str, Any]]) -> tuple[list[str], dict[str, int]]:
    order: list[str] = []
    counts: dict[str, int] = defaultdict(int)
    completed: set[str] = set()
    previous: str | None = None
    for game in games:
        opponent = str(game.get("opponent", ""))
        if not opponent:
            raise ValueError("official CPU game lacks opponent identity")
        if opponent != previous:
            if opponent in completed:
                raise ValueError("official CPU opponent blocks are not contiguous")
            if previous is not None:
                completed.add(previous)
            order.append(opponent)
            previous = opponent
        counts[opponent] += 1
    return order, dict(counts)


def _cpu_guard_adjudication(game: Mapping[str, Any]) -> bool:
    correctness = game.get("metric_refs", {}).get("correctness", {})
    outcome = game.get("metric_refs", {}).get("outcome", {})
    return (
        game.get("status") == "finished"
        and game.get("error_kind") is None
        and correctness.get("value") == "engine_error"
        and outcome.get("value") == "error"
    )


def _validate_and_pair(
    cpu: Mapping[str, Any],
    cuda: Mapping[str, Any],
    cuda_manifest: Mapping[str, Any],
    *,
    cuda_games_sha256: str,
) -> list[dict[str, Any]]:
    cpu_manifest = cpu.get("manifest")
    cpu_games = cpu.get("games")
    cuda_games = cuda.get("games")
    if not isinstance(cpu_manifest, Mapping) or not isinstance(cpu_games, list):
        raise ValueError("official CPU payload is incomplete")
    if not isinstance(cuda_games, list):
        raise ValueError("CUDA games payload is incomplete")
    expected = FROZEN_0806_EVALUATION_GAMES
    if len(cpu_games) != expected or len(cuda_games) != expected:
        raise ValueError("both standard-panel inputs must contain exactly 2,048 games")
    if (
        cpu_manifest.get("games") != expected
        or cpu_manifest.get("seed") != FROZEN_0806_EVALUATION_SEED
        or cuda.get("contract_id") != FROZEN_0806_LEGACY_BALANCED_SEAT_CONTRACT_ID
        or cuda.get("evaluation_seed") != FROZEN_0806_EVALUATION_SEED
    ):
        raise ValueError("Frozen-0806 evaluation contract or seed mismatch")
    if cpu_manifest.get("opponent_schedule_id") != _FROZEN_0806_SCHEDULE_ID:
        raise ValueError("official CPU opponent schedule identity is not Frozen-0806 v2")
    if not _HEX64.fullmatch(str(cuda.get("schedule_sha256", ""))):
        raise ValueError("CUDA schedule identity is invalid")

    candidate_manifest = cpu_manifest.get("candidate", {}).get("package_manifest", {})
    if (
        candidate_manifest.get("deck_id") != cuda.get("deck_id")
        or candidate_manifest.get("exact_deck_sha256") != cuda.get("exact_deck_sha256")
        or candidate_manifest.get("frozen_deck_number") != cuda.get("deck_number")
    ):
        raise ValueError("CPU/CUDA focal deck identity mismatch")
    policy_hash = candidate_manifest.get("checkpoint_sha256")
    if policy_hash != cuda_manifest.get("policy_sha256"):
        raise ValueError("CPU/CUDA policy checkpoint identity mismatch")

    reports = cuda_manifest.get("reports")
    matching_reports = (
        [row for row in reports if row.get("deck_id") == cuda.get("deck_id")]
        if isinstance(reports, list)
        else []
    )
    if len(matching_reports) != 1:
        raise ValueError("CUDA manifest must contain one matching deck report")
    cuda_report = matching_reports[0]
    if (
        cuda_report.get("games") != expected
        or cuda_report.get("game_records_sha256") != cuda_games_sha256
        or cuda_report.get("schedule_sha256") != cuda.get("schedule_sha256")
    ):
        raise ValueError("CUDA manifest does not commit the supplied games/schedule")

    order, counts = _ordered_cpu_blocks(cpu_games)
    expected_counts = list(cpu_manifest.get("games_per_opponent", []))
    if expected_counts != [counts[name] for name in order]:
        raise ValueError("official CPU opponent block counts mismatch manifest")
    if sum(counts.values()) != expected or any(
        count % FROZEN_0806_EVALUATION_UNITS for count in counts.values()
    ):
        raise ValueError("official CPU schedule is not eight complete replicas")
    base_counts = {name: count // FROZEN_0806_EVALUATION_UNITS for name, count in counts.items()}
    if sum(base_counts.values()) != FROZEN_0806_UNIT_GAMES:
        raise ValueError("official CPU base schedule is not 256 slots")
    offsets: dict[str, int] = {}
    cursor = 0
    for name in order:
        offsets[name] = cursor
        cursor += base_counts[name]

    cuda_by_index: dict[int, Mapping[str, Any]] = {}
    for game in cuda_games:
        index = game.get("schedule_index")
        if type(index) is not int or index in cuda_by_index:
            raise ValueError("CUDA schedule indices must be unique integers")
        cuda_by_index[index] = game
    if set(cuda_by_index) != set(range(expected)):
        raise ValueError("CUDA schedule indices must cover [0, 2048)")
    schedule_payload = {
        "schema": "policy_0806_cuda_seeded2048_v2",
        "evaluation_seed": FROZEN_0806_EVALUATION_SEED,
        "focal_deck_id": cuda.get("deck_id"),
        "jobs": [
            {
                key: cuda_by_index[index][key]
                for key in (
                    "game_id",
                    "opponent_id",
                    "replica",
                    "slot",
                    "engine_seed",
                    "search_seed",
                    "focal_first",
                )
            }
            for index in range(expected)
        ],
    }
    reconstructed_schedule_sha256 = hashlib.sha256(
        json.dumps(schedule_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if cuda.get("schedule_sha256") != reconstructed_schedule_sha256:
        raise ValueError("CUDA schedule SHA-256 does not match the supplied jobs")
    raw_results = [cuda_by_index[index].get("raw_engine_result") for index in range(expected)]
    if any(type(value) is not int for value in raw_results):
        raise ValueError("CUDA records lack integer raw_engine_result values")
    reconstructed_results_sha256 = hashlib.sha256(
        json.dumps(raw_results, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    if cuda.get("game_results_sha256") != reconstructed_results_sha256:
        raise ValueError("CUDA result SHA-256 does not match per-game engine results")

    rows: list[dict[str, Any]] = []
    local_number: dict[str, int] = defaultdict(int)
    for cpu_game in cpu_games:
        opponent = str(cpu_game["opponent"])
        local_number[opponent] += 1
        number = local_number[opponent]
        base_slots = base_counts[opponent]
        replica = (number - 1) // base_slots
        slot = (number - 1) % base_slots
        schedule_index = replica * FROZEN_0806_UNIT_GAMES + offsets[opponent] + slot
        cuda_game = cuda_by_index[schedule_index]
        focal_first = replica % 2 == 0
        if (
            cpu_game.get("game_id") != f"{opponent}-{number:03d}"
            or cpu_game.get("candidate_first") is not focal_first
            or cpu_game.get("status") != "finished"
            or cuda_game.get("opponent_id") != opponent
            or cuda_game.get("replica") != replica
            or cuda_game.get("slot") != slot
            or cuda_game.get("focal_first") is not focal_first
            or cuda_game.get("game_id") != f"r{replica + 1:02d}-{opponent}-{slot + 1:03d}"
        ):
            raise ValueError(f"CPU/CUDA mapping or seat mismatch at schedule_index={schedule_index}")
        raw_engine_result = cuda_game.get("raw_engine_result")
        expected_cuda_outcome = (
            "draw"
            if raw_engine_result == 0
            else "win"
            if (raw_engine_result == 1) is focal_first
            else "loss"
            if raw_engine_result in {1, 2}
            else None
        )
        if cuda_game.get("focal_outcome") != expected_cuda_outcome:
            raise ValueError(f"CUDA raw/focal outcome mismatch at schedule_index={schedule_index}")
        focal = str(cuda.get("deck_id"))
        expected_engine_seed = evaluation_game_seed(
            focal_identity=focal,
            opponent_identity=opponent,
            slot=slot,
            replica=replica,
        )
        expected_search_seed = evaluation_game_seed(
            focal_identity=focal,
            opponent_identity=opponent,
            slot=slot,
            replica=replica,
            namespace="search",
        )
        if (
            cuda_game.get("engine_seed") != expected_engine_seed
            or cuda_game.get("search_seed") != expected_search_seed
        ):
            raise ValueError(f"CUDA seed contract mismatch at schedule_index={schedule_index}")
        rows.append(
            {
                "schedule_index": schedule_index,
                "replica": replica,
                "opponent": opponent,
                "seat": "first" if focal_first else "second",
                "cpu": cpu_game,
                "cuda": cuda_game,
                "cpu_outcome": _outcome_cpu(cpu_game),
                "cuda_outcome": _outcome_cuda(cuda_game),
            }
        )
    rows.sort(key=lambda row: row["schedule_index"])
    return rows


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "cpu": _summary(row["cpu_outcome"] for row in rows),
        "cuda": _summary(row["cuda_outcome"] for row in rows),
        "paired": _paired(
            [str(row["cpu_outcome"]) for row in rows],
            [str(row["cuda_outcome"]) for row in rows],
        ),
    }


def analyze_standard_panel_reference(
    cpu_report_html: Path,
    cuda_games_json: Path,
    *,
    cuda_manifest_json: Path | None = None,
) -> dict[str, Any]:
    """Return a fail-closed shared-panel reference comparison."""

    cpu_report_html = cpu_report_html.resolve()
    cuda_games_json = cuda_games_json.resolve()
    if cuda_manifest_json is None:
        cuda_manifest_json = cuda_games_json.parent.parent / "manifest.json"
    cuda_manifest_json = cuda_manifest_json.resolve()
    cpu = _load_cpu_report(cpu_report_html)
    cuda = _load_object(cuda_games_json, "CUDA games")
    cuda_manifest = _load_object(cuda_manifest_json, "CUDA manifest")
    cuda_games_sha256 = _sha256(cuda_games_json)
    rows = _validate_and_pair(
        cpu, cuda, cuda_manifest, cuda_games_sha256=cuda_games_sha256
    )

    cpu_games = [row["cpu"] for row in rows]
    cuda_games = [row["cuda"] for row in rows]
    cpu_summary = cpu["summary"]
    direct_cpu = _summary(row["cpu_outcome"] for row in rows)
    if any(
        int(cpu_summary.get(key, -1)) != direct_cpu[key]
        for key in ("wins", "losses", "draws")
    ):
        raise ValueError("official CPU summary disagrees with per-game winners")

    cuda_report = next(
        row for row in cuda_manifest["reports"] if row["deck_id"] == cuda["deck_id"]
    )
    direct_cuda = _summary(row["cuda_outcome"] for row in rows)
    if any(
        int(cuda_report.get(key, -1)) != direct_cuda[key]
        for key in ("wins", "losses", "draws")
    ):
        raise ValueError("CUDA manifest disagrees with per-game outcomes")

    cpu_rounds: list[int] = []
    cpu_selects: list[int] = []
    for game in cpu_games:
        length = game.get("metric_refs", {}).get("length", {}).get("payload", {})
        round_value = length.get("round")
        selection_value = length.get("action_selections")
        if type(round_value) is not int or type(selection_value) is not int:
            raise ValueError("official CPU report lacks per-game length/select diagnostics")
        cpu_rounds.append(round_value)
        cpu_selects.append(selection_value)
    cpu_lifecycle_errors = sum(
        game.get("status") != "finished" or game.get("error_kind") is not None
        for game in cpu_games
    )
    cpu_guard_indices = [
        int(row["schedule_index"])
        for row in rows
        if _cpu_guard_adjudication(row["cpu"])
    ]
    cpu_guards = len(cpu_guard_indices)
    metric_error_count = int(cpu.get("metrics", {}).get("correctness", {}).get("numerator", 0))
    if metric_error_count != cpu_guards:
        raise ValueError(
            "official CPU finished-adjudication count disagrees with correctness metric"
        )
    cuda_errors = sum(bool(game.get("error", False)) for game in cuda_games)
    cuda_guard_indices = sorted(
        int(game["schedule_index"])
        for game in cuda_games
        if bool(game.get("repeat_forfeit", False))
    )
    cuda_guards = len(cuda_guard_indices)
    cuda_turn_limit = sum(bool(game.get("turn_limit_draw", False)) for game in cuda_games)
    if cuda_guards != len(cuda_report.get("progress_guard_forfeits", [])):
        raise ValueError("CUDA per-game/manifest repeat-forfeit count mismatch")

    by_seat = {
        seat: _group_summary([row for row in rows if row["seat"] == seat])
        for seat in ("first", "second")
    }
    by_shard = [
        {"replica": replica, **_group_summary([row for row in rows if row["replica"] == replica])}
        for replica in range(FROZEN_0806_EVALUATION_UNITS)
    ]
    opponents = []
    for opponent in dict.fromkeys(str(row["opponent"]) for row in rows):
        opponents.append(
            {"opponent": opponent, **_group_summary([row for row in rows if row["opponent"] == opponent])}
        )

    cpu_performance = cpu_summary.get("performance", {})
    cpu_wall = float(cpu_performance.get("wall_time_seconds", 0.0))
    cuda_wall = float(cuda_report.get("wall_seconds", 0.0))
    if cpu_wall <= 0.0 or cuda_wall <= 0.0:
        raise ValueError("both inputs must commit positive wall-clock throughput")
    return {
        "schema": SCHEMA,
        "evidence_type": EVIDENCE_TYPE,
        "gate_g_eligible": False,
        "inputs": {
            "cpu_report_html": {"path": str(cpu_report_html), "sha256": _sha256(cpu_report_html)},
            "cuda_games_json": {"path": str(cuda_games_json), "sha256": cuda_games_sha256},
            "cuda_manifest_json": {"path": str(cuda_manifest_json), "sha256": _sha256(cuda_manifest_json)},
        },
        "contract": {
            "contract_id": FROZEN_0806_LEGACY_BALANCED_SEAT_CONTRACT_ID,
            "evaluation_seed": FROZEN_0806_EVALUATION_SEED,
            "games": FROZEN_0806_EVALUATION_GAMES,
            "shards": FROZEN_0806_EVALUATION_UNITS,
            "games_per_shard": FROZEN_0806_UNIT_GAMES,
            "cpu_schedule_id": cpu["manifest"]["opponent_schedule_id"],
            "base_schedule_sha256": _FROZEN_0806_BASE_SCHEDULE_SHA256,
            "cuda_schedule_sha256": cuda["schedule_sha256"],
            "policy_sha256": cuda_manifest["policy_sha256"],
            "exact_deck_sha256": cuda["exact_deck_sha256"],
            "mapping": "opponent + CPU consecutive number -> (replica, local slot, CUDA schedule_index)",
        },
        "overall": _group_summary(rows),
        "by_seat": by_seat,
        "by_shard": by_shard,
        "by_opponent": opponents,
        "diagnostics": {
            "cpu": {
                "mean_complete_rounds": sum(cpu_rounds) / len(cpu_rounds),
                "mean_official_selects": sum(cpu_selects) / len(cpu_selects),
                "official_selects": sum(cpu_selects),
                "lifecycle_errors": cpu_lifecycle_errors,
                "progress_guard_adjudications": cpu_guards,
                "progress_guard_schedule_indices": cpu_guard_indices,
                "guard_derivation": "finished + null error_kind + correctness=engine_error + outcome=error",
            },
            "cuda": {
                "errors": cuda_errors,
                "repeat_forfeits": cuda_guards,
                "repeat_forfeit_schedule_indices": cuda_guard_indices,
                "turn_limit_draws": cuda_turn_limit,
            },
            "guard_mapping": {
                "intersection": sorted(set(cpu_guard_indices) & set(cuda_guard_indices)),
                "cpu_only": sorted(set(cpu_guard_indices) - set(cuda_guard_indices)),
                "cuda_only": sorted(set(cuda_guard_indices) - set(cpu_guard_indices)),
            },
        },
        "throughput": {
            "cpu": {
                "wall_seconds": cpu_wall,
                "games_per_second": FROZEN_0806_EVALUATION_GAMES / cpu_wall,
                "official_selects_per_second": sum(cpu_selects) / cpu_wall,
            },
            "cuda": {
                "wall_seconds": cuda_wall,
                "games_per_second": FROZEN_0806_EVALUATION_GAMES / cuda_wall,
            },
            "cuda_over_cpu_games_per_second": cpu_wall / cuda_wall,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-report", type=Path, required=True)
    parser.add_argument("--cuda-games", type=Path, required=True)
    parser.add_argument("--cuda-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze_standard_panel_reference(
        args.cpu_report,
        args.cuda_games,
        cuda_manifest_json=args.cuda_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "evidence_type": result["evidence_type"],
        "cpu_win_rate": result["overall"]["cpu"]["win_rate"],
        "cuda_win_rate": result["overall"]["cuda"]["win_rate"],
        "speedup": result["throughput"]["cuda_over_cpu_games_per_second"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
