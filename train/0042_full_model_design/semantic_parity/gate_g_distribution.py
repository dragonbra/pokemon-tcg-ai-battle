"""Independent-seed end-to-end distribution equivalence gate."""

from __future__ import annotations

from dataclasses import dataclass
import math
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from evaluation.frozen_0806_contract import (
    FROZEN_0806_EVALUATION_GAMES,
    FROZEN_0806_EVALUATION_UNITS,
    evaluation_game_seed,
)

from .statistical_equivalence import (
    binary_difference_interval,
    multinomial_total_variation,
)


@dataclass(frozen=True, slots=True)
class MetricVerdict:
    status: str
    estimate: float | None
    lower_95: float | None
    upper_95: float | None
    margin: float
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class DistributionEquivalenceContract:
    """Pre-registered margins and evidence requirements for Gate G.

    These are equivalence margins, not null-hypothesis significance thresholds.
    An interval that merely overlaps the margin remains ``INCOMPLETE``.
    """

    win_rate_margin: float = 0.03
    game_length_margin: float = 0.5
    prize_differential_margin: float = 0.25
    decision_count_absolute_margin: float = 0.03
    decision_count_relative_margin: float = 0.10
    action_type_tv_margin: float = 0.03
    minimum_games_per_stratum: int = 200
    require_identical_opponent_seat_mix: bool = True
    require_disjoint_engine_seed_sets: bool = True
    require_zero_runtime_errors: bool = True


DEFAULT_CONTRACT = DistributionEquivalenceContract()


def _binary(
    cpu: Sequence[Mapping[str, Any]],
    cuda: Sequence[Mapping[str, Any]],
    *,
    margin: float,
    minimum_games: int,
) -> MetricVerdict:
    if not cpu or not cuda:
        return MetricVerdict("INCOMPLETE", None, None, None, margin, "missing samples")
    result = binary_difference_interval(
        sum(row.get("outcome") == "win" for row in cpu), len(cpu),
        sum(row.get("outcome") == "win" for row in cuda), len(cuda), margin=margin,
    )
    if min(len(cpu), len(cuda)) < minimum_games:
        status = "INCOMPLETE"
        reason = (
            f"pre-registered minimum {minimum_games} games per backend stratum not met"
        )
    elif result.passed:
        status = "PASS"
        reason = None
    elif result.lower > margin or result.upper < -margin:
        status = "FAIL"
        reason = None
    else:
        status = "INCOMPLETE"
        reason = "confidence interval is not contained by equivalence margin"
    return MetricVerdict(status, result.estimate, result.lower, result.upper, margin, reason)


def _mean(
    cpu: Sequence[Mapping[str, Any]],
    cuda: Sequence[Mapping[str, Any]],
    field: str,
    *,
    absolute_margin: float,
    relative_margin: float | None = None,
    minimum_games: int,
) -> MetricVerdict:
    try:
        left = [float(row[field]) for row in cpu]
        right = [float(row[field]) for row in cuda]
    except (KeyError, TypeError, ValueError):
        return MetricVerdict(
            "INCOMPLETE", None, None, None, absolute_margin,
            f"missing per-game {field}",
        )
    reference_mean = (sum(left) / len(left) + sum(right) / len(right)) / 2
    margin = max(
        absolute_margin,
        abs(reference_mean) * relative_margin if relative_margin is not None else 0.0,
    )
    if min(len(left), len(right)) < minimum_games:
        return MetricVerdict(
            "INCOMPLETE", None, None, None, margin,
            f"pre-registered minimum {minimum_games} games per backend stratum not met",
        )
    estimate = sum(left) / len(left) - sum(right) / len(right)
    var_left = sum((value - sum(left) / len(left)) ** 2 for value in left) / (len(left) - 1)
    var_right = sum((value - sum(right) / len(right)) ** 2 for value in right) / (len(right) - 1)
    half = 1.959963984540054 * math.sqrt(var_left / len(left) + var_right / len(right))
    lower, upper = estimate - half, estimate + half
    status = "PASS" if lower >= -margin and upper <= margin else (
        "FAIL" if lower > margin or upper < -margin else "INCOMPLETE"
    )
    return MetricVerdict(status, estimate, lower, upper, margin,
                         None if status != "INCOMPLETE" else "confidence interval is not contained by equivalence margin")


def _action_type_distribution(
    cpu: Sequence[Mapping[str, Any]],
    cuda: Sequence[Mapping[str, Any]],
    *,
    margin: float,
    minimum_games: int,
) -> dict[str, Any]:
    if min(len(cpu), len(cuda)) < minimum_games:
        return {
            "status": "INCOMPLETE",
            "reason": (
                f"pre-registered minimum {minimum_games} games per backend not met"
            ),
            "margin": margin,
        }
    left: dict[str, int] = defaultdict(int)
    right: dict[str, int] = defaultdict(int)
    for backend, rows, counts in (("cpu", cpu, left), ("cuda", cuda, right)):
        for index, row in enumerate(rows):
            raw = row.get("action_type_counts")
            if not isinstance(raw, Mapping):
                return {
                    "status": "INCOMPLETE",
                    "reason": f"{backend}[{index}] missing action_type_counts",
                    "margin": margin,
                }
            for action_type, count in raw.items():
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError(
                        f"{backend}[{index}] invalid action count {action_type!r}={count!r}"
                    )
                counts[str(action_type)] += count
    if not sum(left.values()) or not sum(right.values()):
        return {
            "status": "INCOMPLETE",
            "reason": "action type totals are empty",
            "margin": margin,
        }
    result = multinomial_total_variation(left, right, margin=margin)
    status = "PASS" if result.passed else (
        "FAIL" if result.estimate > result.margin else "INCOMPLETE"
    )
    return {
        "status": status,
        "estimate_tv": result.estimate,
        "upper_95": result.upper_95,
        "margin": result.margin,
        "cpu_counts": dict(sorted(left.items())),
        "cuda_counts": dict(sorted(right.items())),
        "reason": (
            None if status != "INCOMPLETE"
            else "TV confidence bound is not contained by equivalence margin"
        ),
    }


def _normalized_strata(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Fraction]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        counts[(str(row["opponent"]), str(row["seat"]))] += 1
    return {key: Fraction(value, len(rows)) for key, value in counts.items()}


def _input_contract(
    cpu: Sequence[Mapping[str, Any]],
    cuda: Sequence[Mapping[str, Any]],
    contract: DistributionEquivalenceContract,
) -> dict[str, Any]:
    cpu_strata = _normalized_strata(cpu)
    cuda_strata = _normalized_strata(cuda)
    strata_equal = cpu_strata == cuda_strata
    strata = {
        "status": (
            "PASS"
            if strata_equal or not contract.require_identical_opponent_seat_mix
            else "INCOMPLETE"
        ),
        "require_identical_normalized_mix": contract.require_identical_opponent_seat_mix,
        "reason": None if strata_equal else "opponent/seat mixture differs between backends",
        "cpu_counts": {
            f"{opponent}|{seat}": sum(
                row["opponent"] == opponent and row["seat"] == seat for row in cpu
            )
            for opponent, seat in sorted(cpu_strata)
        },
        "cuda_counts": {
            f"{opponent}|{seat}": sum(
                row["opponent"] == opponent and row["seat"] == seat for row in cuda
            )
            for opponent, seat in sorted(cuda_strata)
        },
    }

    missing_cpu = [index for index, row in enumerate(cpu) if "engine_seed" not in row]
    missing_cuda = [index for index, row in enumerate(cuda) if "engine_seed" not in row]
    if missing_cpu or missing_cuda:
        seeds = {
            "status": "INCOMPLETE",
            "require_disjoint_sets": contract.require_disjoint_engine_seed_sets,
            "reason": (
                f"missing engine_seed provenance: cpu={len(missing_cpu)}, "
                f"cuda={len(missing_cuda)}"
            ),
        }
    else:
        cpu_seeds = [int(row["engine_seed"]) for row in cpu]
        cuda_seeds = [int(row["engine_seed"]) for row in cuda]
        duplicate_cpu = len(cpu_seeds) - len(set(cpu_seeds))
        duplicate_cuda = len(cuda_seeds) - len(set(cuda_seeds))
        overlap = set(cpu_seeds) & set(cuda_seeds)
        passed = (
            duplicate_cpu == duplicate_cuda == 0
            and (not overlap or not contract.require_disjoint_engine_seed_sets)
        )
        seeds = {
            "status": "PASS" if passed else "INCOMPLETE",
            "require_disjoint_sets": contract.require_disjoint_engine_seed_sets,
            "cpu_unique": len(set(cpu_seeds)),
            "cuda_unique": len(set(cuda_seeds)),
            "cpu_duplicates": duplicate_cpu,
            "cuda_duplicates": duplicate_cuda,
            "overlap": len(overlap),
            "reason": None if passed else "seed sets are duplicated or not independent/disjoint",
        }
    return {"strata": strata, "engine_seeds": seeds}


def _zero_runtime_event(
    cpu: Sequence[Mapping[str, Any]],
    cuda: Sequence[Mapping[str, Any]],
    field: str,
    *,
    require_zero: bool,
) -> dict[str, Any]:
    missing_cpu = sum(field not in row for row in cpu)
    missing_cuda = sum(field not in row for row in cuda)
    if missing_cpu or missing_cuda:
        return {
            "cpu": None,
            "cuda": None,
            "status": "INCOMPLETE",
            "reason": f"missing {field}: cpu={missing_cpu}, cuda={missing_cuda}",
        }
    cpu_count = sum(bool(row[field]) for row in cpu)
    cuda_count = sum(bool(row[field]) for row in cuda)
    passed = cpu_count == cuda_count == 0
    if not require_zero:
        # Non-zero rate equivalence is intentionally not used by the current release
        # contract. Preserve the counts but require a future explicit interval contract.
        return {
            "cpu": cpu_count,
            "cuda": cuda_count,
            "status": "INCOMPLETE",
            "reason": "non-zero runtime-event equivalence margin is not configured",
        }
    return {
        "cpu": cpu_count,
        "cuda": cuda_count,
        "status": "PASS" if passed else "FAIL",
        "reason": None if passed else "release contract requires zero events on both backends",
    }


def compare_runtime_distributions(
    cpu_games: Sequence[Mapping[str, Any]],
    cuda_games: Sequence[Mapping[str, Any]],
    *,
    contract: DistributionEquivalenceContract = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    if not cpu_games or not cuda_games:
        raise ValueError("both backends require game records")
    required_identity = ("outcome", "seat", "opponent")
    for backend, rows in (("cpu", cpu_games), ("cuda", cuda_games)):
        for index, row in enumerate(rows):
            missing = [field for field in required_identity if field not in row]
            if missing:
                raise ValueError(f"{backend}[{index}] missing {missing}")
            if row["outcome"] not in {"win", "loss", "draw", "error", "unfinished"}:
                raise ValueError(f"{backend}[{index}] invalid outcome {row['outcome']!r}")
            if row["seat"] not in {"first", "second"}:
                raise ValueError(f"{backend}[{index}] invalid seat {row['seat']!r}")
    input_contract = _input_contract(cpu_games, cuda_games, contract)
    metrics: dict[str, MetricVerdict] = {
        "overall_win_rate": _binary(
            cpu_games, cuda_games, margin=contract.win_rate_margin,
            minimum_games=contract.minimum_games_per_stratum,
        ),
        "first_seat_win_rate": _binary(
            [row for row in cpu_games if row["seat"] == "first"],
            [row for row in cuda_games if row["seat"] == "first"],
            margin=contract.win_rate_margin,
            minimum_games=contract.minimum_games_per_stratum,
        ),
        "second_seat_win_rate": _binary(
            [row for row in cpu_games if row["seat"] == "second"],
            [row for row in cuda_games if row["seat"] == "second"],
            margin=contract.win_rate_margin,
            minimum_games=contract.minimum_games_per_stratum,
        ),
        "game_length": _mean(
            cpu_games, cuda_games, "game_length",
            absolute_margin=contract.game_length_margin,
            minimum_games=contract.minimum_games_per_stratum,
        ),
        "prize_differential": _mean(
            cpu_games, cuda_games, "prize_differential",
            absolute_margin=contract.prize_differential_margin,
            minimum_games=contract.minimum_games_per_stratum,
        ),
        "strategic_decisions": _mean(
            cpu_games, cuda_games, "strategic_decisions",
            absolute_margin=contract.decision_count_absolute_margin,
            relative_margin=contract.decision_count_relative_margin,
            minimum_games=contract.minimum_games_per_stratum,
        ),
        "macro_count": _mean(
            cpu_games, cuda_games, "macro_count",
            absolute_margin=contract.decision_count_absolute_margin,
            relative_margin=contract.decision_count_relative_margin,
            minimum_games=contract.minimum_games_per_stratum,
        ),
        "forced_count": _mean(
            cpu_games, cuda_games, "forced_count",
            absolute_margin=contract.decision_count_absolute_margin,
            relative_margin=contract.decision_count_relative_margin,
            minimum_games=contract.minimum_games_per_stratum,
        ),
    }
    matchups: dict[str, MetricVerdict] = {}
    opponents = sorted({str(row["opponent"]) for row in cpu_games} | {str(row["opponent"]) for row in cuda_games})
    for opponent in opponents:
        matchups[opponent] = _binary(
            [row for row in cpu_games if row["opponent"] == opponent],
            [row for row in cuda_games if row["opponent"] == opponent],
            margin=contract.win_rate_margin,
            minimum_games=contract.minimum_games_per_stratum,
        )
    error_fields = ("error", "continuation_error", "fallback")
    error_rates = {}
    for field in error_fields:
        error_rates[field] = _zero_runtime_event(
            cpu_games, cuda_games, field,
            require_zero=contract.require_zero_runtime_errors,
        )
    action_type_distribution = _action_type_distribution(
        cpu_games, cuda_games,
        margin=contract.action_type_tv_margin,
        minimum_games=contract.minimum_games_per_stratum,
    )
    statuses = (
        [row["status"] for row in input_contract.values()]
        +
        [verdict.status for verdict in metrics.values()]
        + [verdict.status for verdict in matchups.values()]
        + [row["status"] for row in error_rates.values()]
        + [str(action_type_distribution["status"])]
    )
    status = "FAIL" if "FAIL" in statuses else "INCOMPLETE" if "INCOMPLETE" in statuses else "PASS"
    serialize = lambda value: {"status": value.status, "estimate": value.estimate,
                               "lower_95": value.lower_95, "upper_95": value.upper_95,
                               "margin": value.margin, "reason": value.reason}
    return {
        "gate": "G", "status": status, "cpu_games": len(cpu_games), "cuda_games": len(cuda_games),
        "seed_contract": "disjoint independent backend engine-seed sets; identical normalized opponent/seat strata",
        "equivalence_contract": {
            field: getattr(contract, field)
            for field in contract.__dataclass_fields__
        },
        "input_contract": input_contract,
        "metrics": {name: serialize(value) for name, value in metrics.items()},
        "matchups": {name: serialize(value) for name, value in matchups.items()},
        "error_rates": error_rates,
        "action_type_distribution": action_type_distribution,
    }


def load_official_cpu_report(path: Path) -> dict[str, Any]:
    """Load the immutable JSON payload embedded by the official evaluator."""

    document = path.read_text(encoding="utf-8")
    marker = '<script id="report-data" type="application/json">'
    try:
        encoded = document.split(marker, 1)[1].split("</script>", 1)[0]
    except IndexError as exc:
        raise ValueError(f"report-data payload is missing: {path}") from exc
    payload = json.loads(encoded)
    if not isinstance(payload, dict):
        raise ValueError(f"report-data payload must be an object: {path}")
    return payload


_OPTIONAL_DISTRIBUTION_FIELDS = (
    "engine_seed",
    "prize_differential",
    "strategic_decisions",
    "macro_count",
    "forced_count",
    "action_type_counts",
    "continuation_error",
    "fallback",
)


def _official_frozen_2048_engine_seeds(
    payload: Mapping[str, Any],
) -> dict[str, int] | None:
    """Reconstruct seeds only after validating the formal eight-replica layout.

    Historical official reports did not persist ``GameRequest.seed`` per game.
    A Frozen-0809 2,048 report is nevertheless deterministic from its manifest:
    each opponent count is eight times its base-slot count, replicas 0/2/4/6 put
    the candidate first, and :func:`evaluation_game_seed` defines the seed.  Any
    ambiguity makes this helper return ``None`` rather than invent provenance.
    """

    manifest = payload.get("manifest")
    raw_games = payload.get("games")
    if not isinstance(manifest, Mapping):
        return None
    if not isinstance(raw_games, Sequence) or isinstance(raw_games, (str, bytes)):
        return None
    if manifest.get("games") != FROZEN_0806_EVALUATION_GAMES:
        return None
    if len(raw_games) != FROZEN_0806_EVALUATION_GAMES:
        return None
    evaluation_seed = manifest.get("seed")
    candidate = manifest.get("candidate")
    opponents = manifest.get("opponents")
    counts = manifest.get("games_per_opponent")
    if isinstance(evaluation_seed, bool) or not isinstance(evaluation_seed, int):
        return None
    if not isinstance(candidate, Mapping) or not isinstance(candidate.get("name"), str):
        return None
    if not manifest.get("opponent_schedule_id"):
        return None
    if not isinstance(opponents, Sequence) or isinstance(opponents, (str, bytes)):
        return None
    if not isinstance(counts, Sequence) or isinstance(counts, (str, bytes)):
        return None
    if len(opponents) != len(counts):
        return None

    games_by_id: dict[str, Mapping[str, Any]] = {}
    for game in raw_games:
        if not isinstance(game, Mapping) or not isinstance(game.get("game_id"), str):
            return None
        game_id = str(game["game_id"])
        if game_id in games_by_id:
            return None
        games_by_id[game_id] = game

    seeds: dict[str, int] = {}
    expected_total = 0
    for opponent, raw_count in zip(opponents, counts, strict=True):
        if not isinstance(opponent, Mapping) or not isinstance(opponent.get("name"), str):
            return None
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 1:
            return None
        if raw_count % FROZEN_0806_EVALUATION_UNITS:
            return None
        opponent_name = str(opponent["name"])
        base_slots = raw_count // FROZEN_0806_EVALUATION_UNITS
        for local_index in range(raw_count):
            game_number = local_index + 1
            game_id = f"{opponent_name}-{game_number:03d}"
            game = games_by_id.get(game_id)
            if game is None or game.get("opponent") != opponent_name:
                return None
            replica = local_index // base_slots
            slot = local_index % base_slots
            expected_first = replica % 2 == 0
            if game.get("candidate_first") is not expected_first:
                return None
            seeds[game_id] = evaluation_game_seed(
                evaluation_seed=evaluation_seed,
                focal_identity=str(candidate["name"]),
                opponent_identity=opponent_name,
                slot=slot,
                replica=replica,
            )
        expected_total += raw_count
    if expected_total != FROZEN_0806_EVALUATION_GAMES or len(seeds) != len(games_by_id):
        return None
    return seeds


def normalize_official_cpu_report(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize only fields actually persisted by an official CPU report.

    In particular, an absent fallback/continuation counter is not rewritten as
    zero. Gate G must remain incomplete until that evidence is recorded.
    """

    raw_games = payload.get("games")
    if not isinstance(raw_games, Sequence) or isinstance(raw_games, (str, bytes)):
        raise ValueError("official report games must be a sequence")
    reconstructed_seeds = _official_frozen_2048_engine_seeds(payload)
    result: list[dict[str, Any]] = []
    for index, game in enumerate(raw_games):
        if not isinstance(game, Mapping):
            raise ValueError(f"official games[{index}] must be a mapping")
        metric_refs = game.get("metric_refs")
        if not isinstance(metric_refs, Mapping):
            raise ValueError(f"official games[{index}] missing metric_refs")
        outcome_metric = metric_refs.get("outcome")
        if not isinstance(outcome_metric, Mapping):
            raise ValueError(f"official games[{index}] missing outcome metric")
        status = str(game.get("status"))
        winner = game.get("winner")
        if status == "finished":
            # The compact report stores candidate-relative winners. This remains
            # authoritative when an auxiliary metric plugin reports a spurious
            # per-game ``error`` despite the engine game having finished.
            outcome = "win" if winner == 0 else "loss" if winner == 1 else "draw"
        else:
            outcome = str(outcome_metric.get("value"))
        row: dict[str, Any] = {
            "game_id": str(game["game_id"]),
            "opponent": str(game["opponent"]),
            "seat": "first" if game.get("candidate_first") is True else "second",
            "outcome": outcome,
            "error": status != "finished",
        }
        if reconstructed_seeds is not None:
            row["engine_seed"] = reconstructed_seeds[str(game["game_id"])]
        length_metric = metric_refs.get("length")
        if isinstance(length_metric, Mapping) and isinstance(length_metric.get("value"), (int, float)):
            row["game_length"] = float(length_metric["value"])
        for field in _OPTIONAL_DISTRIBUTION_FIELDS:
            if field in game:
                row[field] = game[field]
        result.append(row)
    return result


def normalize_cuda_game_results(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize persisted CUDA game records without inventing diagnostics."""

    raw_games = payload.get("games")
    if not isinstance(raw_games, Sequence) or isinstance(raw_games, (str, bytes)):
        raise ValueError("CUDA game results must be a sequence")
    result: list[dict[str, Any]] = []
    for index, game in enumerate(raw_games):
        if not isinstance(game, Mapping):
            raise ValueError(f"CUDA games[{index}] must be a mapping")
        row: dict[str, Any] = {
            "game_id": str(game["game_id"]),
            "opponent": str(game["opponent_id"]),
            "seat": "first" if game.get("focal_first") is True else "second",
            "outcome": str(game["focal_outcome"]),
        }
        for field in _OPTIONAL_DISTRIBUTION_FIELDS + ("error", "game_length"):
            if field in game:
                row[field] = game[field]
        result.append(row)
    return result


__all__ = [
    "DEFAULT_CONTRACT",
    "DistributionEquivalenceContract",
    "MetricVerdict",
    "compare_runtime_distributions",
    "load_official_cpu_report",
    "normalize_cuda_game_results",
    "normalize_official_cpu_report",
]
