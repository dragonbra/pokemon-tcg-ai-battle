"""Compare two seat-balanced frozen-policy official-engine evaluations."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPORT_DATA_START = '<script id="report-data" type="application/json">'
REPORT_DATA_END = "</script>"
Z_95 = 1.959963984540054


def load_report(path: Path) -> dict[str, Any]:
    document = path.read_text(encoding="utf-8")
    try:
        payload = document.split(REPORT_DATA_START, 1)[1].split(REPORT_DATA_END, 1)[0]
    except IndexError as error:
        raise ValueError(f"report-data payload is missing: {path}") from error
    value = json.loads(payload)
    if not isinstance(value, dict) or not isinstance(value.get("games"), list):
        raise ValueError(f"invalid report-data payload: {path}")
    return value


def wilson_interval(wins: int, games: int, z: float = Z_95) -> tuple[float, float]:
    if games < 1 or not 0 <= wins <= games:
        raise ValueError("wins/games must describe a nonempty binomial sample")
    rate = wins / games
    denominator = 1.0 + z * z / games
    center = (rate + z * z / (2.0 * games)) / denominator
    radius = z * math.sqrt(
        rate * (1.0 - rate) / games + z * z / (4.0 * games * games)
    ) / denominator
    return center - radius, center + radius


def compare_binomial(
    baseline_wins: int,
    baseline_games: int,
    candidate_wins: int,
    candidate_games: int,
) -> dict[str, Any]:
    baseline_low, baseline_high = wilson_interval(baseline_wins, baseline_games)
    candidate_low, candidate_high = wilson_interval(candidate_wins, candidate_games)
    baseline_rate = baseline_wins / baseline_games
    candidate_rate = candidate_wins / candidate_games
    difference = candidate_rate - baseline_rate

    # Newcombe's hybrid-score interval for two independent proportions.
    difference_low = difference - math.sqrt(
        (candidate_rate - candidate_low) ** 2
        + (baseline_high - baseline_rate) ** 2
    )
    difference_high = difference + math.sqrt(
        (candidate_high - candidate_rate) ** 2
        + (baseline_rate - baseline_low) ** 2
    )

    pooled_rate = (baseline_wins + candidate_wins) / (
        baseline_games + candidate_games
    )
    standard_error = math.sqrt(
        pooled_rate
        * (1.0 - pooled_rate)
        * (1.0 / baseline_games + 1.0 / candidate_games)
    )
    if standard_error == 0.0:
        p_value = 1.0 if difference == 0.0 else 0.0
    else:
        z_score = difference / standard_error
        p_value = math.erfc(abs(z_score) / math.sqrt(2.0))
    return {
        "baseline": {
            "wins": baseline_wins,
            "games": baseline_games,
            "win_rate": baseline_rate,
            "wilson_95": [baseline_low, baseline_high],
        },
        "candidate": {
            "wins": candidate_wins,
            "games": candidate_games,
            "win_rate": candidate_rate,
            "wilson_95": [candidate_low, candidate_high],
        },
        "difference": difference,
        "difference_95": [difference_low, difference_high],
        "two_sided_score_p": p_value,
    }


def _strata(games: Sequence[Mapping[str, Any]]) -> Counter[tuple[str, bool]]:
    return Counter(
        (str(game.get("opponent")), bool(game.get("candidate_first")))
        for game in games
    )


def validate_comparison_contract(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    baseline_games = baseline.get("games")
    candidate_games = candidate.get("games")
    if not isinstance(baseline_games, list) or not isinstance(candidate_games, list):
        raise ValueError("both reports must contain game records")
    if _strata(baseline_games) != _strata(candidate_games):
        raise ValueError("opponent/seat allocation differs between reports")
    for label, report in (("baseline", baseline), ("candidate", candidate)):
        manifest = report.get("manifest")
        if not isinstance(manifest, dict):
            raise ValueError(f"{label} manifest is missing")
        if manifest.get("swap_policy") != "alternate_candidate_first":
            raise ValueError(f"{label} seat policy is not alternating")
    baseline_manifest = baseline["manifest"]
    candidate_manifest = candidate["manifest"]
    for field in ("engine_runtime", "metric_profile", "worker_cpu_threads"):
        if baseline_manifest.get(field) != candidate_manifest.get(field):
            raise ValueError(f"evaluation contract differs at manifest.{field}")
    baseline_opponents = [item.get("package_hash") for item in baseline_manifest["opponents"]]
    candidate_opponents = [item.get("package_hash") for item in candidate_manifest["opponents"]]
    if baseline_opponents != candidate_opponents:
        raise ValueError("opponent package snapshot differs between reports")


def _selected_games(
    games: Sequence[Mapping[str, Any]],
    *,
    opponents: set[str] | None = None,
    candidate_first: bool | None = None,
) -> list[Mapping[str, Any]]:
    return [
        game
        for game in games
        if (opponents is None or str(game.get("opponent")) in opponents)
        and (candidate_first is None or bool(game.get("candidate_first")) is candidate_first)
    ]


def _outcomes(games: Iterable[Mapping[str, Any]]) -> tuple[int, int, int, int]:
    values = list(games)
    wins = sum(game.get("status") == "finished" and game.get("winner") == 0 for game in values)
    completed = sum(game.get("status") == "finished" for game in values)
    errors = sum(game.get("status") != "finished" for game in values)
    return wins, len(values), completed, errors


def _group_comparison(
    baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    baseline_wins, baseline_games, baseline_completed, baseline_errors = _outcomes(baseline)
    candidate_wins, candidate_games, candidate_completed, candidate_errors = _outcomes(candidate)
    result = compare_binomial(
        baseline_wins, baseline_games, candidate_wins, candidate_games
    )
    result["baseline"].update(completed=baseline_completed, errors=baseline_errors)
    result["candidate"].update(completed=candidate_completed, errors=candidate_errors)
    return result


def compare_reports(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    validate_comparison_contract(baseline, candidate)
    baseline_games = baseline["games"]
    candidate_games = candidate["games"]
    all_opponents = {str(game["opponent"]) for game in baseline_games}

    groups: dict[str, Any] = {
        "overall": _group_comparison(baseline_games, candidate_games),
        "candidate_first": _group_comparison(
            _selected_games(baseline_games, candidate_first=True),
            _selected_games(candidate_games, candidate_first=True),
        ),
        "candidate_second": _group_comparison(
            _selected_games(baseline_games, candidate_first=False),
            _selected_games(candidate_games, candidate_first=False),
        ),
    }
    by_opponent = {
        opponent: _group_comparison(
            _selected_games(baseline_games, opponents={opponent}),
            _selected_games(candidate_games, opponents={opponent}),
        )
        for opponent in sorted(all_opponents)
    }
    overall = groups["overall"]
    lower, upper = overall["difference_95"]
    if lower > 0 and not overall["baseline"]["errors"] and not overall["candidate"]["errors"]:
        verdict = "improved"
    elif upper < 0:
        verdict = "regressed"
    else:
        verdict = "inconclusive"
    return {
        "schema_version": "0018_frozen_policy_ab_v1",
        "design": "independent_stratified_balanced_ab",
        "paired": False,
        "paired_limit": (
            "The unmodified official binary seeds battles with random_device and exposes no "
            "battle-seed API."
        ),
        "verdict": verdict,
        "groups": groups,
        "by_opponent": by_opponent,
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_markdown(result: Mapping[str, Any], baseline_path: Path, candidate_path: Path) -> str:
    lines = [
        "# 0018 frozen-policy A/B result",
        "",
        f"Verdict: **{result['verdict']}**",
        "",
        "This is an independently randomized, opponent/seat-balanced A/B evaluation, not a "
        "shared-seed paired test. The unmodified official binary uses `random_device` internally.",
        "",
        f"- Baseline report: `{baseline_path}`",
        f"- Candidate report: `{candidate_path}`",
        "",
        "| Group | BC wins/games | BC rate | RL wins/games | RL rate | RL − BC (95% CI) | p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "overall": "Overall",
        "candidate_first": "Candidate first",
        "candidate_second": "Candidate second",
    }
    for key, label in labels.items():
        group = result["groups"][key]
        baseline = group["baseline"]
        candidate = group["candidate"]
        low, high = group["difference_95"]
        lines.append(
            f"| {label} | {baseline['wins']}/{baseline['games']} | "
            f"{_percent(baseline['win_rate'])} | {candidate['wins']}/{candidate['games']} | "
            f"{_percent(candidate['win_rate'])} | {_percent(group['difference'])} "
            f"[{_percent(low)}, {_percent(high)}] | {group['two_sided_score_p']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Per-opponent audit",
            "",
            "| Opponent | BC | RL | Difference | 95% CI | p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for opponent, group in result["by_opponent"].items():
        low, high = group["difference_95"]
        lines.append(
            f"| `{opponent}` | {_percent(group['baseline']['win_rate'])} | "
            f"{_percent(group['candidate']['win_rate'])} | {_percent(group['difference'])} | "
            f"[{_percent(low)}, {_percent(high)}] | {group['two_sided_score_p']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            "RL is called improved only when the overall 95% difference interval excludes zero "
            "in the positive direction, both arms complete without runtime errors, and the result "
            "is not solely attributable to one seat or one matchup. Training rolling-window peaks "
            "are not used in this verdict.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args(argv)
    baseline = load_report(args.baseline)
    candidate = load_report(args.candidate)
    result = compare_reports(baseline, candidate)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(
        render_markdown(result, args.baseline, args.candidate), encoding="utf-8"
    )
    print(json.dumps(result["groups"]["overall"], indent=2, sort_keys=True))
    print(f"verdict: {result['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
