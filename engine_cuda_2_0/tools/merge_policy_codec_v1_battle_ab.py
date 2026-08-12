from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_ARCHETYPES = {
    "lucario",
    "cynthia",
    "kangaskhan_crustle",
    "marnie",
    "alakazam",
    "dragapult",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("scope") != "CPU official engine; CPU/GPU frozen PolicyCodecV1 policy A/B; no training":
        raise ValueError(f"unexpected report scope: {path}")
    return payload


def sum_matchups(matchups: list[dict[str, Any]]) -> dict[str, Any]:
    count_fields = (
        "pairs",
        "complete_pairs",
        "decided_pairs",
        "errors",
        "invariant_errors",
        "comparable_target_decisions",
        "raw_action_differences",
        "semantic_action_differences",
        "trajectory_divergences",
        "outcome_discordances",
        "cpu_wins",
        "gpu_wins",
        "cpu_only_wins",
        "gpu_only_wins",
    )
    totals = {name: sum(int(row[name]) for row in matchups) for name in count_fields}
    comparable = max(1, totals["comparable_target_decisions"])
    decided = max(1, totals["decided_pairs"])
    complete = max(1, totals["complete_pairs"])
    totals.update(
        raw_action_difference_rate=totals["raw_action_differences"] / comparable,
        semantic_action_difference_rate=totals["semantic_action_differences"] / comparable,
        outcome_discordance_rate=totals["outcome_discordances"] / decided,
        cpu_win_rate=totals["cpu_wins"] / decided,
        gpu_win_rate=totals["gpu_wins"] / decided,
        gpu_minus_cpu_win_rate=(totals["gpu_wins"] - totals["cpu_wins"]) / decided,
        average_cpu_steps=sum(float(row["average_cpu_steps"]) * int(row["complete_pairs"]) for row in matchups) / complete,
        average_gpu_steps=sum(float(row["average_gpu_steps"]) * int(row["complete_pairs"]) for row in matchups) / complete,
    )
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description="Replace selected matchup rows and merge a final six-BC A/B report.")
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--replacement-report", type=Path, required=True)
    parser.add_argument("--replace-archetype", default="marnie")
    parser.add_argument("--opponent-pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = load_report(args.base_report.resolve())
    replacement = load_report(args.replacement_report.resolve())
    matchups = [
        row for row in base["matchups"] if row.get("archetype") != args.replace_archetype
    ] + [
        row for row in replacement["matchups"] if row.get("archetype") == args.replace_archetype
    ]
    if len(matchups) != 6 or {row.get("archetype") for row in matchups} != EXPECTED_ARCHETYPES:
        raise RuntimeError("merged report does not contain exactly one row for each requested archetype")
    matchups.sort(key=lambda row: int(row["opponent_pool_index"]))
    totals = sum_matchups(matchups)
    thresholds = dict(base["thresholds"])
    checks = {
        "all_pairs_complete": totals["complete_pairs"] == totals["pairs"],
        "no_invariant_errors": totals["invariant_errors"] == 0,
        "outcome_discordance_within_limit": totals["outcome_discordance_rate"] <= thresholds["max_outcome_discordance_rate"],
        "win_rate_delta_within_limit": abs(totals["gpu_minus_cpu_win_rate"]) <= thresholds["max_win_rate_delta"],
        "semantic_action_difference_within_limit": totals["semantic_action_difference_rate"] <= thresholds["max_semantic_action_difference_rate"],
    }
    old_name = next(
        row["opponent"] for row in base["matchups"] if row.get("archetype") == args.replace_archetype
    )
    divergences = [
        row for row in base.get("first_divergences", []) if row.get("opponent") != old_name
    ] + list(replacement.get("first_divergences", []))
    opponent_pool = args.opponent_pool.resolve()
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "scope": base["scope"],
        "composition": "five matchups retained from base plus rerun Prize Control Marnie matchup",
        "source_reports": [str(args.base_report.resolve()), str(args.replacement_report.resolve())],
        "opponent_pool_manifest": str(opponent_pool),
        "opponent_pool_manifest_sha256": sha256_file(opponent_pool),
        "games_per_opponent": 100,
        "opponent_count": 6,
        "thresholds": thresholds,
        "checks": checks,
        "totals": totals,
        "matchups": matchups,
        "first_divergences": divergences,
        "known_engine_divergence": "KNOWN_DIVERGENCE_660_1207 retained and not redefined",
    }
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.out.resolve().write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
