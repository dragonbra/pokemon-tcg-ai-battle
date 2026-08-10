"""Aggregate the paired U270 CPU256 Value Search V0 experiment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _outcome(row: dict[str, Any]) -> str:
    return "W" if row.get("winner") == 0 else "L" if row.get("winner") == 1 else "D"


def _stats(values: list[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(finite),
        "mean": mean(finite) if finite else None,
        "median": median(finite) if finite else None,
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
    }


def analyze(baseline_root: Path, search_root: Path) -> dict[str, Any]:
    baseline_summary = _load(baseline_root / "summary.json")
    search_summary = _load(search_root / "summary.json")
    baseline_rows = _load(baseline_root / "games.json")["entries"]
    search_rows = _load(search_root / "games.json")["entries"]
    baseline_by_id = {row["game_id"]: row for row in baseline_rows}
    search_by_id = {row["game_id"]: row for row in search_rows}
    if baseline_by_id.keys() != search_by_id.keys():
        raise RuntimeError("paired runs do not contain the same game IDs")
    manifest_fields = (
        "opponent", "engine_seed", "policy_seed", "search_seed", "seed_replica",
        "seed_slot", "candidate_first", "candidate_won_toss",
        "toss_winner_selected_first",
    )
    manifest_mismatches = []
    paired = Counter({f"{before}->{after}": 0 for before in "WLD" for after in "WLD"})
    for game_id in sorted(baseline_by_id):
        before = baseline_by_id[game_id]
        after = search_by_id[game_id]
        differences = {
            field: [before.get(field), after.get(field)]
            for field in manifest_fields
            if before.get(field) != after.get(field)
        }
        if differences:
            manifest_mismatches.append({"game_id": game_id, "differences": differences})
        paired[f"{_outcome(before)}->{_outcome(after)}"] += 1

    decisions: list[dict[str, Any]] = []
    for path in sorted((search_root / "search_logs").glob("*.jsonl")):
        decisions.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    searched = [row for row in decisions if "search_seconds" in row]
    successful = [row for row in searched if row.get("eligible")]
    overrides = [row for row in successful if row.get("override")]
    fallbacks = [row for row in decisions if row.get("fallback_reason")]
    reject_reasons = Counter(
        str(row.get("fallback_reason") or row.get("eligibility_reason") or "UNKNOWN")
        for row in decisions if not row.get("eligible")
    )

    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        if row.get("decision_family"):
            family_rows[str(row["decision_family"])].append(row)
    families = {}
    family_names = (
        "basic_placement", "energy_target", "retreat_payment",
        "retreat_switch_target", "opponent_target", "attached_resource_destination",
    )
    for family in family_names:
        rows = family_rows[family]
        completed = [row for row in rows if row.get("eligible")]
        family_overrides = [row for row in completed if row.get("override")]
        families[family] = {
            "occurrences": len(rows),
            "searches": sum("search_seconds" in row for row in rows),
            "successful": len(completed),
            "candidate_states": sum(int(row.get("candidate_count", 0)) for row in completed),
            "overrides": len(family_overrides),
            "agreements": len(completed) - len(family_overrides),
            "agreement_rate": (
                (len(completed) - len(family_overrides)) / len(completed)
                if completed else None
            ),
            "margin": _stats([row["value_margin"] for row in completed]),
            "override_margin": _stats(
                [row["value_margin"] for row in family_overrides]
            ),
        }

    examples = []
    for row in overrides[:12]:
        game_id = row["game_id"]
        examples.append(
            {
                "game_id": game_id,
                "turn": row.get("turn"),
                "step": row.get("step"),
                "family": row.get("decision_family"),
                "board_summary": row.get("board_summary"),
                "policy_selection": row.get("policy_selection"),
                "candidate_selections": row.get("candidate_selections"),
                "candidate_option_payloads": row.get("candidate_option_payloads"),
                "candidate_values": row.get("candidate_values"),
                "winner": row.get("search_winner"),
                "margin": row.get("value_margin"),
                "baseline_result": _outcome(baseline_by_id[game_id]),
                "search_result": _outcome(search_by_id[game_id]),
            }
        )

    result = {
        "schema_version": "0040_u270_value_search_v0_ab_analysis_v1",
        "manifest_exact": not manifest_mismatches,
        "manifest_mismatches": manifest_mismatches,
        "baseline": baseline_summary,
        "search": search_summary,
        "win_rate_delta": search_summary["win_rate"] - baseline_summary["win_rate"],
        "paired": dict(sorted(paired.items())),
        "search_statistics": {
            "total_policy_decisions": len(decisions),
            "eligible_decisions": len(successful),
            "search_invocations": len(searched),
            "candidate_states_evaluated": sum(
                int(row.get("candidate_count", 0)) for row in successful
            ),
            "overrides": len(overrides),
            "agreements": len(successful) - len(overrides),
            "agreement_rate": (
                (len(successful) - len(overrides)) / len(successful)
                if successful else None
            ),
            "override_rate": len(overrides) / len(successful) if successful else None,
            "eligible_rate_of_policy_decisions": (
                len(successful) / len(decisions) if decisions else None
            ),
            "override_rate_of_policy_decisions": (
                len(overrides) / len(decisions) if decisions else None
            ),
            "value_margin": _stats([row["value_margin"] for row in successful]),
            "override_value_margin": _stats(
                [row["value_margin"] for row in overrides]
            ),
        },
        "family_breakdown": families,
        "failures_and_rejects": {
            "fallbacks": len(fallbacks),
            "fallback_reasons": dict(Counter(str(row["fallback_reason"]) for row in fallbacks)),
            "noneligible_reasons": dict(reject_reasons.most_common()),
        },
        "performance": {
            "baseline_wall_seconds": baseline_summary["wall_seconds"],
            "search_wall_seconds": search_summary["wall_seconds"],
            "wall_overhead_seconds": search_summary["wall_seconds"] - baseline_summary["wall_seconds"],
            "wall_overhead_fraction": (
                search_summary["wall_seconds"] / baseline_summary["wall_seconds"] - 1.0
            ),
            "recorded_search_seconds": sum(float(row.get("search_seconds", 0.0)) for row in searched),
            "recorded_value_seconds": sum(float(row.get("value_seconds", 0.0)) for row in searched),
            "average_recorded_cost_per_game": sum(
                float(row.get("total_seconds", 0.0)) for row in searched
            ) / len(search_rows),
            "average_recorded_cost_per_searched_decision": mean(
                [float(row.get("total_seconds", 0.0)) for row in searched]
            ) if searched else None,
        },
        "representative_disagreements": examples,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.baseline, args.search)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["search_statistics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
