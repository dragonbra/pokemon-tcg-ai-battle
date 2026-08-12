"""Validate and summarize sharded CUDA pair-matrix output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("outputs", type=Path, nargs="+")
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--games-per-case", type=int, required=True)
    args = parser.parse_args()
    cases = {}
    for path in args.outputs:
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.startswith("CASE\t"):
                continue
            _, ordinal_text, payload = raw.split("\t", 2)
            ordinal = int(ordinal_text)
            if ordinal in cases:
                raise SystemExit(f"duplicate case ordinal: {ordinal}")
            cases[ordinal] = json.loads(payload)
    expected = set(range(args.expected_cases))
    missing = sorted(expected - set(cases))
    unexpected = sorted(set(cases) - expected)
    mismatch_fields = (
        "canonical_byte_mismatches", "outcome_mismatches",
        "state_mismatches", "status_mismatches",
    )
    bad = {
        ordinal: row for ordinal, row in cases.items()
        if not row.get("passed")
        or row.get("seed_count") != args.games_per_case
        or any(row.get(field, 0) != 0 for field in mismatch_fields)
    }
    result = {
        "passed": not missing and not unexpected and not bad,
        "expected_cases": args.expected_cases,
        "completed_cases": len(cases),
        "games": len(cases) * args.games_per_case,
        "decisions_compared": sum(
            row.get("decisions_compared", 0) for row in cases.values()
        ),
        "player0_wins": sum(row.get("player0_wins", 0) for row in cases.values()),
        "player1_wins": sum(row.get("player1_wins", 0) for row in cases.values()),
        "draws": sum(row.get("draws", 0) for row in cases.values()),
        "unfinished_battles": sum(
            row.get("unfinished_battles", 0) for row in cases.values()
        ),
        **{
            field: sum(row.get(field, 0) for row in cases.values())
            for field in mismatch_fields
        },
        "missing_ordinals": missing,
        "unexpected_ordinals": unexpected,
        "bad_ordinals": sorted(bad),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
