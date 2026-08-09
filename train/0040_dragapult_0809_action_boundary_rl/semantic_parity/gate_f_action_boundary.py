"""Release-gate validator for Action Boundary counters and macro lifecycles."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def validate_action_boundary_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    covered_n: set[int] = set()
    for case in cases:
        name = str(case.get("name", "unnamed"))
        kind = case.get("kind")
        if kind == "forced":
            expected = {"policy_calls": 0, "value_calls": 0, "ppo_transitions": 0}
            for field, value in expected.items():
                if case.get(field) != value:
                    failures.append({"case": name, "field": field, "expected": value, "actual": case.get(field)})
            if case.get("history_before") == case.get("history_after"):
                failures.append({"case": name, "field": "observe_only_history", "expected": "updated", "actual": "unchanged"})
        elif kind == "phantom":
            n = int(case.get("target_count", 0))
            covered_n.add(n)
            expected = {"policy_calls": 1, "value_calls": 1, "ppo_transitions": 1, "macro_transactions": 1, "primitive_selects": 6}
            for field, value in expected.items():
                if case.get(field) != value:
                    failures.append({"case": name, "field": field, "expected": value, "actual": case.get(field)})
            if case.get("allocation_count") != math.comb(n + 5, 6):
                failures.append({"case": name, "field": "allocation_count", "expected": math.comb(n + 5, 6), "actual": case.get("allocation_count")})
            if case.get("macro_drift") and case.get("policy_calls_after_drift", 1) != 1:
                failures.append({"case": name, "field": "policy_calls_after_drift", "expected": 1, "actual": case.get("policy_calls_after_drift")})
            for boolean in ("stable_serial_relocation", "canonical_alias_collapse", "primitive_state_parity"):
                if case.get(boolean) is not True:
                    failures.append({"case": name, "field": boolean, "expected": True, "actual": case.get(boolean)})
    if covered_n != set(range(1, 9)):
        failures.append({"case": "coverage", "field": "phantom_target_counts", "expected": list(range(1, 9)), "actual": sorted(covered_n)})
    return {"gate": "F", "status": "PASS" if not failures else "FAIL", "case_count": len(cases), "failures": failures}


__all__ = ["validate_action_boundary_cases"]
