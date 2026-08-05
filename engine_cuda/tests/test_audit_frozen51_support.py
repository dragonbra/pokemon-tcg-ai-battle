from __future__ import annotations

import unittest

from engine_cuda.tools.audit_frozen51_support import (
    reachable_rules,
    validate_matrix_evidence,
    walk_refs,
)


class Frozen51SupportAuditTests(unittest.TestCase):
    def test_walk_refs_finds_nested_links(self) -> None:
        value = {"effects": [{"linked_skill_id": 7}, {"nested": {"attack_id": 9}}]}
        self.assertEqual(set(walk_refs(value)), {("skills", 7), ("attacks", 9)})

    def test_reachable_rules_follows_transitive_attack_and_skill_links(self) -> None:
        rules = {
            "cards": [{"id": 1, "ability_id": 2, "play_id": 0, "delay_id": 0, "attack_ids": [3]}],
            "skills": [{"id": 2, "effects": [{"linked_attack_id": 4}]}],
            "attacks": [
                {"id": 3, "pre_effects": [{"linked_skill_id": 5}], "post_effects": []},
                {"id": 4, "pre_effects": [], "post_effects": []},
            ],
        }
        # Missing linked owner 5 is still part of the dependency closure.
        self.assertEqual(
            reachable_rules(rules, [1]),
            {"cards": [1], "skills": [2, 5], "attacks": [3, 4]},
        )

    def test_matrix_evidence_requires_every_non_mirror_ordered_pair(self) -> None:
        cases = []
        for left in ("a", "b", "c"):
            for right in ("a", "b", "c"):
                if left == right:
                    continue
                cases.append(
                    {
                        "deck0_name": left,
                        "deck1_name": right,
                        "passed": True,
                        "state_mismatches": 0,
                        "status_mismatches": 0,
                        "outcome_mismatches": 0,
                        "unfinished_battles": 0,
                    }
                )
        report = {
            "passed": True,
            "contract": "official_cpu_reference_cuda_ordered_battle_matrix_v1",
            "case_count": 6,
            "completed_cases": 6,
            "battles_compared": 6,
            "state_mismatches": 0,
            "status_mismatches": 0,
            "outcome_mismatches": 0,
            "unfinished_battles": 0,
            "first_failure": None,
            "cases": cases,
        }
        self.assertEqual(len(validate_matrix_evidence(report, ("a", "b", "c"))), 6)
        report["cases"] = cases[:-1] + [cases[0]]
        with self.assertRaisesRegex(ValueError, "missing, duplicated"):
            validate_matrix_evidence(report, ("a", "b", "c"))


if __name__ == "__main__":
    unittest.main()
