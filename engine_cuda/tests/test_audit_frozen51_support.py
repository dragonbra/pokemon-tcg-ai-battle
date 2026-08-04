from __future__ import annotations

import unittest

from engine_cuda.tools.audit_frozen51_support import reachable_rules, walk_refs


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


if __name__ == "__main__":
    unittest.main()
