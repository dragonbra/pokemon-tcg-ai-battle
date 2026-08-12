from __future__ import annotations

import unittest

from engine_cuda_2_0.tools.analyze_0022_effect_branch_coverage import (
    build_effect_inventory,
)


def effect(
    *,
    is_condition: bool = False,
    linked_skill_id: int = 0,
    linked_attack_id: int = 0,
) -> dict[str, int]:
    return {
        "flags": 1 if is_condition else 0,
        "type": 0,
        "condition_type": 2 if is_condition else 0,
        "linked_skill_id": linked_skill_id,
        "linked_attack_id": linked_attack_id,
    }


class EffectBranchInventoryTest(unittest.TestCase):
    def test_offsets_follow_rule_pack_order_and_linked_owners(self) -> None:
        payload = {
            "cards": [
                {
                    "id": 10,
                    "ability_id": 1,
                    "play_id": 0,
                    "delay_id": 0,
                    "attack_ids": [3],
                }
            ],
            "skills": [
                {
                    "id": 1,
                    "card_id": 10,
                    "effects": [effect(linked_skill_id=2)],
                },
                {
                    "id": 2,
                    "card_id": 11,
                    "effects": [effect(is_condition=True)],
                },
            ],
            "attacks": [
                {
                    "id": 3,
                    "card_id": 10,
                    "pre_effects": [effect()],
                    "post_effects": [effect(linked_skill_id=2)],
                }
            ],
        }

        inventory, metadata = build_effect_inventory(payload, {10})

        self.assertEqual(sorted(inventory), [0, 1, 2, 3])
        self.assertEqual(inventory[0]["owner_kind"], "skill")
        self.assertEqual(inventory[1]["owner_id"], 2)
        self.assertTrue(inventory[1]["is_condition"])
        self.assertEqual(inventory[2]["phase"], "pre_effects")
        self.assertEqual(inventory[3]["phase"], "post_effects")
        self.assertEqual(metadata["flattened_effect_count"], 4)
        self.assertEqual(metadata["reachable_skill_count"], 2)
        self.assertEqual(metadata["reachable_attack_count"], 1)
        self.assertEqual(metadata["missing_card_ids"], [])
        self.assertEqual(metadata["missing_owner_ids"], [])


if __name__ == "__main__":
    unittest.main()
