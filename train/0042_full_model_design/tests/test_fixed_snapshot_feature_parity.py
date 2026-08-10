from __future__ import annotations

import copy
import importlib
import unittest


parity = importlib.import_module("train.0042_full_model_design.semantic_parity.fixed_snapshot_parity")


def record() -> dict:
    return {
        "observation_schema": "0038", "option_skill_relations": [1, 2],
        "option_effect_relations": [3, 4], "event_history": [5],
        "resource_ledger": {"deck": 33}, "deck_membership": [1, 0],
        "option_ordering": ["a", "b"], "option_mask": [True, True],
        "decision_gate": "STRATEGIC", "observation_tensor": [0.1, 0.2],
        "state_representation": [0.3], "option_representation": [0.4, 0.5],
        "root_logits": [1.0, 0.5], "allocation_logits": [0.2, 0.1],
        "value": [0.25],
    }


class FixedSnapshotFeatureParityTest(unittest.TestCase):
    def test_exact_and_float_tolerance_contract(self):
        same = copy.deepcopy(record())
        same["observation_tensor"][0] += 1e-7
        result = parity.compare_fixed_snapshot(record(), same)
        self.assertEqual(result.status, "PASS")
        self.assertTrue(result.greedy_top1_equal)
        changed = copy.deepcopy(record())
        changed["option_effect_relations"][0] += 1
        result = parity.compare_fixed_snapshot(record(), changed)
        self.assertEqual(result.first_difference.stage, "option_effect_relations")

    def test_stops_before_logits_on_input_failure(self):
        changed = copy.deepcopy(record())
        changed["option_mask"][0] = False
        changed["root_logits"] = [-100.0, 100.0]
        result = parity.compare_fixed_snapshot(record(), changed)
        self.assertEqual(result.first_difference.stage, "option_mask")
        self.assertIsNone(result.greedy_top1_equal)


if __name__ == "__main__":
    unittest.main()
