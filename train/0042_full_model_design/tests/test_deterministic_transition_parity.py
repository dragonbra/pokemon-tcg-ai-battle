from __future__ import annotations

import copy
import importlib
import unittest


canonical_test = importlib.import_module("train.0042_full_model_design.tests.test_semantic_parity_canonical_state")
transition = importlib.import_module("train.0042_full_model_design.semantic_parity.deterministic_transition")


def step(backend: str) -> dict:
    return {"primitive_action": [0], "random_outcome": None,
            "continuation": [{"opcode": 79}], "legal_options": [{"type": 4}],
            "event_trace": [{"type": "move"}], "reward_delta": 0.0,
            "post_state": canonical_test.snapshot(backend)}


class DeterministicTransitionParityTest(unittest.TestCase):
    def test_identical_primitive_transition_passes(self):
        self.assertEqual(transition.compare_primitive_trace([step("official_cpu")], [step("cuda")]).status, "PASS")

    def test_first_continuation_or_state_difference_is_reported(self):
        cuda = step("cuda")
        cuda["continuation"][0]["opcode"] = 80
        result = transition.compare_primitive_trace([step("official_cpu")], [cuda])
        self.assertEqual(result.first_divergence.stage, "continuation")
        cuda = step("cuda")
        cuda["post_state"]["cards"][0]["damage"] += 10
        result = transition.compare_primitive_trace([step("official_cpu")], [cuda])
        self.assertEqual(result.first_divergence.stage, "post_state")
        self.assertEqual(result.first_divergence.field_difference.path, "cards[0].damage")


if __name__ == "__main__":
    unittest.main()
