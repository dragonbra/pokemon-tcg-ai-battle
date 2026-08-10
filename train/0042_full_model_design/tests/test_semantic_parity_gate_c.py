from __future__ import annotations

import importlib
import unittest


PROJECT = "train.0042_full_model_design"


class SemanticParityGateCTest(unittest.TestCase):
    def test_input_mismatch_precedes_logit_attribution(self) -> None:
        module = importlib.import_module(
            f"{PROJECT}.semantic_parity.gate_c_snapshot"
        )
        result = module.summarize({
            "case": "fixture",
            "decisions_replayed": 1,
            "fixed_action_cuda_state_errors": 0,
            "tensor_mismatch_counts": {"option_skill_mask": 1},
            "tensor_mismatch_details": [{
                "decision": 0,
                "field": "option_skill_mask",
                "flat_index": 0,
                "expected": True,
                "actual": False,
            }],
            "model_focal_decisions": 1,
            "model": {
                "passed": False,
                "root_logits_tolerance_failures": 1,
                "first_step_top1_divergences": 1,
            },
            "value_model": {},
        })
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["first_divergence"]["stage"], "observation_tensor")
        self.assertEqual(result["first_divergence"]["field"], "option_skill_mask")
        self.assertFalse(result["numeric_only_comparison_reached"])

    def test_exact_inputs_and_model_pass_release_gate(self) -> None:
        module = importlib.import_module(
            f"{PROJECT}.semantic_parity.gate_c_snapshot"
        )
        result = module.summarize({
            "fixed_action_cuda_state_errors": 0,
            "tensor_mismatch_counts": {},
            "tensor_mismatch_details": [],
            "model": {"passed": True},
            "value_model": {},
        })
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["numeric_only_comparison_reached"])


if __name__ == "__main__":
    unittest.main()
