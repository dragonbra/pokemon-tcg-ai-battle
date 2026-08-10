from __future__ import annotations

import importlib
import json
import unittest


gate_a = importlib.import_module(
    "train.0042_full_model_design.semantic_parity.gate_a_rules"
)


class SemanticParityGateATest(unittest.TestCase):
    def test_pass_record_preserves_paired_metrics(self) -> None:
        payload = {
            "passed": True,
            "decisions_compared": 4199,
            "state_mismatches": 0,
            "status_mismatches": 0,
            "outcome_mismatches": 0,
        }
        result = gate_a.parse_paired_output(
            "START\t1\nCASE\t1\t" + json.dumps(payload) + "\n"
        )
        self.assertEqual(result["status"], "PASS")
        self.assertIsNone(result["first_divergence"])
        self.assertEqual(result["metrics"]["decisions_compared"], 4199)

    def test_failure_stops_at_first_model_free_divergence(self) -> None:
        first = (
            "CPU status/official terminal mismatch seed=1.decision=129."
            "pod_error=20.pod_detail=403"
        )
        result = gate_a.parse_paired_output(
            "START\t1\n" + first + "\nCASE\t1\t{\"passed\":true}\n"
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["case_ordinal"], 1)
        self.assertEqual(result["first_divergence"]["category"], "CUDA rules")
        self.assertEqual(result["first_divergence"]["message"], first)

    def test_truncated_process_is_incomplete_not_pass(self) -> None:
        result = gate_a.parse_paired_output("START\t2\n")
        self.assertEqual(result["status"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
