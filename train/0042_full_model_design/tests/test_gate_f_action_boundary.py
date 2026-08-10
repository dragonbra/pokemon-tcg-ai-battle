from __future__ import annotations

import importlib
import math
import unittest


gate = importlib.import_module("train.0042_full_model_design.semantic_parity.gate_f_action_boundary")


class GateFActionBoundaryTest(unittest.TestCase):
    def test_complete_forced_and_n1_n8_contract_passes(self):
        cases = [{"name": "forced", "kind": "forced", "policy_calls": 0,
                  "value_calls": 0, "ppo_transitions": 0,
                  "history_before": "a", "history_after": "b"}]
        for n in range(1, 9):
            cases.append({"name": f"n{n}", "kind": "phantom", "target_count": n,
                          "policy_calls": 1, "value_calls": 1, "ppo_transitions": 1,
                          "macro_transactions": 1, "primitive_selects": 6,
                          "allocation_count": math.comb(n + 5, 6),
                          "stable_serial_relocation": True,
                          "canonical_alias_collapse": True,
                          "primitive_state_parity": True})
        self.assertEqual(gate.validate_action_boundary_cases(cases)["status"], "PASS")

    def test_drift_cannot_reenter_policy(self):
        cases = [{"name": f"n{n}", "kind": "phantom", "target_count": n,
                  "policy_calls": 1, "value_calls": 1, "ppo_transitions": 1,
                  "macro_transactions": 1, "primitive_selects": 6,
                  "allocation_count": math.comb(n + 5, 6),
                  "stable_serial_relocation": True, "canonical_alias_collapse": True,
                  "primitive_state_parity": True,
                  **({"macro_drift": True, "policy_calls_after_drift": 2} if n == 6 else {})}
                 for n in range(1, 9)]
        result = gate.validate_action_boundary_cases(cases)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any(row["field"] == "policy_calls_after_drift" for row in result["failures"]))


if __name__ == "__main__":
    unittest.main()
