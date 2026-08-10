from __future__ import annotations

import importlib
import unittest


audit = importlib.import_module(
    "train.0040_dragapult_0809_action_boundary_rl.diagnostics.strategy_adapter_v2_audit"
)


class StrategyAdapterV2AuditTest(unittest.TestCase):
    def test_zero_gate_starts_then_releases_residual_branch(self) -> None:
        result = audit.zero_gate_startup_probe()
        self.assertTrue(result["passes"], result)
        self.assertNotEqual(result["first_gate_grad"], 0.0)
        self.assertEqual(result["first_residual_grad_l1"], 0.0)
        self.assertNotEqual(result["gate_after_step"], 0.0)
        self.assertGreater(result["second_residual_grad_l1"], 0.0)


if __name__ == "__main__":
    unittest.main()
