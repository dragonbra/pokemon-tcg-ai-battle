from __future__ import annotations

import importlib
import unittest

power = importlib.import_module("train.0040_dragapult_0809_action_boundary_rl.evaluation.power")


class EvaluationPowerTest(unittest.TestCase):
    def test_one_point_requires_four_times_two_point_budget(self):
        one = power.paired_required_games(.01, .1)
        two = power.paired_required_games(.02, .1)
        self.assertAlmostEqual(one / two, 4.0, delta=.01)

    def test_2048_caveat_is_explicit(self):
        report = power.panel_power_report()
        self.assertIn("not automatically significant", report["interpretation"])
        self.assertGreater(report["mde_80_power"]["0.1"], .019)


if __name__ == "__main__":
    unittest.main()
