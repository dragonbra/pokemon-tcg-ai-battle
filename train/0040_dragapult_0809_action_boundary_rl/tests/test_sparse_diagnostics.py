from __future__ import annotations

import importlib
import unittest

import torch

diagnostics = importlib.import_module("train.0040_dragapult_0809_action_boundary_rl.integrated.diagnostics")


class SparseDiagnosticsTest(unittest.TestCase):
    def test_norms_and_cosines_use_one_fixed_graph(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        metrics = diagnostics.gradient_diagnostics({
            "win": parameter.sum(), "prize": (parameter * 2).sum(),
            "meta": (parameter * torch.tensor([1.0, -1.0])).sum(),
        }, [parameter])
        self.assertAlmostEqual(metrics["gradient/win_prize/cosine"], 1.0, places=6)
        self.assertIn("gradient/meta/norm", metrics)

    def test_margin(self):
        margin = diagnostics.allocation_margin(torch.tensor([[2.0, 1.0, 0.0]]))
        self.assertGreater(float(margin), 0)


if __name__ == "__main__":
    unittest.main()
