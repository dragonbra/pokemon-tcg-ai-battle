from __future__ import annotations

import importlib
from pathlib import Path
import unittest

module = importlib.import_module("train.0040_dragapult_0809_action_boundary_rl.training.allocation_bc")


class AllocationBCTrainingTest(unittest.TestCase):
    def test_repository_relative_accepts_relative_cli_path(self):
        path = Path("rl_runs/0040_dragapult_0809_action_boundary_rl/example.pt")
        self.assertEqual(module._repository_relative(path), str(path))


if __name__ == "__main__":
    unittest.main()
