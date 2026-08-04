from __future__ import annotations

import sys
import unittest
from pathlib import Path


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))

from ptcg_cuda_engine.memory import GIB, estimate_memory  # noqa: E402
from ptcg_cuda_engine.policy_pool import PolicyPoolManifest  # noqa: E402


class MemoryEstimateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = PolicyPoolManifest.load(
            CUDA_ENGINE_ROOT / "configs" / "policy_pool.example.json"
        )

    def test_engine_only_estimate_matches_target_bands(self) -> None:
        small = estimate_memory(4096, self.manifest, workspace_root=WORKSPACE_ROOT)
        large = estimate_memory(16384, self.manifest, workspace_root=WORKSPACE_ROOT)
        self.assertGreater(small.engine_operational_bytes / GIB, 0.75)
        self.assertLess(small.engine_operational_bytes / GIB, 1.0)
        self.assertGreater(large.engine_operational_bytes / GIB, 2.6)
        self.assertLess(large.engine_operational_bytes / GIB, 3.2)
        self.assertEqual(small.assumptions["state_contract"], "OfficialStatePod ABI v6")

    def test_full_example_pool_fits_32_gib_estimate(self) -> None:
        estimate = estimate_memory(16384, self.manifest, workspace_root=WORKSPACE_ROOT)
        self.assertEqual(estimate.frozen_policy_count, 10)
        self.assertGreater(estimate.frozen_weight_bytes, 0)
        self.assertGreater(estimate.learner_training_bytes, 0)
        self.assertLess(estimate.total_bytes / GIB, 32.0)
        self.assertTrue(estimate.assumptions["cohort_capacity_requires_reset_quota"])


if __name__ == "__main__":
    unittest.main()
