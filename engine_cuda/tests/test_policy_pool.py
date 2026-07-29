from __future__ import annotations

import sys
import unittest
from pathlib import Path


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))

from ptcg_cuda_engine.policy_pool import FixedRoutePlanner, PolicyPoolManifest  # noqa: E402


class PolicyPoolTest(unittest.TestCase):
    def test_example_has_ten_plus_frozen_and_one_learner(self) -> None:
        manifest = PolicyPoolManifest.load(
            CUDA_ENGINE_ROOT / "configs" / "policy_pool.example.json"
        )
        self.assertGreaterEqual(manifest.frozen_policy_count, 10)
        self.assertIsNotNone(manifest.learner)
        self.assertEqual(sum(not policy.frozen for policy in manifest.policies), 1)
        self.assertGreater(len(manifest.adapter_families), 3)
        self.assertGreater(len(manifest.codecs), 1)
        self.assertTrue(
            all(Path(policy.checkpoint).suffix == ".pt" for policy in manifest.policies)
        )
        self.assertNotIn("alakazam_independent_v1", manifest.adapter_families)
        self.assertNotIn("alakazam_strategy_codec_v1", manifest.codecs)

    def test_fixed_route_planner_reports_overflow(self) -> None:
        planner = FixedRoutePlanner(policy_count=3, capacity=2)
        plan = planner.route([0, 1, 0, 2, 0, 1, 2], [True] * 7)
        self.assertEqual(plan.counts, (3, 2, 2))
        self.assertEqual(plan.routes[0], (0, 2))
        self.assertEqual(plan.overflow_envs, (4,))

    def test_not_ready_environments_are_not_routed(self) -> None:
        planner = FixedRoutePlanner(policy_count=2, capacity=3)
        plan = planner.route([0, 1, 1, 0], [True, False, True, False])
        self.assertEqual(plan.counts, (1, 1))
        self.assertEqual(plan.routes, ((0, -1, -1), (2, -1, -1)))


if __name__ == "__main__":
    unittest.main()
