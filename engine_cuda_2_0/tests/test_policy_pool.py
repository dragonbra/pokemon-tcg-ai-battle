from __future__ import annotations

import sys
import unittest
from pathlib import Path


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))

from ptcg_cuda_engine.policy_pool import MAX_POLICIES, FixedRoutePlanner, PolicyPoolManifest  # noqa: E402


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

    def test_forty_policy_manifest_and_router_are_supported(self) -> None:
        policies = [
            {
                "policy_id": policy_id,
                "name": f"policy_{policy_id}",
                "deck": f"deck_{policy_id}.csv",
                "checkpoint": f"policy_{policy_id}.pt",
                "adapter": "foundation",
                "codec": "foundation_0020_codec_v1",
                "frozen": True,
            }
            for policy_id in range(40)
        ]
        manifest = PolicyPoolManifest.from_dict(
            {"name": "deck40", "max_policies": MAX_POLICIES, "policies": policies}
        )
        self.assertEqual(len(manifest.policies), 40)
        self.assertEqual(manifest.policies[-1].policy_id, 39)

        planner = FixedRoutePlanner(policy_count=40, capacity=2)
        plan = planner.route(list(range(40)) + list(range(40)), [True] * 80)
        self.assertEqual(plan.counts, (2,) * 40)
        self.assertEqual(plan.routes[39], (39, 79))


if __name__ == "__main__":
    unittest.main()
