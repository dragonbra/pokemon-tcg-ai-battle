from __future__ import annotations

import inspect
import unittest

from engine_cuda.tools import evaluate_policy_0806_cuda as subject


REPLICATION_SEED = 843573811


class Policy0806CudaSeedReplicationTests(unittest.TestCase):
    def test_report_renderer_accepts_the_actual_evaluation_seed(self) -> None:
        self.assertIn("evaluation_seed", inspect.signature(subject._report_html).parameters)

    def test_replication_schedule_has_no_canonical_engine_seed_overlap(self) -> None:
        catalog, candidates = subject._catalog()
        candidate = next(
            item
            for item in candidates
            if item.package_manifest["frozen_deck_number"] == "002"
        )
        canonical = subject.build_cuda_schedule(
            focal_deck_id=candidate.name,
            entries=catalog.pool.schedule,
            evaluation_seed=subject.EVALUATION_SEED,
        )
        replication = subject.build_cuda_schedule(
            focal_deck_id=candidate.name,
            entries=catalog.pool.schedule,
            evaluation_seed=REPLICATION_SEED,
        )
        canonical_seeds = {job["engine_seed"] for job in canonical["jobs"]}
        replication_seeds = {job["engine_seed"] for job in replication["jobs"]}
        self.assertEqual(len(replication["jobs"]), 2048)
        self.assertEqual(len(replication_seeds), 2048)
        self.assertTrue(canonical_seeds.isdisjoint(replication_seeds))
        self.assertTrue(
            all(type(job["focal_won_toss"]) is bool for job in replication["jobs"])
        )
        self.assertTrue(all("focal_first" not in job for job in replication["jobs"]))


if __name__ == "__main__":
    unittest.main()
