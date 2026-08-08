from __future__ import annotations

import importlib
import unittest

import torch


PROJECT = "train.0038_action_boundary_rl"


class WorkerFeatureParityTest(unittest.TestCase):
    def test_worker_local_stateless_compiler_matches_central_reference(self) -> None:
        runner = importlib.import_module(f"{PROJECT}.training.run_full_semantic")
        parity = importlib.import_module(f"{PROJECT}.parity")
        worker_module = importlib.import_module(f"{PROJECT}.rollout.worker_compiler")
        online_module = importlib.import_module(
            f"{PROJECT}.semantic_policy.deployment.online_runtime"
        )
        collate = importlib.import_module(f"{PROJECT}.semantic_policy.features.collate")
        config_module = importlib.import_module(f"{PROJECT}.semantic_policy.model.config")
        opponent = runner.load_frozen_catalog()[0]
        observations = parity.collect_official_observations(
            runner.runtime_root(),
            runner.focal_deck(),
            opponent.deck,
            focal_index=0,
            decisions=4,
            seed=370037,
        )
        reference = online_module.OnlineCausalEncoder(
            0, runner.focal_deck(), config_module.ModelConfig()
        )
        worker = worker_module.WorkerLocalCompiler(0, runner.focal_deck())
        for observation in observations:
            expected = reference.encode(observation)
            actual = collate.collate_canonical_records([worker.compile(observation)])
            self.assertEqual(set(actual), set(expected))
            for name in expected:
                self.assertTrue(torch.equal(actual[name], expected[name]), name)


if __name__ == "__main__":
    unittest.main()
