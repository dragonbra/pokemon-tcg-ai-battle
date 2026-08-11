from __future__ import annotations

import importlib
import unittest


PROJECT = "train.0042_full_model_design"
RUNNER = importlib.import_module(f"{PROJECT}.training.run_full_semantic")
PPO = importlib.import_module(f"{PROJECT}.training.ppo_full_semantic")
FROZEN = importlib.import_module(f"{PROJECT}.evaluation.frozen_jobs")


class LongRun0042ContractTest(unittest.TestCase):
    def test_default_formal_run_is_normal_lr_unbounded_prize(self) -> None:
        config = RUNNER.RunConfig()
        self.assertEqual(config.version, "V1_ppo_protocol_v2_baseline")
        self.assertIsNone(config.updates)
        self.assertEqual(config.preset_name, "FULL_MODEL")
        self.assertEqual(config.games_per_update, 256)
        self.assertEqual(config.trajectory_games_per_update, 256)
        self.assertEqual(config.ppo_minibatch_size, 2048)
        self.assertEqual(config.ppo_epochs, 3)
        self.assertEqual(config.eval_every, 10)
        self.assertEqual(config.ppo.decoder_learning_rate, 5.0e-6)
        self.assertEqual(config.ppo.policy_adapter_learning_rate, 5.0e-6)
        self.assertEqual(config.ppo.allocation_learning_rate, 5.0e-6)
        self.assertEqual(config.ppo.value_learning_rate, 1.0e-4)
        self.assertEqual(config.ppo.prize_learning_rate, 1.0e-4)

    def test_optimizer_base_rates_are_not_accelerated(self) -> None:
        self.assertEqual(PPO.PPOConfig().decoder_learning_rate, 5.0e-6)
        self.assertEqual(PPO.PPOConfig().policy_adapter_learning_rate, 5.0e-6)
        self.assertEqual(PPO.PPOConfig().allocation_learning_rate, 5.0e-6)
        self.assertEqual(PPO.PPOConfig().value_learning_rate, 1.0e-4)
        self.assertEqual(PPO.PPOConfig().prize_learning_rate, 1.0e-4)
        self.assertEqual(PPO.PPOConfig().meta_anchor_coef, 0.10)

    def test_fresh_checkpoint_branch_cannot_alias_update0_resume(self) -> None:
        config = RUNNER.RunConfig(
            version="V2_checkpoint_branch_test",
            launch_formal=True,
            resume_update0=True,
            initial_model_checkpoint="sealed-u250.pt",
        )
        with self.assertRaisesRegex(ValueError, "cannot also branch"):
            config.validate()

    def test_frozen_contract_is_policy0809_identity_bound(self) -> None:
        self.assertEqual(
            FROZEN.CANONICAL_CONTRACT_ID,
            "0042_frozen_0809_seeded_agent_first_player_v1",
        )
        jobs, schedule = FROZEN.build_frozen_jobs(
            focal_deck_id=RUNNER.FOCAL_DECK_ID,
            focal_deck=RUNNER.focal_deck(),
            runtime_root=RUNNER.runtime_root(),
            source_policy_update=5,
            focal_deployment_identity=FROZEN.EXPECTED_007_U0_DEPLOYMENT_SHA256,
            opponent_effective_policy_sha256=(
                "0d0091140d72e78f1070c549b8367583a9d4f5537d0cb67decab40ac3bb9da96"
            ),
        )
        self.assertEqual(len(jobs), 2048)
        self.assertEqual(schedule, FROZEN.EXPECTED_007_SCHEDULE_SHA256)
        self.assertTrue(all(job.focal_won_toss is not None for job in jobs))
        self.assertEqual(len({job.seed for job in jobs}), 2048)

    def test_cpu_cuda_value_parity_tolerance_covers_observed_reduction_jitter(self) -> None:
        self.assertEqual(RUNNER.ATTESTED_CPU_CUDA_VALUE_ATOL, 5.0e-6)
        self.assertLess(2.2649765014648438e-6, RUNNER.ATTESTED_CPU_CUDA_VALUE_ATOL)


if __name__ == "__main__":
    unittest.main()
