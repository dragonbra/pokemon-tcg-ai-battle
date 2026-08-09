from __future__ import annotations

import importlib
import unittest


PROJECT = "train.0040_dragapult_0809_action_boundary_rl"
RUNNER = importlib.import_module(f"{PROJECT}.training.run_full_semantic")
PPO = importlib.import_module(f"{PROJECT}.training.ppo_full_semantic")
FROZEN = importlib.import_module(f"{PROJECT}.evaluation.frozen_jobs")


class LongRun0040ContractTest(unittest.TestCase):
    def test_default_formal_run_is_normal_lr_unbounded_prize(self) -> None:
        config = RUNNER.RunConfig()
        self.assertEqual(config.version, "V2_snapshot_loader_fix_long_run")
        self.assertIsNone(config.updates)
        self.assertEqual(config.preset_name, "PRIZE")
        self.assertFalse(config.accelerated_transfer_acceptance)
        self.assertEqual(config.games_per_update, 512)
        self.assertEqual(config.trajectory_games_per_update, 512)
        self.assertEqual(config.optimizer_steps_per_update, 32)
        self.assertEqual(config.eval_every, 5)
        self.assertEqual(config.optimization_mode, "fixed_optimizer_budget")
        self.assertEqual(config.ppo.actor_learning_rate, 1.0e-5)
        self.assertEqual(config.ppo.value_learning_rate, 1.0e-4)
        self.assertEqual(config.ppo.prize_learning_rate, 1.0e-4)

    def test_optimizer_base_rates_are_not_accelerated(self) -> None:
        self.assertEqual(PPO.PPOConfig().actor_learning_rate, 1.0e-5)
        self.assertEqual(PPO.PPOConfig().value_learning_rate, 1.0e-4)
        self.assertEqual(PPO.PPOConfig().prize_learning_rate, 1.0e-4)
        self.assertEqual(RUNNER.AdaptationConfig().adapter_learning_rate, 3.0e-5)

    def test_frozen_contract_is_seeded_agent_choice_v3(self) -> None:
        self.assertEqual(
            FROZEN.CANONICAL_CONTRACT_ID,
            "frozen_0806_seeded_agent_first_player_v3",
        )
        jobs, schedule = FROZEN.build_frozen_jobs(
            focal_deck_id=RUNNER.FOCAL_DECK_ID,
            focal_deck=RUNNER.focal_deck(),
            runtime_root=RUNNER.runtime_root(),
            source_policy_update=5,
        )
        self.assertEqual(len(jobs), 2048)
        self.assertEqual(schedule, FROZEN.EXPECTED_007_SCHEDULE_SHA256)
        self.assertTrue(all(job.focal_won_toss is not None for job in jobs))
        self.assertEqual(len({job.seed for job in jobs}), 2048)


if __name__ == "__main__":
    unittest.main()
