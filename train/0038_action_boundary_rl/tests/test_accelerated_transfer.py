from __future__ import annotations

import importlib
import unittest


module = importlib.import_module(
    "train.0038_action_boundary_rl.training.accelerated_transfer"
)


class AcceleratedTransferTest(unittest.TestCase):
    def test_schedule_and_group_ownership(self):
        controller = module.AcceleratedTransferController()
        base = {
            "action_decoder": 1e-5,
            "allocation_head": 1e-5,
            "last_option_qv_lora": 3e-5,
            "value_win": 1e-4,
            "value_prize": 1e-4,
        }
        self.assertAlmostEqual(controller.learning_rates(base, update=1)["action_decoder"], 3e-5)
        self.assertAlmostEqual(controller.learning_rates(base, update=3)["action_decoder"], 5e-5)
        self.assertAlmostEqual(controller.learning_rates(base, update=6)["action_decoder"], 1e-4)
        self.assertAlmostEqual(controller.learning_rates(base, update=6)["last_option_qv_lora"], 3e-4)
        self.assertAlmostEqual(controller.learning_rates(base, update=6)["value_win"], 2e-4)
        with self.assertRaises(ValueError):
            controller.learning_rates({"opponent_meta": 1e-4}, update=1)

    def test_health_thresholds_and_cap_reduction(self):
        controller = module.AcceleratedTransferController()
        warning = controller.health({"ppo/behavior_kl": 0.006, "ppo/clip_fraction": 0.01})
        self.assertTrue(warning.warning)
        self.assertFalse(warning.reduce_actor_lr)
        reduce = controller.health({"ppo/behavior_kl": 0.011, "ppo/clip_fraction": 0.01})
        self.assertTrue(reduce.reduce_actor_lr)
        self.assertFalse(reduce.rollback)
        rollback = controller.health({
            "ppo/behavior_kl": 0.003,
            "ppo/rejected_behavior_kl": 0.021,
            "ppo/target_kl_early_stop": 1.0,
        })
        self.assertTrue(rollback.rollback)
        self.assertEqual(controller.reduce_actor_cap(), 5.0)
        self.assertEqual(controller.actor_multiplier(20), 5.0)
        self.assertEqual(controller.reduce_actor_cap(), 3.0)
        with self.assertRaises(RuntimeError):
            controller.reduce_actor_cap()

    def test_formal_acceptance_is_unbounded_cuda_prize_only(self):
        runner = importlib.import_module(
            "train.0038_action_boundary_rl.training.run_full_semantic"
        )
        config = runner.RunConfig(
            version="V999_accelerated_transfer_test",
            preset_name="PRIZE",
            accelerated_transfer_acceptance=True,
        )
        config.validate()
        with self.assertRaises(ValueError):
            runner.RunConfig(
                version="V999_accelerated_transfer_test",
                updates=50,
                preset_name="PRIZE",
                accelerated_transfer_acceptance=True,
            ).validate()
        with self.assertRaises(ValueError):
            runner.RunConfig(
                version="V999_accelerated_transfer_test",
                preset_name="INTEGRATED",
                accelerated_transfer_acceptance=True,
            ).validate()


if __name__ == "__main__":
    unittest.main()
