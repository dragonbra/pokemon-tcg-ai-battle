from __future__ import annotations

import importlib
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace


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

    def test_attested_package_gate_uses_exact_immutable_283_trace(self):
        runner = importlib.import_module(
            "train.0038_action_boundary_rl.training.run_full_semantic"
        )
        trace = (
            Path(runner.ROOT)
            / ".tmp/evaluation/0038_semantic_parity_audit/gate_c_final/"
            "fixed_283_primitive_trace.jsonl"
        )
        self.assertTrue(trace.is_file())
        self.assertEqual(sum(bool(line.strip()) for line in trace.read_text().splitlines()), 283)
        self.assertEqual(
            hashlib.sha256(trace.read_bytes()).hexdigest(),
            runner.IMMUTABLE_GATE_C_TRACE_SHA256,
        )

    def test_attested_package_gate_reads_root_divergence_from_report_schema(self):
        runner = importlib.import_module(
            "train.0038_action_boundary_rl.training.run_full_semantic"
        )
        report = {
            "full_cpu_causalknowledge_parity": True,
            "training_to_package": {
                "passed": True,
                "root": {"greedy_action_divergences": 0},
            },
        }
        self.assertTrue(runner._attested_package_parity_passed(report))
        report["training_to_package"]["root"]["greedy_action_divergences"] = 1
        self.assertFalse(runner._attested_package_parity_passed(report))

    def test_chance_boundary_episode_is_replaced_in_the_same_environment_slot(self):
        runner = importlib.import_module(
            "train.0038_action_boundary_rl.training.run_full_semantic"
        )
        protocol = importlib.import_module(
            "train.0038_action_boundary_rl.rollout.protocol"
        )
        with tempfile.TemporaryDirectory() as temporary:
            job = protocol.RolloutJob(
                game_id="rollout-u0000-0103",
                opponent_id="barbaracle",
                focal_first=True,
                seed=11,
                source_policy_update=0,
                focal_deck=(1,) * 60,
                opponent_deck=(2,) * 60,
                runtime_root=Path(temporary),
                policy_seed=12,
                search_seed=13,
                action_boundary_mode="enabled",
            )
            episode = protocol.EpisodeTrajectory(job)
            episode.finish(-1.0, 16)
            episode.diagnostics = {
                "macro_fallback": 1,
                "macro_fallback_reason": "chance_boundary_before_allocation",
            }

            def collect(replacements):
                self.assertEqual(len(replacements), 1)
                replacement = replacements[0]
                self.assertEqual(replacement.opponent_id, job.opponent_id)
                self.assertEqual(replacement.focal_first, job.focal_first)
                self.assertNotEqual(replacement.seed, job.seed)
                clean = protocol.EpisodeTrajectory(replacement)
                clean.finish(1.0, 12)
                clean.diagnostics = {"macro_fallback": 0}
                return [clean]

            accepted, evidence = runner._replace_chance_boundary_episodes(
                [episode], collect
            )
            self.assertEqual(len(accepted), 1)
            self.assertEqual(accepted[0].job.opponent_id, job.opponent_id)
            self.assertEqual(accepted[0].job.focal_first, job.focal_first)
            self.assertEqual(len(evidence), 1)
            self.assertEqual(evidence[0]["reason"], "chance_boundary_before_allocation")

    def test_nonchance_macro_invalid_still_fails_closed(self):
        runner = importlib.import_module(
            "train.0038_action_boundary_rl.training.run_full_semantic"
        )
        protocol = importlib.import_module(
            "train.0038_action_boundary_rl.rollout.protocol"
        )
        with tempfile.TemporaryDirectory() as temporary:
            job = protocol.RolloutJob(
                game_id="drift",
                opponent_id="opponent",
                focal_first=False,
                seed=1,
                source_policy_update=0,
                focal_deck=(1,) * 60,
                opponent_deck=(2,) * 60,
                runtime_root=Path(temporary),
                action_boundary_mode="enabled",
            )
            episode = protocol.EpisodeTrajectory(job)
            episode.finish(-1.0, 4)
            episode.diagnostics = {
                "macro_fallback": 1,
                "macro_fallback_reason": "stable_target_identity_drift",
            }
            with self.assertRaisesRegex(RuntimeError, "non-resampleable macro failure"):
                runner._replace_chance_boundary_episodes([episode], lambda _: [])

    def test_frozen_health_retains_legal_chance_boundary(self):
        runner = importlib.import_module(
            "train.0038_action_boundary_rl.training.run_full_semantic"
        )
        episode = SimpleNamespace(diagnostics={
            "macro_fallback": 1,
            "macro_fallback_reason": "chance_boundary_before_allocation",
        })
        metrics = runner._assert_acceptance_episode_health(
            [episode], {"rollout/invalid_macros": 1.0},
            scope="eval/core", allow_chance_boundary=True,
        )
        self.assertEqual(metrics["eval/core/chance_boundaries"], 1.0)
        self.assertEqual(metrics["eval/core/fallback"], 0.0)

    def test_frozen_health_still_rejects_nonchance_fallback(self):
        runner = importlib.import_module(
            "train.0038_action_boundary_rl.training.run_full_semantic"
        )
        episode = SimpleNamespace(diagnostics={
            "macro_fallback": 1,
            "macro_fallback_reason": "stable_target_identity_drift",
        })
        with self.assertRaisesRegex(RuntimeError, "stable_target_identity_drift"):
            runner._assert_acceptance_episode_health(
                [episode], {"rollout/invalid_macros": 1.0},
                scope="eval/core", allow_chance_boundary=True,
            )

    def test_rollout_health_does_not_accept_unreplaced_chance_boundary(self):
        runner = importlib.import_module(
            "train.0038_action_boundary_rl.training.run_full_semantic"
        )
        episode = SimpleNamespace(diagnostics={
            "macro_fallback": 1,
            "macro_fallback_reason": "chance_boundary_before_allocation",
        })
        with self.assertRaisesRegex(RuntimeError, "chance_boundary_before_allocation"):
            runner._assert_acceptance_episode_health(
                [episode], {"rollout/invalid_macros": 1.0}, scope="rollout"
            )

    def test_frozen_persistence_keeps_seed_and_chance_reason(self):
        runner = importlib.import_module(
            "train.0038_action_boundary_rl.training.run_full_semantic"
        )
        episodes = []
        for seed in range(2048):
            reason = "chance_boundary_before_allocation" if seed == 17 else None
            episodes.append(SimpleNamespace(
                reward=1.0,
                turns=4,
                valid=True,
                error=None,
                job=SimpleNamespace(
                    game_id=f"game-{seed}", seed=seed, opponent_id="opponent",
                    focal_first=bool(seed % 2),
                ),
                diagnostics={
                    "macro_fallback": int(reason is not None),
                    "macro_fallback_reason": reason,
                },
            ))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "frozen.json"
            outcomes = runner._persist_frozen_results(
                output, episodes, checkpoint_update=5, panel_version="fixed"
            )
            payload = json.loads(output.read_text())
        self.assertEqual(len(outcomes), 2048)
        self.assertEqual(payload["entries"][17]["seed"], 17)
        self.assertTrue(payload["entries"][17]["chance_boundary"])
        self.assertFalse(payload["entries"][17]["semantic_fallback"])
        self.assertEqual(
            payload["entries"][17]["fallback_reason"],
            "chance_boundary_before_allocation",
        )

    def test_frozen_runtime_metrics_do_not_overwrite_rollout_namespace(self):
        runner = importlib.import_module(
            "train.0038_action_boundary_rl.training.run_full_semantic"
        )
        mapped = runner._evaluation_runtime_metrics({
            "rollout/invalid_macros": 1.0,
            "rollout/cuda_games_per_second": 12.0,
        })
        self.assertEqual(mapped["eval/runtime/invalid_macros"], 1.0)
        self.assertNotIn("rollout/invalid_macros", mapped)


if __name__ == "__main__":
    unittest.main()
