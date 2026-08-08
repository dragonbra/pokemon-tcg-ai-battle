from __future__ import annotations

import importlib
from collections import Counter
import json
from pathlib import Path
import re
import unittest
actor_critic = importlib.import_module(
    "train.0038_action_boundary_rl.policy.actor_critic"
)
runtime = importlib.import_module(
    "train.0038_action_boundary_rl.training.runtime"
)
transfer = importlib.import_module("train.0038_action_boundary_rl.transfer")
exporter = importlib.import_module(
    "train.0038_action_boundary_rl.export_zero_shot_candidate"
)
full_runner = importlib.import_module(
    "train.0038_action_boundary_rl.training.run_full_semantic"
)
league = importlib.import_module("train.0038_action_boundary_rl.league")


class ProjectIdentityTest(unittest.TestCase):
    def test_exact_focal_deck_identity(self) -> None:
        deck = runtime.read_deck(runtime.FOCAL_DECK_PATH)
        self.assertEqual(len(deck), 60)
        self.assertEqual(
            runtime.deck_multiset_sha256(deck), runtime.FOCAL_DECK_MULTISET_SHA256
        )
        manifest = json.loads(runtime.FOCAL_DECK_PATH.with_name("manifest.json").read_text())
        self.assertEqual(manifest["deck_id"], runtime.FOCAL_DECK_ID)
        self.assertEqual(
            manifest["deck_file_sha256"], runtime.FOCAL_DECK_FILE_SHA256
        )
        self.assertEqual(
            manifest["deck_multiset_sha256"], runtime.FOCAL_DECK_MULTISET_SHA256
        )

    def test_large_model_0806_epoch11_source_identity(self) -> None:
        self.assertEqual(transfer.SOURCE_EPOCH, 11)
        self.assertEqual(transfer.SOURCE_GLOBAL_STEP, 141878)
        self.assertEqual(
            transfer.SOURCE_VERSION,
            "V2_full_winners_bs1024_20260616_20260803",
        )
        self.assertEqual(
            transfer.SOURCE_CHECKPOINT_SHA256,
            "0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8",
        )

    def test_no_executable_cross_project_imports(self) -> None:
        project = Path("train/0038_action_boundary_rl")
        forbidden = re.compile(r"(?:from|import)\s+train\.00(?:0[0-9]|[12][0-9]|3[0-6])[_\w.]*")
        violations = []
        for path in project.rglob("*.py"):
            if match := forbidden.search(path.read_text(encoding="utf-8")):
                violations.append(f"{path}: {match.group(0)}")
        self.assertEqual(violations, [])

    def test_exporter_normalizes_relative_repository_path(self) -> None:
        relative = Path(
            "archive/pretrained/0031_friend_0806_epoch11_best_validation_loss/model.pt"
        )
        self.assertEqual(exporter._repo_relative(relative), relative)

    def test_wandb_display_name_keeps_project_prefix(self) -> None:
        self.assertTrue(full_runner.WANDB_DISPLAY_PREFIX.startswith("0038"))

    def test_selected_0036_value_checkpoint_identity(self) -> None:
        self.assertEqual(
            actor_critic.EXPECTED_VALUE_SHA256,
            "e88b2f18089911c4ebd810fa3abfba800a103189184b6265b9d38c03a9e36360",
        )
        self.assertEqual(
            full_runner._sha256(actor_critic.DEFAULT_VALUE_CHECKPOINT),
            actor_critic.EXPECTED_VALUE_SHA256,
        )

    def test_frozen_0806_catalog_is_exact_fixed_256_distribution(self) -> None:
        catalog = league.load_frozen_catalog()

        self.assertEqual(len(catalog), 55)
        self.assertEqual(len({item.deck_sha256 for item in catalog}), 55)
        self.assertEqual(sum(item.games for item in catalog), 256)
        self.assertTrue(all(len(item.deck) == 60 for item in catalog))

    def test_formal_rollout_preserves_each_256_slot_frequency_unit(self) -> None:
        jobs = full_runner.build_jobs(source_policy_update=0, seed=123, count=2048)
        expected = Counter({item.deck_id: item.games for item in league.load_frozen_catalog()})

        self.assertEqual(len(jobs), 2048)
        self.assertEqual(sum(job.focal_first for job in jobs), 1024)
        self.assertEqual(len({job.seed for job in jobs}), 2048)
        for start in range(0, len(jobs), 256):
            unit = jobs[start : start + 256]
            self.assertEqual(Counter(job.opponent_id for job in unit), expected)
        replay = full_runner.build_jobs(source_policy_update=0, seed=123, count=2048)
        self.assertEqual(
            [(j.opponent_id, j.focal_first, j.seed, j.search_seed, j.policy_seed) for j in jobs],
            [(j.opponent_id, j.focal_first, j.seed, j.search_seed, j.policy_seed) for j in replay],
        )

    def test_formal_rollout_rejects_partial_frequency_unit(self) -> None:
        with self.assertRaisesRegex(ValueError, "256"):
            full_runner.build_jobs(source_policy_update=0, seed=123, count=512 + 2)

    def test_full_semantic_focal_deck_is_exact_frozen_007(self) -> None:
        frozen_007 = next(
            item for item in league.load_frozen_catalog()
            if item.deck_id == "dragapult_ex_07bedfffbfad"
        )

        self.assertEqual(full_runner.FOCAL_DECK_ID, frozen_007.deck_id)
        self.assertEqual(full_runner.focal_deck(), frozen_007.deck)
        self.assertEqual(
            full_runner.FOCAL_EXACT_DECK_SHA256,
            "07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725",
        )

    def test_formal_run_uses_0038_zero_shot_identity(self) -> None:
        config = full_runner.RunConfig()

        self.assertEqual(config.games_per_update, 2048)
        self.assertEqual(config.ppo.credit_clock, "turn")
        self.assertEqual(config.ppo.gae_lambda, 0.95)
        self.assertEqual(config.ppo.batch_size, 1024)
        self.assertEqual(config.ppo.epochs, 4)
        self.assertEqual(config.version, "V6_canonical_frozen_cuda_fresh_rl")
        self.assertIsNone(config.updates)
        self.assertEqual(config.eval_every, 5)
        self.assertEqual(config.adaptation_arm, "lora")
        self.assertEqual(config.preset_name, "INTEGRATED")
        self.assertEqual(config.worker_processes, 16)
        self.assertEqual(config.engines_per_worker, 8)
        self.assertEqual(config.inference_channels_per_role, 8)

    def test_formal_ppo_is_blocked_before_manual_update0_approval(self) -> None:
        self.assertIn("V3_update0_chance_boundary_fallback", str(full_runner.COMMON_UPDATE0_CHECKPOINT))
        with self.assertRaisesRegex(RuntimeError, "explicit user approval"):
            full_runner.run(full_runner.RunConfig(updates=1))

    def test_sparse_diagnostic_predicate_is_bound_in_formal_runner(self) -> None:
        self.assertTrue(full_runner.is_sparse_diagnostic_update(5))

    def test_rollout_topology_rejects_invalid_channel_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "inference_channels_per_role"):
            full_runner.RunConfig(
                engines_per_worker=4,
                inference_channels_per_role=5,
            ).validate()

    def test_greedy_probe_seed_schedule_is_checkpoint_independent(self) -> None:
        update_0 = full_runner.build_jobs(
            source_policy_update=0, seed=123, count=512, greedy=True
        )
        update_10 = full_runner.build_jobs(
            source_policy_update=10, seed=123, count=512, greedy=True
        )
        sampled_0 = full_runner.build_jobs(source_policy_update=0, seed=123, count=512)
        sampled_10 = full_runner.build_jobs(source_policy_update=10, seed=123, count=512)

        paired_0 = [
            (job.seed, job.focal_first, job.opponent_id, job.opponent_deck)
            for job in update_0
        ]
        paired_10 = [
            (job.seed, job.focal_first, job.opponent_id, job.opponent_deck)
            for job in update_10
        ]
        self.assertEqual(paired_0, paired_10)
        self.assertNotEqual(
            [job.seed for job in sampled_0],
            [job.seed for job in sampled_10],
        )


if __name__ == "__main__":
    unittest.main()
