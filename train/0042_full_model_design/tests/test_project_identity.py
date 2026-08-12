from __future__ import annotations

import importlib
from collections import Counter
import json
from pathlib import Path
import re
import unittest
actor_critic = importlib.import_module(
    "train.0042_full_model_design.policy.actor_critic"
)
runtime = importlib.import_module(
    "train.0042_full_model_design.training.runtime"
)
transfer = importlib.import_module("train.0042_full_model_design.transfer")
exporter = importlib.import_module(
    "train.0042_full_model_design.export_zero_shot_candidate"
)
full_runner = importlib.import_module(
    "train.0042_full_model_design.training.run_full_semantic"
)
league = importlib.import_module("train.0042_full_model_design.league")
own_archetype = importlib.import_module(
    "train.0042_full_model_design.policy.own_archetype"
)


class ProjectIdentityTest(unittest.TestCase):
    def test_exact_focal_deck_identity(self) -> None:
        deck = runtime.read_deck(runtime.FOCAL_DECK_PATH)
        self.assertEqual(len(deck), 60)
        self.assertEqual(
            runtime.deck_multiset_sha256(deck), runtime.FOCAL_DECK_MULTISET_SHA256
        )
        manifest = json.loads(runtime.FOCAL_DECK_PATH.with_name("manifest.json").read_text())
        self.assertEqual(manifest["deck_id"], runtime.FOCAL_DECK_ID)
        self.assertEqual(manifest["exact_deck_sha256"], runtime.FOCAL_DECK_MULTISET_SHA256)

    def test_policy_0809_v5_epoch20_source_identity(self) -> None:
        self.assertEqual(transfer.SOURCE_EPOCH, 20)
        self.assertEqual(transfer.SOURCE_GLOBAL_STEP, 115820)
        self.assertEqual(
            transfer.SOURCE_VERSION,
            "V5_medal_zone_gsb_all_days",
        )
        self.assertEqual(
            transfer.SOURCE_CHECKPOINT_SHA256,
            "926321955b6f3144b62e65899b5041ca3a47b17f305202ba9dffc0c92aaa7c7f",
        )

    def test_no_executable_cross_project_imports(self) -> None:
        project = Path("train/0042_full_model_design")
        forbidden = re.compile(r"(?:from|import)\s+train\.00(?:0[0-9]|[12][0-9]|3[0-6])[_\w.]*")
        violations = []
        for path in project.rglob("*.py"):
            if match := forbidden.search(path.read_text(encoding="utf-8")):
                violations.append(f"{path}: {match.group(0)}")
        self.assertEqual(violations, [])

    def test_exporter_normalizes_relative_repository_path(self) -> None:
        relative = Path(
            "archive/pretrained/0031_friend_0809_gsb_v5_value_v9/model.pt"
        )
        self.assertEqual(exporter._repo_relative(relative), relative)

    def test_wandb_display_name_keeps_project_prefix(self) -> None:
        self.assertTrue(full_runner.WANDB_DISPLAY_PREFIX.startswith("0042"))

    def test_selected_0036_value_checkpoint_identity(self) -> None:
        self.assertEqual(
            actor_critic.EXPECTED_VALUE_SHA256,
            "f8cb92f45f625e6519b6f103deb6d4ef68323fca93037ebd4c25db9da122a486",
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

    def test_full_semantic_custom_focal_exact_deck_is_explicit_and_routed(self) -> None:
        config = full_runner.RunConfig(
            version="V6_dragapult_ex_0042_v6",
            focal_deck_path=(
                "train/0042_full_model_design/focal_decks/"
                "dragapult_ex_0042_v6/deck.csv"
            ),
            focal_deck_id="dragapult_ex_0042_v6",
            focal_exact_deck_sha256=(
                "7bdb3bb183008d9204efc68ad77b6df1039ad760c98476aa82d5809f5c446ca3"
            ),
            focal_deck_display_name="Dragapult ex 0042 V6",
            focal_deck_source="user_supplied_2026-08-12",
        )
        config.validate()
        deck = full_runner.focal_deck(config)
        jobs = full_runner.build_jobs(
            source_policy_update=0,
            seed=123,
            count=256,
            focal_cards=deck,
        )

        self.assertEqual(len(deck), 60)
        self.assertEqual(full_runner.exact_deck_sha256(deck), config.focal_exact_deck_sha256)
        self.assertEqual({job.focal_deck for job in jobs}, {deck})
        self.assertNotEqual(deck, full_runner.focal_deck())

    def test_v7_023_grass_focal_identity_and_archetype_are_explicit(self) -> None:
        config = full_runner.RunConfig(
            version="V7_hydrapple_ex_meganium_023",
            focal_deck_path=(
                "train/0042_full_model_design/focal_decks/"
                "hydrapple_ex_meganium_023/deck.csv"
            ),
            focal_deck_id="hydrapple_ex_meganium_415af0a5541c",
            focal_exact_deck_sha256=(
                "415af0a5541c9c046b4a80abc365dbf91fc693b8546e49796fec6463a181c53d"
            ),
            focal_deck_display_name="023 · Hydrapple ex / Meganium",
            focal_deck_source="0042_policy_0809_neutral_55_v1_catalog_023",
        )
        config.validate()
        deck = full_runner.focal_deck(config)
        jobs = full_runner.build_jobs(
            source_policy_update=0,
            seed=123,
            count=256,
            focal_cards=deck,
        )
        archetype = own_archetype.OwnArchetypeVocabulary.load().classify_own_deck(
            deck
        )

        self.assertEqual(len(deck), 60)
        self.assertEqual(full_runner.exact_deck_sha256(deck), config.focal_exact_deck_sha256)
        self.assertEqual({job.focal_deck for job in jobs}, {deck})
        self.assertEqual(archetype.value, 6)
        self.assertEqual(
            own_archetype.OwnArchetypeVocabulary.load().classes[archetype.value].name,
            "festival_lead",
        )

    def test_v8_002_alakazam_focal_identity_and_archetype_are_explicit(self) -> None:
        config = full_runner.RunConfig(
            version="V8_alakazam_dudunsparce_002",
            focal_deck_path=(
                "train/0042_full_model_design/focal_decks/"
                "alakazam_dudunsparce_002/deck.csv"
            ),
            focal_deck_id="alakazam_dudunsparce_3f4515092dc5",
            focal_exact_deck_sha256=(
                "3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf"
            ),
            focal_deck_display_name="002 · Alakazam / Dudunsparce",
            focal_deck_source="0042_policy_0809_neutral_55_v1_catalog_002",
            initial_model_checkpoint=(
                "rl_runs/0042_full_model_design/versions/"
                "V7_hydrapple_ex_meganium_023/checkpoint/update-000270.pt"
            ),
        )
        config.validate()
        deck = full_runner.focal_deck(config)
        jobs = full_runner.build_jobs(
            source_policy_update=0,
            seed=123,
            count=256,
            focal_cards=deck,
        )
        vocabulary = own_archetype.OwnArchetypeVocabulary.load()
        archetype = vocabulary.classify_own_deck(deck)

        self.assertEqual(len(deck), 60)
        self.assertEqual(
            full_runner.exact_deck_sha256(deck),
            config.focal_exact_deck_sha256,
        )
        self.assertEqual({job.focal_deck for job in jobs}, {deck})
        self.assertEqual(archetype.value, 3)
        self.assertEqual(vocabulary.classes[archetype.value].name, "alakazam")

    def test_v10_001_grimmsnarl_focal_identity_and_u90_lineage_are_explicit(self) -> None:
        config = full_runner.RunConfig(
            version="V10_marnies_grimmsnarl_ex_froslass_001",
            focal_deck_path=(
                "train/0042_full_model_design/focal_decks/"
                "marnies_grimmsnarl_ex_froslass_001/deck.csv"
            ),
            focal_deck_id="marnie_s_grimmsnarl_ex_froslass_c20a8a46f5c6",
            focal_exact_deck_sha256=(
                "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
            ),
            focal_deck_display_name="001 · Marnie's Grimmsnarl ex / Froslass",
            focal_deck_source="0042_policy_0809_neutral_55_v1_catalog_001",
            initial_model_checkpoint=(
                "rl_runs/0042_full_model_design/versions/"
                "V9_alakazam_dudunsparce_002_fp16_drift_allowed/"
                "checkpoint/update-000090.pt"
            ),
            allow_fp16_deployment_numeric_drift=True,
        )
        config.validate()
        deck = full_runner.focal_deck(config)
        jobs = full_runner.build_jobs(
            source_policy_update=0,
            seed=123,
            count=256,
            focal_cards=deck,
        )
        vocabulary = own_archetype.OwnArchetypeVocabulary.load()
        archetype = vocabulary.classify_own_deck(deck)

        self.assertEqual(len(deck), 60)
        self.assertEqual(
            full_runner.exact_deck_sha256(deck),
            config.focal_exact_deck_sha256,
        )
        self.assertEqual({job.focal_deck for job in jobs}, {deck})
        self.assertEqual(archetype.value, 2)
        self.assertEqual(
            vocabulary.classes[archetype.value].name,
            "marnies_grimmsnarl_ex",
        )
        checkpoint = Path(config.initial_model_checkpoint)
        self.assertEqual(
            full_runner._sha256(checkpoint),
            "b9d754214deb679c9ab0fb45c35ee39dfee39692b8aef73f88b06c4c1f894776",
        )

    def test_formal_run_uses_0042_strategy_identity(self) -> None:
        config = full_runner.RunConfig()

        self.assertEqual(config.ppo.credit_clock, "turn")
        self.assertEqual(config.ppo.gae_lambda, 0.95)
        self.assertEqual(config.ppo.batch_size, 2048)
        self.assertEqual(config.ppo.epochs, 3)
        self.assertEqual(
            config.version, "V1_ppo_protocol_v2_baseline"
        )
        self.assertEqual(config.games_per_update, 256)
        self.assertEqual(config.trajectory_games_per_update, 256)
        self.assertIsNone(config.updates)
        self.assertEqual(config.eval_every, 10)
        self.assertEqual(config.adaptation_arm, "strategy")
        self.assertEqual(config.preset_name, "FULL_MODEL")
        self.assertEqual(config.worker_processes, 16)
        self.assertEqual(config.engines_per_worker, 8)
        self.assertEqual(config.inference_channels_per_role, 8)

    def test_formal_rollout_uses_fifty_full_round_draw_limit(self) -> None:
        jobs = full_runner.build_jobs(source_policy_update=0, seed=123, count=256)

        self.assertEqual({job.full_round_draw_limit for job in jobs}, {50})

    def test_formal_ppo_is_blocked_before_manual_update0_approval(self) -> None:
        self.assertEqual(
            full_runner.ALLOCATION_HEAD_CHECKPOINT.parent,
            full_runner.ROOT / "train/0042_full_model_design/assets",
        )
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
