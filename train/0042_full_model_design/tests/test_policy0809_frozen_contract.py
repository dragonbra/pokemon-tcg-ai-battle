from __future__ import annotations

import importlib
import json
from pathlib import Path
import unittest


PROJECT = "train.0042_full_model_design"
ROOT = Path(__file__).resolve().parents[3]
EXPECTED_OPPONENT_EFFECTIVE = (
    "0d0091140d72e78f1070c549b8367583a9d4f5537d0cb67decab40ac3bb9da96"
)


class Policy0809FrozenContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = importlib.import_module(f"{PROJECT}.training.run_full_semantic")
        cls.frozen = importlib.import_module(f"{PROJECT}.evaluation.frozen_jobs")
        cls.identity = importlib.import_module(f"{PROJECT}.policy_identity")

    def test_formal_runtime_is_unbounded_007_and_evaluates_every_ten_updates(self) -> None:
        config = self.runner.RunConfig()
        config.validate()
        self.assertIsNone(config.updates)
        self.assertEqual(config.eval_every, 10)
        self.assertEqual(config.games_per_update, 256)
        self.assertEqual(config.trajectory_games_per_update, 256)
        self.assertEqual(self.runner.FOCAL_DECK_PATH.parent.name, "007_dragapult_ex")
        self.assertEqual(
            self.runner.exact_deck_sha256(self.runner.focal_deck()),
            self.runner.FOCAL_EXACT_DECK_SHA256,
        )

    def test_active_registry_contains_only_complete_policy0809(self) -> None:
        registry = self.identity.load_policy_registry()
        self.assertEqual(set(registry), {"Policy-0809"})
        entry = registry["Policy-0809"]
        self.assertEqual(entry["effective_policy_sha256"], EXPECTED_OPPONENT_EFFECTIVE)
        self.assertEqual(set(entry["components"]), set(self.identity.EFFECTIVE_COMPONENTS))
        self.assertTrue(all(
            row["source_policy_id"] == "Policy-0809"
            for row in entry["components"].values()
        ))

    def test_frozen_schedule_is_candidate_identity_bound(self) -> None:
        common = dict(
            focal_deck_id=self.runner.FOCAL_DECK_ID,
            focal_deck=self.runner.focal_deck(),
            runtime_root=self.runner.runtime_root(),
            source_policy_update=10,
            opponent_effective_policy_sha256=EXPECTED_OPPONENT_EFFECTIVE,
        )
        jobs_a, schedule_a = self.frozen.build_frozen_jobs(
            **common, focal_deployment_identity="a" * 64
        )
        jobs_b, schedule_b = self.frozen.build_frozen_jobs(
            **common, focal_deployment_identity="b" * 64
        )
        self.assertNotEqual(schedule_a, schedule_b)
        self.assertTrue(set(job.seed for job in jobs_a).isdisjoint(
            set(job.seed for job in jobs_b)
        ))
        self.assertEqual({job.opponent_policy_id for job in jobs_a}, {"Policy-0809"})

    def test_formal_active_files_do_not_reference_policy0806(self) -> None:
        active = (
            ROOT / "train/0042_full_model_design/training/run_full_semantic.py",
            ROOT / "train/0042_full_model_design/evaluation/frozen_jobs.py",
            ROOT / "train/0042_full_model_design/export_full_semantic_candidate.py",
            ROOT / "train/0042_full_model_design/candidate_deployment.py",
            ROOT / "train/0042_full_model_design/policy_registry.json",
            ROOT / "train/0042_full_model_design/rollout/cuda_collector.py",
        )
        forbidden = ("Policy-0806", "Frozen-0806", "frozen_0806")
        hits = {
            str(path.relative_to(ROOT)): token
            for path in active
            for token in forbidden
            if token in path.read_text(encoding="utf-8")
        }
        self.assertEqual(hits, {})

    def test_stale_seeded_evaluator_and_renderer_are_fail_closed(self) -> None:
        stale = (
            ROOT / "train/0042_full_model_design/evaluate_seeded_checkpoint_series.py",
            ROOT / "train/0042_full_model_design/render_frozen_checkpoint_report.py",
        )
        for path in stale:
            source = path.read_text(encoding="utf-8")
            self.assertIn("FATAL: legacy", source)
            self.assertIn("disabled", source)

    def test_cpu256_and_cuda2048_result_kinds_cannot_be_mislabeled(self) -> None:
        with self.assertRaisesRegex(ValueError, "kind.*game count"):
            self.runner._persist_frozen_results(
                Path("unused.json"),
                [],
                checkpoint_update=0,
                panel_version="unused",
                schedule_sha256="a" * 64,
                opponent_identity_audit=None,
                candidate_deployment_audit=None,
                expected_games=256,
                benchmark_kind="cuda_2048",
            )

    def test_all_fifty_five_decks_are_numbered_and_policy_neutral(self) -> None:
        catalog = json.loads(
            (ROOT / "train/0042_full_model_design/league/frozen_catalog.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(catalog["pool_id"], "0042_policy_0809_neutral_55_v1")
        roots = sorted(
            path for path in (
                ROOT / "train/0042_full_model_design/league/decks"
            ).iterdir() if path.is_dir()
        )
        self.assertEqual(len(roots), 55)
        self.assertEqual(
            [path.name[:3] for path in roots],
            [f"{index:03d}" for index in range(1, 56)],
        )


if __name__ == "__main__":
    unittest.main()
