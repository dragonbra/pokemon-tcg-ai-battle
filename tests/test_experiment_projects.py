from __future__ import annotations

import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rl_environment.runs import (
    initialize_project,
    initialize_version,
    project_version_paths,
    record_evaluation,
    refresh_experiment_index,
    numbered_artifact_path,
    training_paths,
    write_version_status,
)


class ExperimentProjectPathTests(unittest.TestCase):
    def test_project_version_paths_nest_all_runtime_assets_under_one_project(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
                paths = project_version_paths(
                    "0013_semantic_goal_policy",
                    "V1_initial_contract",
                )

            self.assertEqual(
                paths.artifact,
                repository
                / "rl_runs/0013_semantic_goal_policy/versions/V1_initial_contract/artifact",
            )
            self.assertEqual(
                paths.checkpoints,
                repository
                / "rl_runs/0013_semantic_goal_policy/versions/V1_initial_contract/checkpoint",
            )
            self.assertEqual(
                paths.evaluation,
                repository
                / "experiments/0013_semantic_goal_policy/evaluation/V1_initial_contract.html",
            )
            self.assertEqual(paths.checkpoint_selection, paths.artifact / "checkpoint_selection.json")
            self.assertEqual(paths.model_contract, paths.artifact / "model_contract.json")
            self.assertEqual(paths.dataset_reference, paths.artifact / "dataset_reference.json")
            self.assertEqual(paths.metrics_snapshot, paths.artifact / "metrics_snapshot.json")
            self.assertEqual(
                paths.wandb_snapshot_manifest,
                paths.artifact / "wandb_snapshot_manifest.json",
            )

    def test_project_version_paths_reject_hyphenated_project_slug(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid project ID"):
            project_version_paths("0013-alakazam", "V1_initial_contract")


class ExperimentProjectScaffoldTests(unittest.TestCase):
    def test_initialize_project_creates_three_project_roots_and_manifest(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            with patch("rl_environment.runs.REPOSITORY_ROOT", repository), patch(
                "rl_environment.runs._git_value", return_value="abc123"
            ):
                paths = initialize_project(
                    "alakazam_rollout_value_calibration",
                    objective="Calibrate value targets from official-engine rollouts",
                    deck="alakazam_dudunsparce",
                    expert_source="team_policy_2026_07",
                    dataset_contract="single_team_episode_v1",
                    engine_revision="official-engine-2026-07-26",
                    opponent_pool_snapshot="catalog-sha256:abc",
                )

            self.assertEqual(paths.project_id, "0013_alakazam_rollout_value_calibration")
            self.assertTrue(paths.train.is_dir())
            self.assertTrue(paths.archive.is_dir())
            self.assertTrue(paths.runs.is_dir())
            manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["deck"], "alakazam_dudunsparce")
            self.assertEqual(
                manifest["paths"]["training"],
                "train/0013_alakazam_rollout_value_calibration",
            )

    def test_initialize_project_rejects_missing_contract_field(self) -> None:
        with TemporaryDirectory() as directory:
            with patch("rl_environment.runs.REPOSITORY_ROOT", Path(directory)):
                with self.assertRaisesRegex(ValueError, "expert_source must not be empty"):
                    initialize_project(
                        "alakazam_rollout",
                        objective="value calibration",
                        deck="alakazam",
                        expert_source="",
                        dataset_contract="dataset_v1",
                        engine_revision="official",
                        opponent_pool_snapshot="snapshot_v1",
                    )


class ExperimentProjectVersionTests(unittest.TestCase):
    PROJECT_ID = "0013_semantic_goal_policy"

    def _create_project_roots(self, repository: Path) -> None:
        (repository / "experiments" / self.PROJECT_ID).mkdir(parents=True)
        (repository / "experiments" / self.PROJECT_ID / "manifest.json").write_text("{}\n")
        (repository / "rl_runs" / self.PROJECT_ID / "versions").mkdir(parents=True)

    def test_initialize_version_creates_all_runtime_paths_once(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            self._create_project_roots(repository)
            with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
                paths = initialize_version(self.PROJECT_ID, "V1_initial_contract")
                self.assertTrue(paths.artifact.is_dir())
                self.assertTrue(paths.checkpoints.is_dir())
                self.assertTrue(paths.tensorboard.is_dir())
                self.assertTrue(paths.wandb.is_dir())
                self.assertEqual(json.loads(paths.status.read_text())["state"], "allocated")
                with self.assertRaises(FileExistsError):
                    initialize_version(self.PROJECT_ID, "V1_initial_contract")

    def test_initialize_version_rejects_every_occupied_runtime_target(self) -> None:
        for target in ("artifact", "checkpoint", "tensorboard", "wandb", "report"):
            with self.subTest(target=target), TemporaryDirectory() as directory:
                repository = Path(directory)
                self._create_project_roots(repository)
                with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
                    paths = project_version_paths(self.PROJECT_ID, "V1_initial_contract")
                    occupied = paths.evaluation if target == "report" else getattr(
                        paths,
                        {
                            "artifact": "artifact",
                            "checkpoint": "checkpoints",
                            "tensorboard": "tensorboard",
                            "wandb": "wandb",
                        }[target],
                    )
                    occupied.parent.mkdir(parents=True, exist_ok=True)
                    occupied.write_text("occupied\n", encoding="utf-8")
                    with self.assertRaisesRegex(FileExistsError, "version assets already exist"):
                        initialize_version(self.PROJECT_ID, "V1_initial_contract")

    def test_initialize_version_requires_the_strict_next_version_across_all_runtime_roots(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            self._create_project_roots(repository)
            version_root = repository / "rl_runs" / self.PROJECT_ID / "versions"
            for directory_name in ("artifact", "checkpoint", "tensorboard", "wandb"):
                (version_root / "V2_runtime_assets" / directory_name).mkdir(parents=True)
            with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
                with self.assertRaisesRegex(ValueError, "version must be V3_<tag>"):
                    initialize_version(self.PROJECT_ID, "V1_initial_contract")

    def test_write_version_status_merges_atomically_without_replacing_existing_fields(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            self._create_project_roots(repository)
            with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
                paths = initialize_version(self.PROJECT_ID, "V1_initial_contract")
                write_version_status(paths, {"state": "training", "wandb": {"sync": "pending"}})

            self.assertEqual(
                json.loads(paths.status.read_text(encoding="utf-8")),
                {
                    "state": "training",
                    "version": "V1_initial_contract",
                    "wandb": {"sync": "pending"},
                },
            )
            self.assertFalse(paths.status.with_name("status.json.tmp").exists())

    def test_write_version_status_leaves_original_unchanged_when_atomic_replacement_fails(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            self._create_project_roots(repository)
            with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
                paths = initialize_version(self.PROJECT_ID, "V1_initial_contract")
                original = paths.status.read_text(encoding="utf-8")
                with patch.object(Path, "replace", side_effect=OSError("disk failure")):
                    with self.assertRaisesRegex(OSError, "disk failure"):
                        write_version_status(paths, {"state": "training"})

            self.assertEqual(paths.status.read_text(encoding="utf-8"), original)
            self.assertFalse(paths.status.with_name("status.json.tmp").exists())

    def test_record_evaluation_requires_existing_report_and_writes_immutable_backlink(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            self._create_project_roots(repository)
            with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
                paths = initialize_version(self.PROJECT_ID, "V1_initial_contract")
                with self.assertRaises(FileNotFoundError):
                    record_evaluation(paths, run_id="run-a", report_sha256="abc")
                paths.evaluation.parent.mkdir(parents=True, exist_ok=True)
                paths.evaluation.write_text("<html></html>\n", encoding="utf-8")
                record_evaluation(paths, run_id="run-a", report_sha256="abc")
                with self.assertRaisesRegex(FileExistsError, "evaluation provenance already exists"):
                    record_evaluation(paths, run_id="run-b", report_sha256="def")

            record = json.loads((paths.artifact / "evaluation.json").read_text())
            self.assertEqual(record["run_id"], "run-a")
            self.assertEqual(
                record["report"],
                "experiments/0013_semantic_goal_policy/evaluation/V1_initial_contract.html",
            )


class LegacyExperimentCompatibilityTests(unittest.TestCase):
    def test_legacy_0001_through_0012_paths_remain_readable(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            with patch("rl_environment.runs.REPOSITORY_ROOT", repository), patch(
                "rl_environment.runs.RUNS_ROOT", repository / "rl_runs"
            ), patch("rl_environment.runs.EXPERIMENT_ROOT", repository / "rl_runs/artifact"), patch(
                "rl_environment.runs.CHECKPOINT_ROOT", repository / "rl_runs/checkpoint"
            ):
                for number in range(1, 13):
                    with self.subTest(number=number):
                        experiment_id = f"{number:04d}-legacy_policy_{number}"
                        legacy_run = repository / "rl_runs/artifact" / experiment_id / "V1_initial_contract"
                        paths = training_paths(legacy_run)
                        self.assertEqual(paths.run, legacy_run)
                        self.assertEqual(
                            paths.checkpoints,
                            repository / "rl_runs/checkpoint" / experiment_id / "V1_initial_contract",
                        )
                        self.assertEqual(
                            paths.tensorboard,
                            repository / "rl_runs/tensorboard" / experiment_id / "V1_initial_contract",
                        )
                        self.assertEqual(
                            numbered_artifact_path(
                                repository / "rl_runs" / experiment_id / "evaluation", "evaluation"
                            ),
                            repository / "rl_runs/evaluation" / experiment_id,
                        )


class ExperimentArchiveTests(unittest.TestCase):
    def test_initialized_project_has_navigation_and_design_templates(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
                paths = initialize_project(
                    "alakazam_value",
                    objective="value calibration",
                    deck="alakazam",
                    expert_source="team_policy",
                    dataset_contract="dataset_v1",
                    engine_revision="official",
                    opponent_pool_snapshot="catalog_v1",
                )

            self.assertTrue((paths.archive / "README.md").is_file())
            self.assertTrue((paths.archive / "DESIGN.md").is_file())
            self.assertTrue((paths.archive / "DESIGN.html").is_file())
            self.assertTrue((paths.archive / "evaluation/index.html").is_file())

    def test_refresh_experiment_index_derives_legacy_identity_without_mutation(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            archive = repository / "experiments" / "0012_alakazam_sota_feature_engineering"
            archive.mkdir(parents=True)
            legacy_manifest = {"objective": "legacy feature engineering", "status": "complete"}
            manifest_path = archive / "manifest.json"
            manifest_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")
            with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
                refresh_experiment_index()

            index = (repository / "experiments" / "INDEX.md").read_text(encoding="utf-8")
            self.assertIn("0012_alakazam_sota_feature_engineering", index)
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8")), legacy_manifest)

    def test_refresh_experiment_index_rejects_new_manifest_without_identity(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            archive = repository / "experiments" / "0013_semantic_goal_policy"
            archive.mkdir(parents=True)
            (archive / "manifest.json").write_text(
                json.dumps({"objective": "new project"}), encoding="utf-8"
            )
            with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
                with self.assertRaisesRegex(ValueError, "missing project_id"):
                    refresh_experiment_index()

    def test_refresh_experiment_index_rejects_new_manifest_identity_mismatch(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            archive = repository / "experiments" / "0013_semantic_goal_policy"
            archive.mkdir(parents=True)
            (archive / "manifest.json").write_text(
                json.dumps({"project_id": "0013_wrong_identity"}), encoding="utf-8"
            )
            with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
                with self.assertRaisesRegex(ValueError, "does not match directory"):
                    refresh_experiment_index()

    def test_refresh_experiment_index_resolves_legacy_and_current_evaluation_links(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            experiments = repository / "experiments"
            experiments.mkdir()
            for number in range(10, 13):
                project_id = f"{number:04d}_legacy_policy"
                archive = experiments / project_id
                archive.mkdir()
                (archive / "README.md").write_text("archive\n", encoding="utf-8")
                (archive / "DESIGN.html").write_text("design\n", encoding="utf-8")
                (archive / "manifest.json").write_text(
                    json.dumps({"objective": "legacy", "status": "complete"}),
                    encoding="utf-8",
                )
                legacy_id = f"{number:04d}-legacy_policy"
                evaluation = repository / "rl_runs" / "evaluation" / legacy_id
                evaluation.mkdir(parents=True)
                (evaluation / "index.html").write_text("evaluation\n", encoding="utf-8")
            current = experiments / "0013_semantic_goal_policy"
            (current / "evaluation").mkdir(parents=True)
            (current / "README.md").write_text("archive\n", encoding="utf-8")
            (current / "DESIGN.html").write_text("design\n", encoding="utf-8")
            (current / "evaluation" / "index.html").write_text("evaluation\n", encoding="utf-8")
            (current / "manifest.json").write_text(
                json.dumps({"project_id": current.name, "objective": "current"}),
                encoding="utf-8",
            )

            with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
                refresh_experiment_index()

            for index_name in ("INDEX.md", "INDEX.html"):
                text = (experiments / index_name).read_text(encoding="utf-8")
                links = re.findall(r'href=["\']([^"\']+)', text) if index_name.endswith("html") else re.findall(r"\[[^]]+\]\(([^)]+)\)", text)
                for link in links:
                    if link.startswith("#"):
                        continue
                    self.assertTrue((experiments / link).resolve().exists(), (index_name, link))
            markdown = (experiments / "INDEX.md").read_text(encoding="utf-8")
            self.assertIn("../rl_runs/evaluation/0010-legacy_policy/index.html", markdown)
            self.assertIn("0013_semantic_goal_policy/evaluation/index.html", markdown)

    def test_refresh_experiment_index_preserves_curated_html_shell(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            experiments = repository / "experiments"
            archive = experiments / "0013_semantic_goal_policy"
            (archive / "evaluation").mkdir(parents=True)
            for relative in ("README.md", "DESIGN.html", "evaluation/index.html"):
                (archive / relative).write_text("target\n", encoding="utf-8")
            (archive / "manifest.json").write_text(
                json.dumps({"project_id": archive.name, "objective": "current"}),
                encoding="utf-8",
            )
            (experiments / "INDEX.html").write_text(
                '<!doctype html><html><body><div class="topbar">CURATED</div>'
                '<section class="projects"><article>OLD</article></section>'
                '<aside class="note">KEEP</aside></body></html>',
                encoding="utf-8",
            )
            with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
                refresh_experiment_index()

            html = (experiments / "INDEX.html").read_text(encoding="utf-8")
            self.assertIn('class="topbar"', html)
            self.assertIn('class="note"', html)
            self.assertIn("CURATED", html)
            self.assertIn("0013_semantic_goal_policy", html)
            self.assertNotIn("OLD", html)

    def test_refresh_experiment_index_links_project_design_and_evaluation(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            archive = repository / "experiments" / "0013_alakazam_value"
            archive.mkdir(parents=True)
            (archive / "manifest.json").write_text(
                json.dumps({"project_id": "0013_alakazam_value", "objective": "value calibration", "status": "initialized"}),
                encoding="utf-8",
            )
            with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
                refresh_experiment_index()

            index = (repository / "experiments" / "INDEX.md").read_text(encoding="utf-8")
            self.assertIn("0013_alakazam_value/DESIGN.html", index)
            self.assertIn("0013_alakazam_value/evaluation/index.html", index)


class ExperimentProjectLifecycleTests(unittest.TestCase):
    def test_new_project_lifecycle_is_navigable_and_immutable(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            with patch("rl_environment.runs.REPOSITORY_ROOT", repository), patch(
                "rl_environment.runs._git_value", return_value="abc123"
            ):
                project = initialize_project(
                    "alakazam_rollout",
                    objective="value calibration",
                    deck="alakazam",
                    expert_source="team_policy",
                    dataset_contract="dataset_v1",
                    engine_revision="official",
                    opponent_pool_snapshot="catalog_v1",
                )
                version = initialize_version(project.project_id, "V1_initial_contract")
                version.evaluation.write_text("<html>report</html>\n", encoding="utf-8")
                record_evaluation(version, run_id="engine-run-123", report_sha256="deadbeef")
                refresh_experiment_index()

            self.assertTrue((project.archive / "README.md").is_file())
            self.assertTrue((project.archive / "DESIGN.html").is_file())
            self.assertTrue((project.archive / "evaluation" / "index.html").is_file())
            self.assertTrue((version.artifact / "evaluation.json").is_file())
            self.assertIn(project.project_id, (repository / "experiments" / "INDEX.md").read_text())
            with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
                with self.assertRaises(FileExistsError):
                    initialize_version(project.project_id, "V1_initial_contract")


if __name__ == "__main__":
    unittest.main()
