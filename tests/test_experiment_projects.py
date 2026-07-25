from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rl_environment.runs import (
    initialize_project,
    initialize_version,
    project_version_paths,
    record_evaluation,
    refresh_experiment_index,
)


class ExperimentProjectPathTests(unittest.TestCase):
    def test_project_version_paths_nest_all_runtime_assets_under_one_project(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
                paths = project_version_paths(
                    "0013_alakazam_rollout_value_calibration",
                    "V1_initial_contract",
                )

            self.assertEqual(
                paths.artifact,
                repository
                / "rl_runs/0013_alakazam_rollout_value_calibration/versions/V1_initial_contract/artifact",
            )
            self.assertEqual(
                paths.checkpoints,
                repository
                / "rl_runs/0013_alakazam_rollout_value_calibration/versions/V1_initial_contract/checkpoint",
            )
            self.assertEqual(
                paths.evaluation,
                repository
                / "experiments/0013_alakazam_rollout_value_calibration/evaluation/V1_initial_contract.html",
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
    def test_initialize_version_creates_all_runtime_paths_once(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            project_id = "0013_alakazam_rollout"
            (repository / "experiments" / project_id).mkdir(parents=True)
            (repository / "experiments" / project_id / "manifest.json").write_text("{}\n")
            (repository / "rl_runs" / project_id / "versions").mkdir(parents=True)
            with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
                paths = initialize_version(project_id, "V1_initial_contract")
                self.assertTrue(paths.artifact.is_dir())
                self.assertTrue(paths.checkpoints.is_dir())
                self.assertTrue(paths.tensorboard.is_dir())
                self.assertTrue(paths.wandb.is_dir())
                self.assertEqual(json.loads(paths.status.read_text())["state"], "allocated")
                with self.assertRaises(FileExistsError):
                    initialize_version(project_id, "V1_initial_contract")

    def test_record_evaluation_requires_existing_report_and_writes_backlink(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            project_id = "0013_alakazam_rollout"
            archive = repository / "experiments" / project_id / "evaluation"
            archive.mkdir(parents=True)
            (repository / "experiments" / project_id / "manifest.json").write_text("{}\n")
            (repository / "rl_runs" / project_id / "versions").mkdir(parents=True)
            with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
                paths = initialize_version(project_id, "V1_initial_contract")
                with self.assertRaises(FileNotFoundError):
                    record_evaluation(paths, run_id="run-a", report_sha256="abc")
                paths.evaluation.write_text("<html></html>\n", encoding="utf-8")
                record_evaluation(paths, run_id="run-a", report_sha256="abc")

            record = json.loads((paths.artifact / "evaluation.json").read_text())
            self.assertEqual(record["run_id"], "run-a")
            self.assertEqual(record["report"], "experiments/0013_alakazam_rollout/evaluation/V1_initial_contract.html")


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
