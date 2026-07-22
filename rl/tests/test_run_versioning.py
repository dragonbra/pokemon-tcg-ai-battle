from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rl.core.runs import training_paths


class RunVersioningTests(unittest.TestCase):
    def test_versioned_attempt_uses_isolated_run_tensorboard_and_checkpoint_paths(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runs_root = root / "rl" / "_runs"
            artifact_root = root / "rl" / "artifact"
            experiment = runs_root / "0003-example"
            experiment.mkdir(parents=True)
            with (
                patch("rl.core.runs.RUNS_ROOT", runs_root),
                patch("rl.core.runs.ARTIFACT_ROOT", artifact_root),
            ):
                paths = training_paths(experiment / "V1_initial_contract")

            self.assertEqual(paths.run, experiment / "V1_initial_contract")
            self.assertEqual(
                paths.tensorboard,
                runs_root / "tensorboard" / experiment.name / "V1_initial_contract",
            )
            self.assertEqual(
                paths.checkpoints,
                artifact_root / "checkpoint" / experiment.name / "V1_initial_contract",
            )

    def test_experiment_root_and_reused_version_fail_closed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runs_root = root / "rl" / "_runs"
            artifact_root = root / "rl" / "artifact"
            experiment = runs_root / "0003-example"
            attempt = experiment / "V1_initial_contract"
            attempt.mkdir(parents=True)
            (attempt / "training_config.json").write_text("{}\n", encoding="utf-8")
            with (
                patch("rl.core.runs.RUNS_ROOT", runs_root),
                patch("rl.core.runs.ARTIFACT_ROOT", artifact_root),
            ):
                with self.assertRaises(ValueError):
                    training_paths(experiment)
                with self.assertRaises(FileExistsError):
                    training_paths(attempt)


if __name__ == "__main__":
    unittest.main()
