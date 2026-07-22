from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluation import cli
from evaluation.packages.loader import PackageValidationError, SubmissionPackage


def package(name: str, root: Path) -> SubmissionPackage:
    return SubmissionPackage(
        name=name,
        root=root,
        deck=[1] * 60,
        entrypoint=root / "main.py",
        package_hash=f"{name}-package-hash",
        deck_hash=f"{name}-deck-hash",
        cg_manifest={"tree_hash": "shared-cg-hash"},
    )


class EvaluationCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.candidate = package("candidate", self.root / "candidate")
        self.control = package("control", self.root / "control")
        self.opponent_a = package("opponent_a", self.root / "opponent_a")
        self.opponent_b = package("opponent_b", self.root / "opponent_b")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_run_help_exposes_metric_profile_selection(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit):
            cli.main(["run", "--help"])

        self.assertIn("--metric-profile", output.getvalue())
        self.assertIn("auto_iteration_v8_setup_relay", output.getvalue())

    def test_validate_prints_standard_package_fingerprints(self) -> None:
        output = io.StringIO()
        with (
            patch.object(cli, "_load_official_card_ids", return_value={1}),
            patch.object(cli, "load_submission_package", return_value=self.candidate),
            redirect_stdout(output),
        ):
            exit_code = cli.main(["validate", str(self.candidate.root)])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "name: candidate",
                "deck_hash: candidate-deck-hash",
                "cg_tree_hash: shared-cg-hash",
                "60-card valid: true",
            ],
        )

    def test_run_delegates_validated_packages_to_batch_with_control(self) -> None:
        output = io.StringIO()
        result = SimpleNamespace(run_id="run-cli-test")
        opponents = tuple(
            package(f"opponent_{index}", self.root / f"opponent_{index}")
            for index in range(18)
        )
        with (
            patch.object(cli, "_load_official_card_ids", return_value={1}),
            patch.object(
                cli,
                "load_submission_package",
                side_effect=[self.candidate, self.control],
            ),
            patch.object(
                cli,
                "load_opponent_catalog",
                return_value=list(opponents),
            ),
            patch.object(cli, "run_batch", return_value=result, create=True) as run_batch,
            redirect_stdout(output),
        ):
            exit_code = cli.main(
                [
                    "run",
                    "--candidate",
                    str(self.candidate.root),
                    "--control",
                    str(self.control.root),
                    "--opponents",
                    "all",
                    "--games",
                    "10",
                    "--output",
                    str(self.root / "reports"),
                    "--no-visualize",
                    "--keep-temp",
                    "--max-steps",
                    "77",
                    "--metric-profile",
                    "auto_iteration_v8_setup_relay",
                    "--metric-module",
                    "metrics/custom.py:CustomPlugin",
                ]
            )

        self.assertEqual(exit_code, 0)
        config = run_batch.call_args.args[0]
        self.assertEqual(config.candidate, self.candidate)
        self.assertEqual(config.opponents, opponents)
        self.assertEqual(config.control, self.control)
        self.assertEqual(config.games_per_opponent, 10)
        self.assertEqual(config.output_root, self.root / "reports")
        self.assertFalse(config.visualize)
        self.assertTrue(config.keep_temp)
        self.assertEqual(config.max_steps, 77)
        self.assertEqual(config.metric_profile_id, "auto_iteration_v8_setup_relay")
        self.assertEqual(config.metric_module_paths, ("metrics/custom.py:CustomPlugin",))
        self.assertIn("run-cli-test", output.getvalue())
        self.assertNotIn("promotion", output.getvalue().lower())
        self.assertNotIn("reject", output.getvalue().lower())

    def test_run_stops_before_batch_when_candidate_validation_fails(self) -> None:
        stderr = io.StringIO()
        with (
            patch.object(cli, "_load_official_card_ids", return_value={1}),
            patch.object(
                cli,
                "load_submission_package",
                side_effect=PackageValidationError("deck.csv must contain 60 cards"),
            ),
            patch.object(cli, "run_batch", create=True) as run_batch,
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as error,
        ):
            cli.main(
                [
                    "run",
                    "--candidate",
                    str(self.candidate.root),
                    "--opponents",
                    "all",
                    "--output",
                    str(self.root / "reports"),
                ]
            )

        self.assertNotEqual(error.exception.code, 0)
        self.assertIn("deck.csv must contain 60 cards", stderr.getvalue())
        run_batch.assert_not_called()

if __name__ == "__main__":
    unittest.main()
