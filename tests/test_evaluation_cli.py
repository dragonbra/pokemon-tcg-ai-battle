from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluation import cli
from evaluation.packages.loader import PackageValidationError, SubmissionPackage
from scripts import alakazam_auto_iter


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
                return_value=[self.opponent_a, self.opponent_b],
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
                    "opponent_b,opponent_a",
                    "--games",
                    "4",
                    "--output",
                    str(self.root / "reports"),
                    "--no-visualize",
                    "--keep-temp",
                    "--max-steps",
                    "77",
                    "--metric-module",
                    "metrics/custom.py:CustomPlugin",
                ]
            )

        self.assertEqual(exit_code, 0)
        config = run_batch.call_args.args[0]
        self.assertEqual(config.candidate, self.candidate)
        self.assertEqual(config.opponents, (self.opponent_b, self.opponent_a))
        self.assertEqual(config.control, self.control)
        self.assertEqual(config.games_per_opponent, 4)
        self.assertEqual(config.output_root, self.root / "reports")
        self.assertFalse(config.visualize)
        self.assertTrue(config.keep_temp)
        self.assertEqual(config.max_steps, 77)
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

    def test_legacy_agent_path_resolves_to_standard_package_root(self) -> None:
        package_root = self.root / "legacy_candidate"
        package_root.mkdir()
        (package_root / "main.py").write_text("def agent(observation): return []\n", encoding="utf-8")
        (package_root / "deck.csv").write_text("1\n", encoding="utf-8")

        self.assertEqual(
            alakazam_auto_iter.legacy_candidate_root(package_root / "main.py"),
            package_root.resolve(),
        )
        with self.assertRaisesRegex(ValueError, "standard submission package"):
            alakazam_auto_iter.legacy_candidate_root(self.root / "not_a_package")

    def test_legacy_analyze_reads_retained_native_evaluation_trace(self) -> None:
        report_root = self.root / "native_report"
        trace_root = report_root / "traces"
        trace_root.mkdir(parents=True)
        (trace_root / "opponent_a-001.json").write_text(
            json.dumps(
                {
                    "trace": [],
                    "result": {"winner": 0, "error_kind": None},
                    "opponent": "opponent_a",
                }
            ),
            encoding="utf-8",
        )

        result = alakazam_auto_iter.analyze_report(report_root)

        self.assertEqual(result.metrics.games, 1)
        self.assertEqual(result.metrics.wins, 1)
        self.assertEqual(result.source_files, ("traces/opponent_a-001.json",))

    def test_legacy_compare_reads_native_summary_and_metric_artifacts(self) -> None:
        control = self.root / "control"
        candidate = self.root / "candidate_report"
        comparison_dir = self.root / "comparison"
        for directory, wins in ((control, 2), (candidate, 3)):
            directory.mkdir()
            (directory / "summary.json").write_text(
                json.dumps(
                    {
                        "total_games": 4,
                        "wins": wins,
                        "losses": 4 - wins,
                        "draws": 0,
                        "errors": 0,
                    }
                ),
                encoding="utf-8",
            )
            (directory / "metrics.json").write_text(
                json.dumps(
                    {
                        "powerful_hand": {"numerator": 1, "denominator": 4},
                        "post_ko_relay": {"numerator": 0, "denominator": 0},
                        "run_away_draw": {"numerator": 0, "denominator": 4},
                    }
                ),
                encoding="utf-8",
            )
            (directory / "cases.jsonl").write_text("", encoding="utf-8")

        comparison = alakazam_auto_iter.compare_reports(control, candidate, comparison_dir)

        self.assertEqual(comparison["control"]["games"], 4)
        self.assertEqual(comparison["candidate"]["wins"], 3)
        self.assertTrue((comparison_dir / "comparison.json").is_file())

    def test_legacy_run_delegates_without_using_evaluator_root(self) -> None:
        package_root = self.root / "legacy_run_candidate"
        (package_root / "cg").mkdir(parents=True)
        (package_root / "main.py").write_text("def agent(observation): return []\n", encoding="utf-8")
        (package_root / "deck.csv").write_text("1\n", encoding="utf-8")
        stderr = io.StringIO()
        with (
            patch.object(alakazam_auto_iter, "evaluation_cli_main", return_value=0) as cli_main,
            redirect_stderr(stderr),
        ):
            exit_code = alakazam_auto_iter.main(
                [
                    "run",
                    "--evaluator-root",
                    str(self.root / "unavailable-adjacent-repo"),
                    "--agent",
                    str(package_root / "main.py"),
                    "--cg-path",
                    str(package_root / "cg"),
                    "--label",
                    "legacy_candidate",
                    "--opponents",
                    "opponent_a",
                    "--games",
                    "2",
                    "--output-dir",
                    str(self.root / "reports"),
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            cli_main.call_args.args[0],
            [
                "run",
                "--candidate",
                str(package_root.resolve()),
                "--opponents",
                "opponent_a",
                "--games",
                "2",
                "--output",
                str(self.root / "reports"),
            ],
        )
        self.assertIn("deprecated and ignored", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
