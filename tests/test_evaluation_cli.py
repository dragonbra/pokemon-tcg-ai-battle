from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluation import cli
from evaluation.frozen_0806_contract import (
    FROZEN_0806_EVALUATION_SEED,
    evaluation_counts,
    evaluation_schedule_id,
)
from evaluation.frozen_0806_runtime import load_frozen_0806_runtime_catalog
from evaluation.frozen import FrozenCatalog
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

    def test_run_defaults_to_fast_report_only_execution_inputs(self) -> None:
        args = cli._parser().parse_args(
            [
                "run",
                "--candidate",
                str(self.candidate.root),
                "--opponents",
                "all",
                "--output",
                str(Path.cwd() / ".tmp" / "evaluation" / "cli-test"),
            ]
        )

        self.assertEqual(args.games, 10)
        self.assertFalse(args.visualize)
        self.assertIsNone(args.workers)
        self.assertEqual(args.worker_cpu_threads, 1)
        self.assertEqual(args.worker_timeout_seconds, 30.0)
        self.assertEqual(args.engine_pool_size, 1)

    def test_frozen_run_uses_seeded_2048_evaluation_contract(self) -> None:
        catalog = load_frozen_0806_runtime_catalog(opponent_policy_label="0806")
        result = SimpleNamespace(run_id="run-frozen-seeded-2048")
        args = cli._parser().parse_args(
            [
                "run",
                "--candidate",
                str(catalog.candidates[0].root),
                "--opponents",
                "all",
                "--output",
                str(Path.cwd() / ".tmp" / "evaluation" / "cli-frozen-2048-test"),
            ]
        )
        with (
            patch.object(cli, "_load_official_card_ids", return_value=set(catalog.candidates[0].deck)),
            patch.object(cli, "load_submission_package", return_value=catalog.candidates[0]),
            patch.object(
                cli, "load_frozen_0806_runtime_catalog", return_value=catalog
            ) as load_frozen,
            patch.object(cli, "assert_cg_compatible"),
            patch.object(cli, "run_batch", return_value=result) as run_batch,
        ):
            self.assertEqual(cli._run(args), result.run_id)

        config = run_batch.call_args.args[0]
        load_frozen.assert_called_once_with(
            cli.DEFAULT_FROZEN_CATALOG,
            cli.DEFAULT_FROZEN_CATALOG.resolve().parent.parent,
            opponent_policy_label="0806",
        )
        expected_counts = evaluation_counts(catalog.pool.schedule)
        self.assertEqual(config.games_by_opponent, expected_counts)
        self.assertEqual(sum(config.games_by_opponent or ()), 2048)
        self.assertTrue(config.independent_engine_seeds)
        self.assertEqual(config.seed, FROZEN_0806_EVALUATION_SEED)
        self.assertTrue(config.seeded_engine)
        self.assertEqual(
            config.opponent_policy_hash,
            catalog.pool.policies["main"]["weights_sha256"],
        )
        self.assertEqual(config.opponent_policy_label, "Policy-0806")
        self.assertEqual(
            config.opponent_schedule_id,
            evaluation_schedule_id(catalog.pool.manifest["schedule_sha256"]),
        )

    def test_frozen_exact_deck_decorates_live_candidate_with_canonical_identity(self) -> None:
        live = package("0022_dragapult_update75", self.root / "live")
        live = SubmissionPackage(
            **{
                **live.__dict__,
                "package_manifest": {"deck_id": "dragapult_ex_001", "update": 75},
            }
        )
        identity = package("dragapult_ex_001", self.root / "frozen")
        identity = SubmissionPackage(
            **{
                **identity.__dict__,
                "display_name": "[Frozen 0019] Dragapult ex - Limitless",
                "representative_cards": ({"card_id": 1, "name": "Dragapult ex"},),
            }
        )
        catalog = FrozenCatalog(
            pool_id="fixture",
            manifest={},
            manifest_sha256="catalog-sha",
            policy=identity,
            opponents=(identity,),
        )

        decorated = cli._decorate_candidate_from_frozen_identity(live, catalog)

        self.assertEqual(decorated.name, "dragapult_ex_001")
        self.assertEqual(decorated.display_name, "Dragapult ex - Limitless")
        self.assertEqual(decorated.package_manifest["update"], 75)
        self.assertEqual(decorated.root, live.root)

    def test_formal_output_is_one_versioned_html_in_project_evaluation_directory(self) -> None:
        requested = Path(
            "experiments/0013_alakazam_sota_feature_engineering/evaluation/"
            "V10_action_contract_fix"
        )

        output_root, report_path = cli._evaluation_output_paths(requested)

        expected = (Path.cwd() / requested).resolve().with_suffix(".html")
        self.assertEqual(report_path, expected)
        self.assertEqual(output_root, expected.parent)

    def test_formal_output_rejects_legacy_rl_runs_evaluation_path(self) -> None:
        requested = Path("rl_runs/evaluation/0012-alakazam_sota_feature_engineering/V1_legacy")

        with self.assertRaisesRegex(PackageValidationError, "legacy"):
            cli._evaluation_output_paths(requested)

    def test_temporary_output_retains_run_directory_layout(self) -> None:
        requested = Path(".tmp/evaluation/path-smoke")

        output_root, report_path = cli._evaluation_output_paths(requested)

        self.assertEqual(output_root, requested)
        self.assertIsNone(report_path)

    def test_output_rejects_arbitrary_directory_and_explicit_flat_report(self) -> None:
        for requested in (
            Path("reports/path-smoke"),
            Path(".tmp/evaluation/path-smoke/report.html"),
        ):
            with self.subTest(requested=requested):
                with self.assertRaisesRegex(PackageValidationError, "temporary evaluation output"):
                    cli._evaluation_output_paths(requested)

    def test_formal_output_rejects_uppercase_or_tagless_version_before_workers(self) -> None:
        for version_name in ("V1", "v1_lowercase", "V1_Uppercase", "V1_two__underscores"):
            requested = Path("experiments/0013_alakazam/evaluation") / version_name
            with self.assertRaisesRegex(PackageValidationError, "formal evaluation report"):
                cli._evaluation_output_paths(requested)

    def test_formal_output_rejects_missing_version_name(self) -> None:
        requested = Path("experiments/0013_alakazam_sota_feature_engineering/evaluation")

        with self.assertRaisesRegex(PackageValidationError, "formal evaluation output"):
            cli._evaluation_output_paths(requested)

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
                    "--pool",
                    "opponents",
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
                    str(Path.cwd() / ".tmp" / "evaluation" / "cli-test"),
                    "--no-visualize",
                    "--keep-temp",
                    "--max-steps",
                    "77",
                    "--workers",
                    "4",
                    "--worker-cpu-threads",
                    "1",
                    "--worker-timeout-seconds",
                    "120",
                    "--engine-pool-size",
                    "2",
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
        self.assertEqual(config.output_root, Path.cwd() / ".tmp" / "evaluation" / "cli-test")
        self.assertFalse(config.visualize)
        self.assertTrue(config.keep_temp)
        self.assertEqual(config.max_steps, 77)
        self.assertEqual(config.engine_pool_size, 2)
        self.assertEqual(config.workers, 4)
        self.assertEqual(config.worker_cpu_threads, 1)
        self.assertEqual(config.worker_timeout_seconds, 120.0)
        self.assertEqual(config.metric_profile_id, "auto_iteration_v8_setup_relay")
        self.assertEqual(config.metric_module_paths, ("metrics/custom.py:CustomPlugin",))
        self.assertIn("run-cli-test", output.getvalue())
        self.assertNotIn("promotion", output.getvalue().lower())
        self.assertNotIn("reject", output.getvalue().lower())

    def test_run_resolves_automatic_worker_default_before_batch(self) -> None:
        result = SimpleNamespace(run_id="run-default-workers")
        output = io.StringIO()
        opponents = tuple(
            package(f"opponent_{index}", self.root / f"opponent_{index}")
            for index in range(18)
        )
        with (
            patch.object(cli, "_load_official_card_ids", return_value={1}),
            patch.object(cli, "load_submission_package", return_value=self.candidate),
            patch.object(cli, "load_opponent_catalog", return_value=list(opponents)),
            patch.object(cli, "default_worker_count", return_value=7),
            patch.object(cli, "run_batch", return_value=result) as run_batch,
            redirect_stdout(output),
        ):
            exit_code = cli.main(
                [
                    "--pool",
                    "opponents",
                    "run",
                    "--candidate",
                    str(self.candidate.root),
                    "--opponents",
                    "all",
                    "--output",
                    str(Path.cwd() / ".tmp" / "evaluation" / "cli-test"),
                ]
            )

        self.assertEqual(exit_code, 0)
        config = run_batch.call_args.args[0]
        self.assertEqual(config.games_per_opponent, 10)
        self.assertEqual(config.workers, 7)
        self.assertEqual(config.worker_cpu_threads, 1)
        self.assertFalse(config.visualize)

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
                    str(Path.cwd() / ".tmp" / "evaluation" / "cli-test"),
                ]
            )

        self.assertNotEqual(error.exception.code, 0)
        self.assertIn("deck.csv must contain 60 cards", stderr.getvalue())
        run_batch.assert_not_called()

if __name__ == "__main__":
    unittest.main()
