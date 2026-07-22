from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path
import tempfile
from unittest.mock import patch

from arena.cli import _load_arena_packages, build_parser, main, run_command, validate_command
from arena.models import DeckRecord, RunRecord
from arena.storage import ArenaStore
from evaluation.packages.loader import SubmissionPackage


class ArenaCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_parser_exposes_all_commands_and_defaults_smoke_to_ten_games(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--phase", "smoke"])

        self.assertEqual(args.phase, "smoke")
        self.assertEqual(args.games, 10)
        subparsers = next(action for action in parser._actions if action.dest == "command")
        self.assertEqual(
            set(subparsers.choices),
            {"collect", "validate", "run", "report", "discussions"},
        )

    def test_report_command_does_not_require_kaggle_credentials(self) -> None:
        with patch("arena.cli.write_report_command") as command:
            self.assertEqual(main(["report", "--run-id", "run-1"]), 0)
            command.assert_called_once()

    def test_package_loader_excludes_metadata_only_and_invalid_catalog_rows(self) -> None:
        package_root = self.repo_root / "arena" / "packages"
        package_root.mkdir(parents=True)
        for name in ("valid", "metadata-only", "invalid"):
            (package_root / name).mkdir()

        store = ArenaStore(self.repo_root / "arena")
        store.initialize()
        for name, status in (("valid", "valid"), ("metadata-only", "metadata_only"), ("invalid", "invalid")):
            store.upsert_deck(
                DeckRecord(
                    name,
                    name,
                    name,
                    "Unknown Archetype",
                    (),
                    f"arena/packages/{name}",
                    None,
                    "public",
                    status,
                    {},
                )
            )

        def fake_load(root: Path, _: set[int]) -> SubmissionPackage:
            return SubmissionPackage(root.name, root, [1] * 60, root / "main.py", "pkg", "deck", {})

        with patch("arena.cli._official_card_ids", return_value={1}), patch(
            "arena.cli.load_submission_package", side_effect=fake_load
        ) as loader:
            packages = _load_arena_packages(self.repo_root)

        self.assertEqual(set(packages), {"valid"})
        self.assertEqual([call.args[0].name for call in loader.call_args_list], ["valid"])

    def test_validate_does_not_revalidate_metadata_only_package(self) -> None:
        package_root = self.repo_root / "arena" / "packages"
        package_root.mkdir(parents=True)
        for name in ("valid", "metadata-only"):
            (package_root / name).mkdir()

        store = ArenaStore(self.repo_root / "arena")
        store.initialize()
        for name, status in (("valid", "valid"), ("metadata-only", "metadata_only")):
            store.upsert_deck(
                DeckRecord(
                    name,
                    name,
                    name,
                    "Unknown Archetype",
                    (),
                    f"arena/packages/{name}",
                    None,
                    "public",
                    status,
                    {},
                )
            )

        def fake_load(root: Path, _: set[int]) -> SubmissionPackage:
            return SubmissionPackage(root.name, root, [1] * 60, root / "main.py", "pkg", "deck", {})

        args = Namespace(repo_root=self.repo_root)
        with patch("arena.cli._official_card_ids", return_value={1}), patch(
            "arena.cli.load_submission_package", side_effect=fake_load
        ) as loader, patch("arena.cli._write_source_catalog"):
            validate_command(args)

        self.assertEqual([call.args[0].name for call in loader.call_args_list], ["valid"])

    def test_continuous_resume_can_continue_a_completed_smoke_run(self) -> None:
        store = ArenaStore(self.repo_root / "arena")
        store.initialize()
        store.create_run(RunRecord("smoke-1", "smoke", {"seed": 7}, "now", "later", "finished"))
        args = Namespace(
            repo_root=self.repo_root,
            phase="continuous",
            games=20,
            limit=1,
            resume="smoke-1",
            include_demoted=False,
            seed=7,
            max_steps=10,
            timeout=1.0,
            checkpoint_every=1,
        )

        class FakeRunner:
            last_instance: "FakeRunner | None" = None

            def __init__(self, *_: object, **__: object) -> None:
                self.continuous_calls: list[int] = []
                FakeRunner.last_instance = self

            def run_continuous_round(self, limit: int) -> None:
                self.continuous_calls.append(limit)

        with patch("arena.cli._load_arena_packages", return_value={"a": object(), "b": object()}), patch(
            "arena.cli._ensure_runtime_deck_records"
        ), patch("arena.cli.ArenaRunner", FakeRunner) as runner_type, patch(
            "arena.cli.build_report_snapshot", return_value={}
        ), patch(
            "arena.cli.write_report_pair",
            return_value={"all_html": "all", "eligible_html": "eligible"},
        ):
            run_command(args)

        self.assertIsNotNone(FakeRunner.last_instance)
        self.assertEqual(FakeRunner.last_instance.continuous_calls, [1])

    def test_keyboard_interrupt_marks_running_arena_as_interrupted(self) -> None:
        store = ArenaStore(self.repo_root / "arena")
        store.initialize()
        store.create_run(RunRecord("run-1", "continuous", {}, "now", None, "running"))
        args = Namespace(
            repo_root=self.repo_root,
            phase="continuous",
            games=20,
            limit=1,
            resume="run-1",
            include_demoted=False,
            seed=7,
            max_steps=10,
            timeout=1.0,
            checkpoint_every=1,
        )

        class InterruptingRunner:
            def __init__(self, *_: object, **__: object) -> None:
                pass

            def run_continuous_round(self, limit: int) -> None:
                raise KeyboardInterrupt

        with patch("arena.cli._load_arena_packages", return_value={"a": object(), "b": object()}), patch(
            "arena.cli._ensure_runtime_deck_records"
        ), patch("arena.cli.ArenaRunner", InterruptingRunner):
            with self.assertRaises(KeyboardInterrupt):
                run_command(args)

        self.assertEqual(store.load_run("run-1")["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
