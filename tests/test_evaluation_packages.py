from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.packages import loader as package_loader
from evaluation.packages.loader import PackageValidationError, load_submission_package
from evaluation.packages.validator import validate_submission_package
from evaluation.schemas.json_schema import (
    validate_case_record,
    validate_game_record,
    validate_manifest,
    validate_metric_payload,
)


class EvaluationPackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_package(
        self,
        *,
        deck: list[int],
        main_source: str | None = None,
    ) -> Path:
        package_root = self.root / "candidate"
        cg_root = package_root / "cg"
        cg_root.mkdir(parents=True)
        (cg_root / "__init__.py").write_text("", encoding="utf-8")
        (cg_root / "api.py").write_text("API_VALUE = 7\n", encoding="utf-8")
        (cg_root / "game.py").write_text(
            "from cg.api import API_VALUE\nGAME_VALUE = API_VALUE\n",
            encoding="utf-8",
        )
        (cg_root / "libcg.so").write_bytes(b"test-runtime")
        deck_text = "\n".join(str(card_id) for card_id in deck) + "\n"
        (package_root / "deck.csv").write_text(deck_text, encoding="utf-8")
        if main_source is None:
            main_source = (
                "from cg.api import API_VALUE\n\n"
                f"def agent(observation):\n    return {deck!r}\n"
            )
        (package_root / "main.py").write_text(main_source, encoding="utf-8")
        return package_root

    def test_loader_reads_deck_and_hashes_package(self) -> None:
        package_root = self.make_package(deck=[1] * 60)

        package = load_submission_package(package_root, {1})

        self.assertEqual(package.deck, [1] * 60)
        self.assertEqual(package.entrypoint, package_root / "main.py")
        self.assertTrue(package.deck_hash)
        self.assertTrue(package.package_hash)
        self.assertGreaterEqual(package.cg_manifest["file_count"], 1)

    def test_loader_validates_main_in_a_subprocess(self) -> None:
        package_root = self.make_package(
            deck=[1] * 60,
            main_source=(
                "import os\n"
                "from pathlib import Path\n\n"
                "from cg.api import API_VALUE\n\n"
                "with (Path(__file__).parent / 'loader_pids.txt').open(\n"
                "    'a', encoding='utf-8'\n"
                ") as pid_file:\n"
                "    pid_file.write(str(os.getpid()) + '\\n')\n\n"
                "def agent(observation):\n"
                "    return [1] * 60\n"
            ),
        )
        (package_root / "loader_pids.txt").unlink(missing_ok=True)

        load_submission_package(package_root, {1})

        loader_pids = (package_root / "loader_pids.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(loader_pids), 1)
        self.assertNotEqual(int(loader_pids[0]), os.getpid())

    def test_kaggle_raw_exec_rejects_unconditional_dunder_file(self) -> None:
        package_root = self.make_package(
            deck=[1] * 60,
            main_source=(
                "from pathlib import Path\n"
                "ROOT = Path(__file__).resolve().parent\n\n"
                "def agent(observation):\n"
                "    return [1] * 60\n"
            ),
        )

        with self.assertRaisesRegex(PackageValidationError, "__file__"):
            package_loader.validate_kaggle_raw_exec(
                package_root / "main.py",
                package_root,
                [1] * 60,
            )

    def test_kaggle_raw_exec_accepts_cwd_fallback(self) -> None:
        package_root = self.make_package(
            deck=[1] * 60,
            main_source=(
                "from pathlib import Path\n"
                "ROOT = Path(globals().get('__file__', Path.cwd())).resolve()\n"
                "if ROOT.is_file():\n"
                "    ROOT = ROOT.parent\n"
                "DECK = [int(value) for value in "
                "(ROOT / 'deck.csv').read_text().splitlines()]\n\n"
                "def agent(observation):\n"
                "    return list(DECK)\n"
            ),
        )

        package_loader.validate_kaggle_raw_exec(
            package_root / "main.py",
            package_root,
            [1] * 60,
        )

    def test_loader_rejects_system_exit_during_main_import(self) -> None:
        package_root = self.make_package(
            deck=[1] * 60,
            main_source="from cg.api import API_VALUE\nraise SystemExit(0)\n",
        )

        with self.assertRaisesRegex(PackageValidationError, "subprocess validation failed"):
            load_submission_package(package_root, {1})

    def test_loader_rejects_system_exit_from_agent_callback(self) -> None:
        package_root = self.make_package(
            deck=[1] * 60,
            main_source=(
                "from cg.api import API_VALUE\n\n"
                "def agent(observation):\n"
                "    raise SystemExit(0)\n"
            ),
        )

        with self.assertRaisesRegex(PackageValidationError, "subprocess validation failed"):
            load_submission_package(package_root, {1})

    def test_loader_rejects_os_exit_zero_during_main_import(self) -> None:
        package_root = self.make_package(
            deck=[1] * 60,
            main_source="import os\nos._exit(0)\n",
        )

        with self.assertRaisesRegex(PackageValidationError, "subprocess validation failed"):
            load_submission_package(package_root, {1})

    def test_loader_rejects_os_exit_nested_success_code_during_main_import(self) -> None:
        package_root = self.make_package(
            deck=[1] * 60,
            main_source="import os\nos._exit(73)\n",
        )

        with self.assertRaisesRegex(PackageValidationError, "subprocess validation failed"):
            load_submission_package(package_root, {1})

    def test_loader_rejects_spoofed_success_marker_before_os_exit(self) -> None:
        package_root = self.make_package(
            deck=[1] * 60,
            main_source=(
                "import os\n"
                "print('EVALUATION_AGENT_VALIDATION_OK', flush=True)\n"
                "os._exit(0)\n"
            ),
        )

        with self.assertRaisesRegex(PackageValidationError, "subprocess validation failed"):
            load_submission_package(package_root, {1})

    def test_loader_does_not_import_shadowed_trusted_loader(self) -> None:
        package_root = self.make_package(
            deck=[1] * 60,
            main_source="VALUE = 1\n",
        )
        shadow_loader = package_root / "evaluation" / "packages"
        shadow_loader.mkdir(parents=True)
        (package_root / "evaluation" / "__init__.py").write_text("", encoding="utf-8")
        (shadow_loader / "__init__.py").write_text("", encoding="utf-8")
        (shadow_loader / "loader.py").write_text(
            "def _validate_agent(entrypoint, root, deck):\n"
            "    return None\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(PackageValidationError, "agent"):
            load_submission_package(package_root, {1})

    def test_loader_wraps_subprocess_timeout(self) -> None:
        package_root = self.make_package(
            deck=[1] * 60,
            main_source=(
                "import time\n"
                "from cg.api import API_VALUE\n\n"
                "time.sleep(0.2)\n\n"
                "def agent(observation):\n"
                "    return [1] * 60\n"
            ),
        )

        with patch.object(
            package_loader,
            "SUBPROCESS_VALIDATION_TIMEOUT_SECONDS",
            0.01,
            create=True,
        ):
            with self.assertRaisesRegex(PackageValidationError, "timed out"):
                load_submission_package(package_root, {1})

    def test_loader_preserves_parent_cg_modules(self) -> None:
        package_root = self.make_package(deck=[1] * 60)
        parent_cg = types.ModuleType("cg")
        parent_cg_api = types.ModuleType("cg.api")

        with patch.dict(sys.modules, {"cg": parent_cg, "cg.api": parent_cg_api}):
            load_submission_package(package_root, {1})
            self.assertIs(sys.modules.get("cg"), parent_cg)
            self.assertIs(sys.modules.get("cg.api"), parent_cg_api)

    def test_loader_rejects_wrong_deck_size(self) -> None:
        package_root = self.make_package(deck=[1] * 59)

        with self.assertRaisesRegex(PackageValidationError, "60"):
            load_submission_package(package_root, {1})

    def test_loader_rejects_unknown_card_id(self) -> None:
        package_root = self.make_package(deck=[999] * 60)

        with self.assertRaisesRegex(PackageValidationError, "999"):
            load_submission_package(package_root, {1})

    def test_loader_rejects_empty_deck(self) -> None:
        package_root = self.make_package(deck=[])

        with self.assertRaisesRegex(PackageValidationError, "empty"):
            load_submission_package(package_root, {1})

    def test_loader_rejects_header_or_non_integer_deck_row(self) -> None:
        package_root = self.make_package(deck=[1] * 60)
        (package_root / "deck.csv").write_text("card_id\n", encoding="utf-8")

        with self.assertRaisesRegex(PackageValidationError, "integer"):
            load_submission_package(package_root, {1})

    def test_loader_rejects_non_positive_card_id(self) -> None:
        package_root = self.make_package(deck=[1] * 59 + [0])

        with self.assertRaisesRegex(PackageValidationError, "positive"):
            load_submission_package(package_root, {1})

    def test_loader_requires_agent_and_deck_callback(self) -> None:
        package_root = self.make_package(deck=[1] * 60, main_source="VALUE = 1\n")

        with self.assertRaisesRegex(PackageValidationError, "agent"):
            load_submission_package(package_root, {1})

    def test_loader_rejects_incomplete_cg_runtime(self) -> None:
        package_root = self.make_package(deck=[1] * 60)
        (package_root / "cg" / "game.py").unlink()

        with self.assertRaisesRegex(PackageValidationError, "game.py"):
            load_submission_package(package_root, {1})

    def test_loader_rejects_cg_runtime_without_package_initializer(self) -> None:
        package_root = self.make_package(deck=[1] * 60)
        (package_root / "cg" / "__init__.py").unlink()

        with self.assertRaisesRegex(PackageValidationError, "__init__.py"):
            load_submission_package(package_root, {1})

    def test_loader_requires_agent_callback_to_return_the_deck(self) -> None:
        package_root = self.make_package(
            deck=[1] * 60,
            main_source="def agent(observation):\n    return [2] * 60\n",
        )

        with self.assertRaisesRegex(PackageValidationError, "deck.csv"):
            load_submission_package(package_root, {1})

    def test_package_hash_ignores_python_cache_files(self) -> None:
        package_root = self.make_package(deck=[1] * 60)
        initial = load_submission_package(package_root, {1})
        cache_root = package_root / "cg" / "__pycache__"
        cache_root.mkdir(exist_ok=True)
        (cache_root / "ignored.pyc").write_bytes(b"cache")

        cached = load_submission_package(package_root, {1})

        self.assertEqual(initial.package_hash, cached.package_hash)

    def test_package_hash_commits_strategy_checkpoint_and_manifest(self) -> None:
        package_root = self.make_package(deck=[1] * 60)
        strategy = package_root / "strategy"
        strategy.mkdir()
        checkpoint = strategy / "model.bin"
        checkpoint.write_bytes(b"checkpoint-a")
        manifest = package_root / "manifest.json"
        manifest.write_text('{"update": 10}\n', encoding="utf-8")
        initial = load_submission_package(package_root, {1})
        self.assertEqual(initial.package_manifest, {"update": 10})

        checkpoint.write_bytes(b"checkpoint-b")
        changed_checkpoint = load_submission_package(package_root, {1})
        self.assertNotEqual(initial.package_hash, changed_checkpoint.package_hash)

        checkpoint.write_bytes(b"checkpoint-a")
        manifest.write_text('{"update": 20}\n', encoding="utf-8")
        changed_manifest = load_submission_package(package_root, {1})
        self.assertNotEqual(initial.package_hash, changed_manifest.package_hash)

    def test_validator_returns_the_loaded_package(self) -> None:
        package_root = self.make_package(deck=[1] * 60)

        package = validate_submission_package(package_root, {1})

        self.assertEqual(package.root, package_root)

    def test_json_schema_validators_reject_missing_required_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "name"):
            validate_manifest({})
        with self.assertRaisesRegex(ValueError, "game_id"):
            validate_game_record({})
        with self.assertRaisesRegex(ValueError, "metric"):
            validate_metric_payload({})
        with self.assertRaisesRegex(ValueError, "case_id"):
            validate_case_record({})

    def test_json_schema_validators_accept_minimal_contracts(self) -> None:
        validate_manifest(
            {"name": "candidate", "package_hash": "a", "deck_hash": "b", "cg_manifest": {}}
        )
        validate_game_record(
            {
                "game_id": "game-1",
                "candidate": "candidate",
                "opponent": "opponent",
                "result": {"winner": "candidate", "reason": "prizes"},
            }
        )
        validate_metric_payload(
            {"metric": "win_rate", "scope": "aggregate", "value": 0.5, "sample_size": 2}
        )
        validate_case_record(
            {"case_id": "case-1", "name": "smoke", "status": "passed", "details": {}}
        )


if __name__ == "__main__":
    unittest.main()
