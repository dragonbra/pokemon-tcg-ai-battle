from __future__ import annotations

import json
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.cli import (
    _validate_research_coverage,
    list_enabled_opponents,
    load_opponent_catalog,
    validate_catalog,
)
from evaluation.packages.loader import PackageValidationError, load_submission_package


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "evaluation" / "configs" / "opponents.json"
EVALUATION_ROOT = ROOT / "evaluation"
EXPECTED_NAMES = [
    entry["name"]
    for entry in json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["opponents"]
    if entry["enabled"]
]


def official_card_ids() -> set[int]:
    ids: set[int] = set()
    with (ROOT / "data" / "official" / "EN_Card_Data.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as card_file:
        for row in csv.reader(card_file):
            if row and row[0] != "Card ID":
                ids.add(int(row[0].split(":")[-1]))
    return ids


class EvaluationCatalogTests(unittest.TestCase):
    def test_catalog_loads_all_enabled_opponents(self) -> None:
        packages = load_opponent_catalog(CATALOG_PATH, EVALUATION_ROOT)

        self.assertEqual([package.name for package in packages], EXPECTED_NAMES)
        self.assertTrue(
            all(
                package.root.parent == EVALUATION_ROOT / "arena" / "opponents"
                for package in packages
            )
        )
        self.assertTrue(all(package.display_name for package in packages))
        self.assertTrue(all(package.representative_cards for package in packages))
        self.assertTrue(
            all(
                str(card["image_url"]).startswith("https://images.")
                for package in packages
                for card in package.representative_cards
            )
        )

    def test_validate_catalog_checks_all_packages_and_card_ids(self) -> None:
        packages = validate_catalog(CATALOG_PATH, official_card_ids())

        self.assertEqual([package.name for package in packages], EXPECTED_NAMES)

    def test_list_enabled_opponents_returns_catalog_order(self) -> None:
        self.assertEqual(list_enabled_opponents(CATALOG_PATH), EXPECTED_NAMES)

    def test_list_enabled_opponents_excludes_disabled_entries(self) -> None:
        payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        payload["opponents"][0]["enabled"] = False

        with patch("evaluation.cli._read_catalog_entries", return_value=payload["opponents"]):
            names = list_enabled_opponents(CATALOG_PATH)

        self.assertEqual(names, EXPECTED_NAMES[1:])

    def test_cli_lists_exactly_enabled_names(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "evaluation",
                "--pool",
                "opponents",
                "list-opponents",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), EXPECTED_NAMES)

    def test_catalog_rejects_duplicate_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "opponents.json"
            payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
            payload["opponents"][1]["name"] = payload["opponents"][0]["name"]
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(PackageValidationError, "duplicate"):
                load_opponent_catalog(path, EVALUATION_ROOT)

    def test_catalog_rejects_duplicate_disabled_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "opponents.json"
            payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
            payload["opponents"][1]["name"] = payload["opponents"][0]["name"]
            payload["opponents"][1]["enabled"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(PackageValidationError, "duplicate"):
                load_opponent_catalog(path, EVALUATION_ROOT)

    def test_catalog_rejects_unknown_fields_and_wrong_types(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "opponents.json"
            payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
            payload["opponents"][0]["extra"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(PackageValidationError, "unknown"):
                load_opponent_catalog(path, EVALUATION_ROOT)

            payload["opponents"][0].pop("extra")
            payload["opponents"][0]["enabled"] = "yes"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(PackageValidationError, "enabled"):
                load_opponent_catalog(path, EVALUATION_ROOT)

    def test_catalog_rejects_path_escape_and_missing_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "opponents.json"
            payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

            payload["opponents"][0]["package"] = "../submission/alakazam_v8"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(PackageValidationError, "inside"):
                load_opponent_catalog(path, EVALUATION_ROOT)

            payload["opponents"][0]["package"] = "arena/opponents/does_not_exist"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(PackageValidationError, "does not exist"):
                load_opponent_catalog(path, EVALUATION_ROOT)

    def test_catalog_rejects_disabled_path_escape_and_missing_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "opponents.json"
            payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
            payload["opponents"][0]["enabled"] = False
            payload["opponents"][0]["package"] = "../submission/alakazam_v8"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(PackageValidationError, "inside"):
                load_opponent_catalog(path, EVALUATION_ROOT)

            payload["opponents"][0]["package"] = "arena/opponents/does_not_exist"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(PackageValidationError, "does not exist"):
                load_opponent_catalog(path, EVALUATION_ROOT)

    def test_catalog_rejects_arena_candidate_packages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "opponents.json"
            payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
            payload["opponents"][0]["package"] = "arena/candidates/staged_opponent"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(PackageValidationError, "arena/opponents"):
                load_opponent_catalog(path, EVALUATION_ROOT)

    def test_research_coverage_requires_full_catalog_and_ten_games(self) -> None:
        with self.assertRaisesRegex(PackageValidationError, "230"):
            _validate_research_coverage("all", 2, tuple(range(23)))  # type: ignore[arg-type]
        with self.assertRaisesRegex(PackageValidationError, "opponents all"):
            _validate_research_coverage("one", 10, tuple(range(23)))  # type: ignore[arg-type]
        _validate_research_coverage("all", 10, tuple(range(23)))  # type: ignore[arg-type]

    def test_catalog_skips_loading_disabled_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "opponents.json"
            payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
            payload["opponents"][0]["enabled"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")

            with patch(
                "evaluation.cli.load_submission_package", wraps=load_submission_package
            ) as load_package:
                load_opponent_catalog(path, EVALUATION_ROOT)

            self.assertEqual(load_package.call_count, len(EXPECTED_NAMES) - 1)


if __name__ == "__main__":
    unittest.main()
