from __future__ import annotations

import csv
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from evaluation.packages.loader import load_submission_package
from evaluation.runtime.loader import compute_cg_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DECK_FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "evaluation_opponent_decks.json"
CG_SOURCE_ROOT = REPOSITORY_ROOT / "submission" / "alakazam_v8" / "cg"
OPPONENT_NAMES = (
    "romanrozen_v9",
    "pilkwang_v2",
    "kokinn_search",
    "penguin_915",
    "crustle_wall",
    "crustle_v1",
)


def read_fixture() -> list[dict[str, object]]:
    return json.loads(DECK_FIXTURE.read_text(encoding="utf-8"))


def fixture_decks() -> dict[str, list[int]]:
    return {
        str(entry["name"]): list(entry["deck"])
        for entry in read_fixture()
        if str(entry["name"]) in OPPONENT_NAMES
    }


def official_card_ids() -> set[int]:
    card_data_path = REPOSITORY_ROOT / "data" / "official" / "EN_Card_Data.csv"
    with card_data_path.open(encoding="utf-8-sig", newline="") as card_data:
        return {int(row["Card ID"]) for row in csv.DictReader(card_data)}


def read_deck(package_root: Path) -> list[int]:
    return [
        int(line)
        for line in (package_root / "deck.csv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class EvaluationOpponentPoolATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.card_ids = official_card_ids()
        cls.decks = fixture_decks()
        cls.expected_cg_tree_hash = compute_cg_manifest(CG_SOURCE_ROOT)["tree_hash"]

    def package_root(self, name: str) -> Path:
        return REPOSITORY_ROOT / "evaluation" / "opponents" / name

    def test_fixture_contains_all_ordered_60_card_decks(self) -> None:
        self.assertEqual(tuple(self.decks), OPPONENT_NAMES)
        for name in OPPONENT_NAMES:
            with self.subTest(opponent=name):
                self.assertEqual(len(self.decks[name]), 60)

    def test_deck_csv_matches_fixture_order_hash_and_card_counts(self) -> None:
        for name in OPPONENT_NAMES:
            with self.subTest(opponent=name):
                expected = self.decks[name]
                deck = read_deck(self.package_root(name))
                self.assertEqual(deck, expected)
                self.assertEqual(len(deck), 60)
                self.assertEqual(Counter(deck), Counter(expected))
                self.assertEqual(set(deck), set(expected))
                self.assertEqual(
                    hashlib.sha256(("\n".join(map(str, deck)) + "\n").encode()).hexdigest(),
                    hashlib.sha256(("\n".join(map(str, expected)) + "\n").encode()).hexdigest(),
                )

    def test_each_opponent_is_independently_loadable(self) -> None:
        for name in OPPONENT_NAMES:
            with self.subTest(opponent=name):
                package = load_submission_package(self.package_root(name), self.card_ids)
                self.assertEqual(package.name, name)
                self.assertEqual(package.deck, self.decks[name])
                self.assertEqual(len(package.deck), 60)

    def test_each_opponent_has_a_physical_complete_cg_runtime(self) -> None:
        native_files = [
            source_path.name
            for source_path in CG_SOURCE_ROOT.iterdir()
            if source_path.is_file() and source_path.suffix in {".dll", ".dylib", ".so"}
        ]
        for name in OPPONENT_NAMES:
            with self.subTest(opponent=name):
                cg_root = self.package_root(name) / "cg"
                self.assertTrue(cg_root.is_dir())
                self.assertFalse(cg_root.is_symlink())
                self.assertFalse(any(path.is_symlink() for path in cg_root.rglob("*")))
                self.assertEqual(compute_cg_manifest(cg_root)["tree_hash"], self.expected_cg_tree_hash)
                for relative_path in ("__init__.py", "api.py", "game.py", "sim.py", *native_files):
                    self.assertTrue((cg_root / relative_path).is_file(), relative_path)


if __name__ == "__main__":
    unittest.main()
