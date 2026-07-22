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
OPPONENTS_ROOT = REPOSITORY_ROOT / "evaluation" / "opponents"
CG_SOURCE_ROOT = (
    REPOSITORY_ROOT / "submission" / "alakazam_gen1_rule_based" / "alakazam_v8" / "cg"
)
OPPONENT_NAMES = (
    "kiyotah_lucario",
    "kiyotah_dragapult",
    "kiyotah_iono",
    "kiyotah_abomasnow",
    "kacchan_anti_wall",
    "nursrijan_lucario",
)


def fixture_decks() -> dict[str, list[int]]:
    entries = json.loads(DECK_FIXTURE.read_text(encoding="utf-8"))
    return {
        str(entry["name"]): list(entry["deck"])
        for entry in entries
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


class EvaluationOpponentPoolBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.card_ids = official_card_ids()
        cls.expected_cg_tree_hash = compute_cg_manifest(CG_SOURCE_ROOT)["tree_hash"]
        cls.decks = fixture_decks()

    def test_fixture_contains_all_ordered_60_card_decks(self) -> None:
        self.assertEqual(tuple(self.decks), OPPONENT_NAMES)
        for name in OPPONENT_NAMES:
            with self.subTest(opponent=name):
                self.assertEqual(len(self.decks[name]), 60)

    def test_deck_csv_matches_fixture_order_hash_and_card_counts(self) -> None:
        for name in OPPONENT_NAMES:
            with self.subTest(opponent=name):
                expected = self.decks[name]
                deck = read_deck(OPPONENTS_ROOT / name)
                self.assertEqual(deck, expected)
                self.assertEqual(len(deck), 60)
                self.assertEqual(
                    hashlib.sha256(("\n".join(map(str, deck)) + "\n").encode()).hexdigest(),
                    hashlib.sha256(("\n".join(map(str, expected)) + "\n").encode()).hexdigest(),
                )
                self.assertEqual(Counter(deck), Counter(expected))
                self.assertEqual(set(deck), set(expected))

    def test_each_package_uses_its_local_deck_and_validates_independently(self) -> None:
        for name in OPPONENT_NAMES:
            with self.subTest(opponent=name):
                package_root = OPPONENTS_ROOT / name
                package = load_submission_package(package_root, self.card_ids)

                self.assertEqual(package.name, name)
                self.assertEqual(package.deck, self.decks[name])
                self.assertEqual(package.deck, read_deck(package_root))
                self.assertEqual(
                    package.deck_hash,
                    hashlib.sha256(
                        ("\n".join(map(str, self.decks[name])) + "\n").encode()
                    ).hexdigest(),
                )
                self.assertEqual(package.root, package_root.absolute())
                self.assertEqual(
                    (package_root / "main.py").read_text(encoding="utf-8").count("PACKAGE_ROOT"),
                    2,
                )

    def test_each_package_has_a_physical_runtime_matching_alakazam_v8(self) -> None:
        for name in OPPONENT_NAMES:
            with self.subTest(opponent=name):
                cg_root = OPPONENTS_ROOT / name / "cg"

                self.assertTrue(cg_root.is_dir())
                self.assertFalse(cg_root.is_symlink())
                self.assertFalse(any(path.is_symlink() for path in cg_root.rglob("*")))
                self.assertEqual(compute_cg_manifest(cg_root)["tree_hash"], self.expected_cg_tree_hash)


if __name__ == "__main__":
    unittest.main()
