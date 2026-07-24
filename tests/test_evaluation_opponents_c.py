from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path

from evaluation.runtime.loader import compute_cg_manifest


ROOT = Path(__file__).resolve().parents[1]
DECK_FIXTURE = ROOT / "tests" / "fixtures" / "evaluation_opponent_decks.json"
OPPONENTS_ROOT = ROOT / "evaluation" / "arena" / "opponents"
CG_SOURCE_ROOT = ROOT / "submission" / "alakazam_gen1_rule_based" / "alakazam_v8" / "cg"
OPPONENT_NAMES = (
    "alakazam_dudunsparce_01",
    "alakazam_dudunsparce_02",
)
YAN_N_Z_FORBIDDEN_REFERENCES = (
    "ptcg-agent-kaggle",
    "AGENT_DIR",
    "AGENT_MAIN",
    "agent/yanxiaohan",
    "_load_agent_module",
)


def fixture_decks() -> dict[str, list[int]]:
    entries = json.loads(DECK_FIXTURE.read_text(encoding="utf-8"))
    return {
        str(entry["name"]): list(entry["deck"])
        for entry in entries
        if str(entry["name"]) in OPPONENT_NAMES
    }


def read_deck(path: Path) -> list[int]:
    return [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_package_in_subprocess(package_root: Path) -> dict[str, object]:
    script = """
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
package_root = Path(sys.argv[2])
sys.path.insert(0, str(root))
from evaluation.packages.loader import load_submission_package

with (root / 'data' / 'official' / 'EN_Card_Data.csv').open(
    newline='', encoding='utf-8-sig'
) as source:
    official_card_ids = {
        int(row[0].split(':')[-1])
        for row in csv.reader(source)
        if row and row[0] != 'Card ID'
    }

package = load_submission_package(package_root, official_card_ids)
print(json.dumps({'name': package.name, 'deck': package.deck, 'root': str(package.root)}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(ROOT), str(package_root)],
        cwd=ROOT / "tests",
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"loader failed for {package_root.name}:\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    return json.loads(result.stdout)


class EvaluationOpponentPoolCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expected_cg_tree_hash = compute_cg_manifest(CG_SOURCE_ROOT)["tree_hash"]
        cls.decks = fixture_decks()

    def package_root(self, name: str) -> Path:
        return OPPONENTS_ROOT / name

    def test_fixture_contains_all_ordered_60_card_decks(self) -> None:
        self.assertEqual(tuple(self.decks), OPPONENT_NAMES)
        for name in OPPONENT_NAMES:
            with self.subTest(opponent=name):
                self.assertEqual(len(self.decks[name]), 60)

    def test_deck_csv_matches_fixture_order_hash_and_card_counts(self) -> None:
        for name in OPPONENT_NAMES:
            with self.subTest(opponent=name):
                deck = read_deck(self.package_root(name) / "deck.csv")
                expected = self.decks[name]
                self.assertEqual(deck, expected)
                self.assertEqual(len(deck), 60)
                self.assertEqual(
                    hashlib.sha256(("\n".join(map(str, deck)) + "\n").encode()).hexdigest(),
                    hashlib.sha256(("\n".join(map(str, expected)) + "\n").encode()).hexdigest(),
                )
                self.assertEqual(Counter(deck), Counter(expected))
                self.assertEqual(set(deck), set(expected))

    def test_each_package_independently_loads_and_returns_its_local_deck(self) -> None:
        for name in OPPONENT_NAMES:
            with self.subTest(opponent=name):
                package_root = self.package_root(name)
                package = load_package_in_subprocess(package_root)

                self.assertEqual(package["name"], name)
                self.assertEqual(package["deck"], self.decks[name])
                self.assertEqual(package["deck"], read_deck(package_root / "deck.csv"))
                self.assertEqual(package["root"], str(package_root.absolute()))

    def test_each_package_has_a_physical_runtime_matching_alakazam_v8(self) -> None:
        for name in OPPONENT_NAMES:
            with self.subTest(opponent=name):
                cg_root = self.package_root(name) / "cg"
                self.assertTrue(cg_root.is_dir())
                self.assertFalse(cg_root.is_symlink())
                self.assertFalse(any(path.is_symlink() for path in cg_root.rglob("*")))
                self.assertEqual(compute_cg_manifest(cg_root)["tree_hash"], self.expected_cg_tree_hash)

    def test_renamed_lucario_package_is_self_contained(self) -> None:
        main_source = (
            self.package_root("mega_lucario_ex_solrock_05") / "main.py"
        ).read_text(encoding="utf-8")

        for forbidden in YAN_N_Z_FORBIDDEN_REFERENCES:
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, main_source)


if __name__ == "__main__":
    unittest.main()
