from __future__ import annotations

import json
import unittest
from pathlib import Path

from evaluation.runtime.loader import compute_cg_manifest


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "evaluation" / "configs" / "opponents.json"
OPPONENTS_ROOT = ROOT / "evaluation" / "opponents"
EVALUATION_CG_BASELINE = (
    ROOT / "submission" / "alakazam_gen1_rule_based" / "alakazam_v8" / "cg"
)
FOREIGN_EVALUATION_REPOSITORY = "/Users/hejinyu/Documents/repos/ptcg-agent-kaggle"
EXPECTED_NAMES = (
    "romanrozen_v9",
    "pilkwang_v2",
    "kokinn_search",
    "penguin_915",
    "crustle_wall",
    "crustle_v1",
    "kiyotah_lucario",
    "kiyotah_dragapult",
    "kiyotah_iono",
    "kiyotah_abomasnow",
    "kacchan_anti_wall",
    "nursrijan_lucario",
    "zoli_dragapult",
    "sue_alakazam",
    "maktha_1084",
    "yanxiaohan",
    "Agent_Lucario",
    "Agent_Aluxian",
)


class EvaluationAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        cls.baseline_hash = compute_cg_manifest(EVALUATION_CG_BASELINE)["tree_hash"]

    def test_catalog_contains_the_exact_enabled_opponent_set(self) -> None:
        opponents = self.catalog["opponents"]

        self.assertEqual([opponent["name"] for opponent in opponents], list(EXPECTED_NAMES))
        self.assertTrue(all(opponent["enabled"] for opponent in opponents))
        self.assertEqual(len(opponents), len(EXPECTED_NAMES))

    def test_each_catalog_opponent_is_a_self_contained_standard_package(self) -> None:
        for opponent in self.catalog["opponents"]:
            name = opponent["name"]
            package_root = OPPONENTS_ROOT / name
            with self.subTest(opponent=name):
                self.assertEqual(opponent["package"], f"opponents/{name}")
                self.assertTrue(package_root.is_dir())
                self.assertTrue((package_root / "main.py").is_file())

                deck_lines = (package_root / "deck.csv").read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(deck_lines), 60)
                self.assertTrue(all(line.strip() for line in deck_lines))

                cg_root = package_root / "cg"
                self.assertTrue(cg_root.is_dir())
                self.assertFalse(cg_root.is_symlink())
                self.assertEqual(compute_cg_manifest(cg_root)["tree_hash"], self.baseline_hash)

    def test_opponent_entrypoints_do_not_reference_the_external_evaluation_repository(self) -> None:
        for name in EXPECTED_NAMES:
            with self.subTest(opponent=name):
                source = (OPPONENTS_ROOT / name / "main.py").read_text(encoding="utf-8")
                self.assertNotIn(FOREIGN_EVALUATION_REPOSITORY, source)


if __name__ == "__main__":
    unittest.main()
