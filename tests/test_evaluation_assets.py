from __future__ import annotations

import json
import unittest
from pathlib import Path

from evaluation.runtime.loader import compute_cg_manifest


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "evaluation" / "configs" / "opponents.json"
ARENA_ROOT = ROOT / "evaluation" / "arena"
OPPONENTS_ROOT = ARENA_ROOT / "opponents"
CANDIDATES_ROOT = ARENA_ROOT / "candidates"
EVALUATION_CG_BASELINE = (
    ROOT / "submission" / "alakazam_gen1_rule_based" / "alakazam_v8" / "cg"
)
FOREIGN_EVALUATION_REPOSITORY = "/Users/hejinyu/Documents/repos/ptcg-agent-kaggle"
EXPECTED_NAMES = (
    "alakazam_dudunsparce_01",
    "alakazam_dudunsparce_02",
    "alakazam_dudunsparce_03",
    "alakazam_dudunsparce_04",
    "crustle_01",
    "crustle_02",
    "dragapult_ex_01",
    "dragapult_ex_02",
    "dragapult_ex_03",
    "ionos_bellibolt_ex_kilowattrel_01",
    "marnies_grimmsnarl_ex_dudunsparce_01",
    "marnies_grimmsnarl_ex_froslass_01",
    "mega_abomasnow_ex_kyogre_01",
    "mega_lucario_ex_solrock_01",
    "mega_lucario_ex_solrock_02",
    "mega_lucario_ex_solrock_03",
    "mega_lucario_ex_solrock_04",
    "mega_lucario_ex_solrock_05",
    "mega_lucario_ex_solrock_06",
    "mega_lucario_ex_solrock_07",
    "mega_lucario_ex_solrock_08",
    "mega_lucario_ex_solrock_09",
    "mega_lucario_ex_solrock_10",
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
                self.assertEqual(opponent["package"], f"arena/opponents/{name}")
                self.assertTrue(opponent["display_name"])
                self.assertIn(len(opponent["representative_card_ids"]), (1, 2))
                self.assertTrue(package_root.is_dir())
                self.assertTrue((package_root / "main.py").is_file())

                deck_lines = (package_root / "deck.csv").read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(deck_lines), 60)
                self.assertTrue(all(line.strip() for line in deck_lines))
                deck_ids = {int(line) for line in deck_lines}
                self.assertTrue(
                    set(opponent["representative_card_ids"]).issubset(deck_ids)
                )

                cg_root = package_root / "cg"
                self.assertTrue(cg_root.is_dir())
                self.assertFalse(cg_root.is_symlink())
                self.assertEqual(compute_cg_manifest(cg_root)["tree_hash"], self.baseline_hash)

    def test_opponent_entrypoints_do_not_reference_the_external_evaluation_repository(self) -> None:
        for name in EXPECTED_NAMES:
            with self.subTest(opponent=name):
                source = (OPPONENTS_ROOT / name / "main.py").read_text(encoding="utf-8")
                self.assertNotIn(FOREIGN_EVALUATION_REPOSITORY, source)

    def test_arena_candidates_is_a_separate_non_catalog_staging_area(self) -> None:
        self.assertTrue(CANDIDATES_ROOT.is_dir())
        self.assertTrue((CANDIDATES_ROOT / "README.md").is_file())
        catalog_packages = {opponent["package"] for opponent in self.catalog["opponents"]}
        self.assertTrue(all(package.startswith("arena/opponents/") for package in catalog_packages))
        self.assertFalse(any(package.startswith("arena/candidates/") for package in catalog_packages))


if __name__ == "__main__":
    unittest.main()
