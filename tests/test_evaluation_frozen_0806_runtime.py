from __future__ import annotations

import unittest
from pathlib import Path

from evaluation.frozen_0806_runtime import load_frozen_0806_runtime_catalog
from evaluation.runtime.loader import assert_cg_compatible


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = ROOT / "evaluation"
CONFIG = EVALUATION_ROOT / "configs" / "frozen_0806.json"


class Frozen0806RuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_frozen_0806_runtime_catalog(CONFIG, EVALUATION_ROOT)

    def test_two_shared_policy_runtimes_cover_all_55_decks(self) -> None:
        catalog = self.catalog

        self.assertEqual(len(catalog.candidates), 55)
        self.assertEqual(len(catalog.opponents), 55)
        self.assertEqual(len({package.root for package in catalog.candidates}), 1)
        self.assertEqual(len({package.root for package in catalog.opponents}), 1)
        self.assertNotEqual(catalog.candidate_policy.root, catalog.opponent_policy.root)
        self.assertTrue(all(package.entrypoint == catalog.candidate_policy.entrypoint for package in catalog.candidates))
        self.assertTrue(all(package.entrypoint == catalog.opponent_policy.entrypoint for package in catalog.opponents))

    def test_deck_routed_identities_match_fixed_schedule(self) -> None:
        catalog = self.catalog
        expected = {entry.deck_id: entry.exact_deck_sha256 for entry in catalog.pool.schedule}

        self.assertEqual(
            {package.name: package.package_manifest["exact_deck_sha256"] for package in catalog.candidates},
            expected,
        )
        self.assertEqual(
            {package.name: package.package_manifest["exact_deck_sha256"] for package in catalog.opponents},
            expected,
        )
        self.assertTrue(all(len(package.deck) == 60 for package in catalog.candidates))

    def test_candidate_and_opponent_use_compatible_official_runtime(self) -> None:
        assert_cg_compatible(self.catalog.candidates[0], self.catalog.opponents[0])
        self.assertEqual(
            self.catalog.candidate_policy.cg_manifest["tree_hash"],
            self.catalog.opponent_policy.cg_manifest["tree_hash"],
        )


if __name__ == "__main__":
    unittest.main()
