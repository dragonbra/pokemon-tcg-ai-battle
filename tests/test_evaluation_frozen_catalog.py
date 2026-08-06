from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from evaluation.frozen import load_frozen_catalog


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = ROOT / "evaluation"
FROZEN_CATALOG = (
    ROOT / "archive" / "evaluation" / "pre_0806_0019"
    / "evaluation_configs" / "frozen.json"
)


class FrozenCatalogTest(unittest.TestCase):
    def test_catalog_loads_51_lightweight_decks_and_one_policy(self) -> None:
        catalog = load_frozen_catalog(FROZEN_CATALOG, EVALUATION_ROOT)

        self.assertEqual(len(catalog.opponents), 51)
        self.assertEqual(len({item.deck_hash for item in catalog.opponents}), 51)
        self.assertEqual(catalog.pool_id, "0019_foundation_51_exact_decks_v4")
        self.assertEqual(catalog.policy.name, "frozen_foundation_policy")
        archived_frozen = (
            ROOT / "archive" / "evaluation" / "pre_0806_0019"
            / "evaluation_arena" / "frozen"
        )
        self.assertTrue(all(item.root.parent == archived_frozen for item in catalog.opponents))
        self.assertTrue(all(item.entrypoint == catalog.policy.entrypoint for item in catalog.opponents))

    def test_cli_defaults_to_frozen_0806_and_preserves_explicit_legacy_pool(self) -> None:
        frozen = subprocess.run(
            [sys.executable, "-m", "evaluation", "list-opponents"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        legacy = subprocess.run(
            [sys.executable, "-m", "evaluation", "--pool", "opponents", "list-opponents"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(frozen.returncode, 0, frozen.stderr)
        self.assertEqual(len(frozen.stdout.splitlines()), 55)
        self.assertIn("dragapult_ex_07bedfffbfad", frozen.stdout.splitlines())
        self.assertIn("mega_starmie_ex_dusknoir_b15853cdfb83", frozen.stdout.splitlines())
        self.assertIn("n_s_zoroark_ex_munkidori_940bc1f37b2a", frozen.stdout.splitlines())
        self.assertEqual(legacy.returncode, 0, legacy.stderr)
        self.assertEqual(len(legacy.stdout.splitlines()), 30)
        self.assertIn("alakazam_dudunsparce_01", legacy.stdout.splitlines())

    def test_frozen_candidate_uses_shared_runtime_and_exact_deck_identity(self) -> None:
        catalog = load_frozen_catalog(FROZEN_CATALOG, EVALUATION_ROOT)
        identity = next(
            item for item in catalog.opponents if item.name == "dragapult_ex_001"
        )

        candidate = catalog.candidate("dragapult_ex_001")

        self.assertEqual(candidate.name, identity.name)
        self.assertEqual(candidate.deck, identity.deck)
        self.assertEqual(candidate.deck_hash, identity.deck_hash)
        self.assertEqual(candidate.package_hash, identity.package_hash)
        self.assertEqual(candidate.root, catalog.policy.root)
        self.assertEqual(candidate.entrypoint, catalog.policy.entrypoint)
        with self.assertRaisesRegex(ValueError, "unknown Frozen candidate"):
            catalog.candidate("missing")


if __name__ == "__main__":
    unittest.main()
