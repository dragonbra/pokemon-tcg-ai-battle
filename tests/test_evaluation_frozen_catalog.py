from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from evaluation.frozen import load_frozen_catalog


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = ROOT / "evaluation"
FROZEN_CATALOG = EVALUATION_ROOT / "configs" / "frozen.json"


class FrozenCatalogTest(unittest.TestCase):
    def test_catalog_loads_48_lightweight_decks_and_one_policy(self) -> None:
        catalog = load_frozen_catalog(FROZEN_CATALOG, EVALUATION_ROOT)

        self.assertEqual(len(catalog.opponents), 48)
        self.assertEqual(len({item.deck_hash for item in catalog.opponents}), 48)
        self.assertEqual(catalog.pool_id, "0019_foundation_48_exact_decks_v1")
        self.assertEqual(catalog.policy.name, "frozen_foundation_policy")
        self.assertTrue(all(item.root.parent == EVALUATION_ROOT / "arena" / "frozen" for item in catalog.opponents))
        self.assertTrue(all(item.entrypoint == catalog.policy.entrypoint for item in catalog.opponents))

    def test_cli_defaults_to_frozen_and_preserves_explicit_legacy_pool(self) -> None:
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
        self.assertEqual(len(frozen.stdout.splitlines()), 48)
        self.assertIn("dragapult_ex_001", frozen.stdout.splitlines())
        self.assertEqual(legacy.returncode, 0, legacy.stderr)
        self.assertEqual(len(legacy.stdout.splitlines()), 30)
        self.assertIn("alakazam_dudunsparce_01", legacy.stdout.splitlines())


if __name__ == "__main__":
    unittest.main()
