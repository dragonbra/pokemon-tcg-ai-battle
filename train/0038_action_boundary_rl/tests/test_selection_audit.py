from __future__ import annotations

import importlib
import unittest


audit = importlib.import_module("train.0038_action_boundary_rl.selection_audit")


class SelectionAuditTest(unittest.TestCase):
    def test_count_manifest_hash_is_order_invariant_and_exact(self) -> None:
        deck = list(range(30)) * 2
        self.assertEqual(
            audit.count_manifest_sha256(deck), audit.count_manifest_sha256(reversed(deck))
        )
        with self.assertRaisesRegex(ValueError, "60"):
            audit.count_manifest_sha256(deck[:-1])


if __name__ == "__main__":
    unittest.main()
