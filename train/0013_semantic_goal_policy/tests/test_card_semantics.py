from __future__ import annotations

import csv
import importlib
import tempfile
import unittest
from pathlib import Path

semantics = importlib.import_module("train.0013_semantic_goal_policy.features.card_semantics")


class CardSemanticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = semantics.CardSemanticRegistry.from_official_csv(
            Path("data/official/EN_Card_Data.csv")
        )

    def test_multiline_card_rows_form_one_ordered_semantic_card(self) -> None:
        card = self.registry.require(30)
        self.assertEqual(card.name, "Magcargo ex")
        self.assertEqual(card.stage, "stage_1")
        self.assertEqual(card.hp.value, 270)
        self.assertEqual([move.name for move in card.moves], ["[Tera]", "Hot Magma", "Ground Burn"])
        self.assertIn("disruption", card.capabilities)
        self.assertIn("attacker", card.capabilities)

    def test_missing_unknown_not_applicable_and_padding_are_distinct(self) -> None:
        energy = self.registry.require(1)
        self.assertEqual(energy.hp.state, semantics.FieldState.NOT_APPLICABLE)
        self.assertEqual(self.registry.lookup(999999).identity_state, semantics.FieldState.UNKNOWN)
        self.assertNotEqual(semantics.UNK_CARD_ID, semantics.PADDING_CARD_ID)
        self.assertEqual(self.registry.padding.identity_state, semantics.FieldState.PADDING)

    def test_registry_hash_is_input_order_independent_and_conflicts_fail(self) -> None:
        with Path("data/official/EN_Card_Data.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        shuffled = list(reversed(rows))
        self.assertEqual(self.registry.sha256, semantics.CardSemanticRegistry.from_rows(shuffled).sha256)
        bad = [dict(rows[0]), dict(rows[0])]
        bad[1]["Card Name"] = "Different"
        with self.assertRaisesRegex(ValueError, "conflicting structural fields"):
            semantics.CardSemanticRegistry.from_rows(bad)

    def test_coverage_audit_counts_unknown_effect_without_losing_identity(self) -> None:
        audit = self.registry.audit_coverage([1, 30, 999999], [30])
        self.assertEqual(audit.identity_total, 4)
        self.assertEqual(audit.identity_known, 3)
        self.assertGreater(audit.effect_instances_total, 0)
        self.assertGreaterEqual(audit.effect_instances_known, 0)
        self.assertLess(audit.identity_coverage, 1.0)


if __name__ == "__main__":
    unittest.main()
