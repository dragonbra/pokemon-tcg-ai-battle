from __future__ import annotations

import unittest

import torch

from ..ablation import remove_initial_deck_information


class InitialDeckAblationTest(unittest.TestCase):
    def _batch(self) -> dict[str, torch.Tensor]:
        return {
            "registered_card_ids": torch.tensor([[121, 131, 133], [121, 131, 133]]),
            "registered_multiplicity": torch.tensor([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]]),
            "registered_mask": torch.ones((2, 3), dtype=torch.bool),
            "ledger_cat": torch.ones((2, 3, 4), dtype=torch.long),
            "ledger_num": torch.zeros((2, 3, 15)),
            "ledger_mask": torch.ones((2, 3), dtype=torch.bool),
            "source_id": torch.tensor([1, 2]),
            "zone_inventory_num": torch.randn(2, 2, 16),
        }

    def test_empty_visible_ledger_gets_one_safe_sentinel(self) -> None:
        batch = self._batch()
        result = remove_initial_deck_information(batch)
        self.assertTrue(
            torch.equal(result["registered_mask"].sum(1), torch.ones(2, dtype=torch.long))
        )
        self.assertTrue(
            torch.equal(
                result["registered_card_ids"],
                torch.zeros_like(batch["registered_card_ids"]),
            )
        )
        self.assertTrue(torch.equal(result["registered_mask"], result["ledger_mask"]))

    def test_visible_live_counts_survive_but_initial_bounds_do_not(self) -> None:
        batch = self._batch()
        batch["ledger_num"][0, 1, 2] = 1.0
        batch["ledger_num"][0, 1, 0] = 2.0
        batch["ledger_num"][0, 1, 8] = 1.0
        result = remove_initial_deck_information(batch)
        self.assertEqual(result["registered_card_ids"][0, 1].item(), 131)
        self.assertEqual(result["ledger_num"][0, 1, 2].item(), 1.0)
        self.assertEqual(result["ledger_num"][0, 1, 0].item(), 0.0)
        self.assertEqual(result["ledger_num"][0, 1, 8].item(), 0.0)
        self.assertTrue(torch.equal(result["source_id"], batch["source_id"]))
        self.assertTrue(torch.equal(result["zone_inventory_num"], batch["zone_inventory_num"]))


if __name__ == "__main__":
    unittest.main()
