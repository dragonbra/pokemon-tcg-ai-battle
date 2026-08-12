from __future__ import annotations

import unittest

import torch

from ptcg_cuda_engine.semantic0031_resident import (
    ResidentActionBypass,
    merge_resident_actions,
    select_semantic_rows,
)


class ResidentActionAdapterTest(unittest.TestCase):
    def test_select_semantic_rows_preserves_non_batch_metadata(self) -> None:
        batch = {
            "global_cat": torch.arange(12).reshape(3, 4),
            "option_mask": torch.tensor([[1, 0], [1, 1], [0, 0]], dtype=torch.bool),
            "schema": "semantic0031",
        }
        rows = torch.tensor([1, 0])

        selected = select_semantic_rows(batch, rows, batch_size=3)

        self.assertEqual(selected["schema"], "semantic0031")
        torch.testing.assert_close(selected["global_cat"], batch["global_cat"][rows])
        torch.testing.assert_close(selected["option_mask"], batch["option_mask"][rows])

    def test_merge_bypass_and_policy_actions(self) -> None:
        bypass = ResidentActionBypass(
            actions=torch.tensor([[0, 0], [3, 0], [0, 0]]),
            lengths=torch.tensor([0, 1, 0]),
            bypass_mask=torch.tensor([False, True, False]),
            forced_mask=torch.tensor([False, True, False]),
            macro_mask=torch.zeros(3, dtype=torch.bool),
        )
        policy_rows = torch.tensor([0, 2])
        actions, lengths = merge_resident_actions(
            bypass=bypass,
            policy_rows=policy_rows,
            policy_actions=torch.tensor([[2, 1], [4, 0]]),
            policy_lengths=torch.tensor([2, 1]),
            lane_count=3,
            action_width=2,
            device=torch.device("cpu"),
        )

        self.assertEqual(actions.tolist(), [[2, 1], [3, 0], [4, 0]])
        self.assertEqual(lengths.tolist(), [2, 1, 1])

    def test_overlapping_bypass_masks_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "forced and macro"):
            ResidentActionBypass(
                actions=torch.zeros((1, 1), dtype=torch.long),
                lengths=torch.ones(1, dtype=torch.long),
                bypass_mask=torch.ones(1, dtype=torch.bool),
                forced_mask=torch.ones(1, dtype=torch.bool),
                macro_mask=torch.ones(1, dtype=torch.bool),
            ).validate(lane_count=1)


if __name__ == "__main__":
    unittest.main()
