from __future__ import annotations

import importlib
import unittest

import torch


exporter = importlib.import_module(
    "train.0037_dragapult_value_initialized_rl.export_full_semantic_candidate"
)


class ExportFullSemanticCandidateTest(unittest.TestCase):
    def test_merge_qv_weight_changes_only_q_and_v_rows(self) -> None:
        width = 3
        base = torch.arange(27, dtype=torch.float32).reshape(9, 3)
        q_a = torch.tensor([[1.0, 2.0, 3.0]])
        q_b = torch.tensor([[1.0], [2.0], [3.0]])
        v_a = torch.tensor([[2.0, 1.0, 0.5]])
        v_b = torch.tensor([[3.0], [2.0], [1.0]])

        merged = exporter._merge_qv_weight(
            base,
            q_a=q_a,
            q_b=q_b,
            v_a=v_a,
            v_b=v_b,
            alpha=2.0,
            rank=1,
        )

        torch.testing.assert_close(
            merged[:width], base[:width] + (q_b @ q_a) * 2.0
        )
        self.assertTrue(torch.equal(merged[width : 2 * width], base[width : 2 * width]))
        torch.testing.assert_close(
            merged[2 * width :], base[2 * width :] + (v_b @ v_a) * 2.0
        )
        self.assertTrue(torch.equal(base, torch.arange(27, dtype=torch.float32).reshape(9, 3)))

    def test_merge_qv_weight_rejects_malformed_factors(self) -> None:
        base = torch.zeros(9, 3)
        good_a = torch.zeros(1, 3)
        good_b = torch.zeros(3, 1)
        with self.assertRaises(ValueError):
            exporter._merge_qv_weight(
                base,
                q_a=torch.zeros(2, 3),
                q_b=good_b,
                v_a=good_a,
                v_b=good_b,
                alpha=2.0,
                rank=1,
            )
        with self.assertRaises(ValueError):
            exporter._merge_qv_weight(
                torch.zeros(8, 3),
                q_a=good_a,
                q_b=good_b,
                v_a=good_a,
                v_b=good_b,
                alpha=2.0,
                rank=1,
            )


if __name__ == "__main__":
    unittest.main()
