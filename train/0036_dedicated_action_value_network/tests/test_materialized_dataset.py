from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ..training.dataset import ValueBatch
from ..training.materialized import (
    _compact_value_batch,
    _load_shard,
    _runtime_value_batch,
    _sha256,
)


class MaterializedValueDatasetTests(unittest.TestCase):
    def _batch(self) -> ValueBatch:
        return ValueBatch(
            features={
                "global_cat": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
                "global_num": torch.tensor([[0.25, 1.5], [2.0, -0.5]], dtype=torch.float32),
                "card_mask": torch.tensor([[True, False], [True, True]]),
                "targets": torch.tensor([[0, -100], [1, 2]], dtype=torch.long),
            },
            value_target=torch.tensor([0.0, 1.0]),
            archetype_target=torch.tensor([3, 14]),
            final_diff_target=torch.tensor([0, 12]),
            episode_weight=torch.tensor([0.125, 1.0 / 73.0]),
            is_exact_007=torch.tensor([True, False]),
        )

    def test_compact_roundtrip_preserves_contract_and_labels(self) -> None:
        original = self._batch()
        restored = _runtime_value_batch(_compact_value_batch(original))
        self.assertTrue(torch.equal(restored.features["global_cat"], original.features["global_cat"]))
        self.assertTrue(torch.equal(restored.features["targets"], original.features["targets"]))
        self.assertTrue(torch.equal(restored.features["card_mask"], original.features["card_mask"]))
        self.assertTrue(torch.allclose(restored.features["global_num"], original.features["global_num"]))
        self.assertTrue(torch.equal(restored.value_target, original.value_target))
        self.assertTrue(torch.equal(restored.archetype_target, original.archetype_target))
        self.assertTrue(torch.equal(restored.final_diff_target, original.final_diff_target))
        self.assertTrue(torch.equal(restored.episode_weight, original.episode_weight))
        self.assertTrue(torch.equal(restored.is_exact_007, original.is_exact_007))

    def test_compaction_rejects_values_outside_int16(self) -> None:
        batch = self._batch()
        batch.features["global_cat"][0, 0] = 32768
        with self.assertRaisesRegex(ValueError, "int16"):
            _compact_value_batch(batch)

    def test_load_shard_rejects_commitment_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train-00000.pt"
            torch.save(_compact_value_batch(self._batch()), path)
            expected = {"sha256": "0" * 64, "bytes": path.stat().st_size, "count": 2}
            with self.assertRaisesRegex(ValueError, "commitment"):
                _load_shard(path, expected, verify_commitment=True)
            expected["sha256"] = _sha256(path)
            loaded = _load_shard(path, expected, verify_commitment=True)
            self.assertEqual(loaded["value_target"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
