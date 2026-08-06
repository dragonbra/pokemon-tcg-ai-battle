from __future__ import annotations

import importlib
import unittest

import torch


contract = importlib.import_module("train.0033_dragapult_third_ptcg_club_rl.contract")
PodNativeBatch = contract.PodNativeBatch


def valid_batch(batch_size: int = 2, device: str = "cpu") -> dict[str, torch.Tensor]:
    tensors = {
        "global_cat": torch.zeros(batch_size, 8, dtype=torch.long, device=device),
        "global_num": torch.zeros(batch_size, 16, device=device),
        "entity_cat": torch.zeros(batch_size, 128, 6, dtype=torch.long, device=device),
        "entity_num": torch.zeros(batch_size, 128, 10, device=device),
        "entity_parent": torch.full((batch_size, 128), -1, dtype=torch.long, device=device),
        "entity_mask": torch.zeros(batch_size, 128, dtype=torch.bool, device=device),
        "option_cat": torch.zeros(batch_size, 128, 12, dtype=torch.long, device=device),
        "option_num": torch.zeros(batch_size, 128, 4, device=device),
        "option_equiv": torch.full((batch_size, 128), -1, dtype=torch.long, device=device),
        "option_mask": torch.zeros(batch_size, 128, dtype=torch.bool, device=device),
        "min_count": torch.ones(batch_size, dtype=torch.long, device=device),
        "max_count": torch.ones(batch_size, dtype=torch.long, device=device),
    }
    tensors["entity_mask"][:, :3] = True
    tensors["option_mask"][:, :4] = True
    tensors["option_equiv"][:, :4] = torch.arange(4, device=device)
    return tensors


class PodNativeBatchTest(unittest.TestCase):
    def test_accepts_exact_contract(self) -> None:
        batch = PodNativeBatch.from_mapping(valid_batch())
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.option_count, 128)
        self.assertEqual(batch.device.type, "cpu")

    def test_rejects_missing_and_extra_fields(self) -> None:
        missing = valid_batch()
        del missing["option_equiv"]
        with self.assertRaisesRegex(ValueError, "missing=.*option_equiv"):
            PodNativeBatch.from_mapping(missing)
        extra = valid_batch()
        extra["source_id"] = torch.zeros(2, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "extra=.*source_id"):
            PodNativeBatch.from_mapping(extra)

    def test_rejects_wrong_shape_and_dtype(self) -> None:
        wrong_shape = valid_batch()
        wrong_shape["option_cat"] = torch.zeros(2, 127, 12, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "option_cat shape mismatch"):
            PodNativeBatch.from_mapping(wrong_shape)
        wrong_dtype = valid_batch()
        wrong_dtype["entity_mask"] = wrong_dtype["entity_mask"].long()
        with self.assertRaisesRegex(ValueError, "entity_mask must have dtype"):
            PodNativeBatch.from_mapping(wrong_dtype)

    def test_rejects_invalid_cpu_selection_and_parent_values(self) -> None:
        bounds = valid_batch()
        bounds["max_count"][0] = 5
        with self.assertRaisesRegex(ValueError, "max_count cannot exceed"):
            PodNativeBatch.from_mapping(bounds)
        parents = valid_batch()
        parents["entity_parent"][0, 0] = 128
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            PodNativeBatch.from_mapping(parents)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_batch_remains_on_device(self) -> None:
        batch = PodNativeBatch.from_mapping(valid_batch(device="cuda"))
        self.assertTrue(all(value.device.type == "cuda" for value in batch.values()))


if __name__ == "__main__":
    unittest.main()
