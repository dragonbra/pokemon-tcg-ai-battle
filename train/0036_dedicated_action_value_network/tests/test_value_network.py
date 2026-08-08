from __future__ import annotations
import unittest
import torch
from ..model.value_network import LatentQueryValueHead, MLPValueHead, masked_mean


class ValueNetworkTests(unittest.TestCase):
    def test_latent_head_shapes_and_gradients(self) -> None:
        head = LatentQueryValueHead(32, queries=8, layers=2, heads=4)
        memory = torch.randn(5, 17, 32)
        mask = torch.ones(5, 17, dtype=torch.bool)
        mask[:, -3:] = False
        outputs = head(memory, mask)
        self.assertEqual(outputs.value_logit.shape, (5,))
        self.assertEqual(outputs.archetype_logits.shape, (5, 15))
        self.assertEqual(outputs.final_diff_logits.shape, (5, 13))
        outputs.value_logit.sum().backward()
        self.assertIsNotNone(head.queries.grad)

    def test_mlp_head_shapes(self) -> None:
        outputs = MLPValueHead(16)(torch.randn(4, 16))
        self.assertEqual(outputs.value.shape, (4,))
        self.assertTrue(torch.all(outputs.value >= -1.0))
        self.assertTrue(torch.all(outputs.value <= 1.0))

    def test_masked_mean_ignores_padding(self) -> None:
        memory = torch.tensor([[[1.0], [3.0], [1000.0]]])
        mask = torch.tensor([[True, True, False]])
        self.assertEqual(float(masked_mean(memory, mask)[0, 0]), 2.0)


if __name__ == "__main__":
    unittest.main()
