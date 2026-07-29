from __future__ import annotations

import sys
import unittest
from pathlib import Path


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))

try:
    import torch
except ImportError:
    torch = None

from ptcg_cuda_engine.policy_pool import (  # noqa: E402
    GPUResidentPolicyPool,
    PolicyPoolManifest,
    normalize_actions_device,
    route_torch,
)


CUDA_AVAILABLE = torch is not None and torch.cuda.is_available()


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA PyTorch is not available")
class TorchPolicyPoolTest(unittest.TestCase):
    def test_route_torch_keeps_fixed_device_cohorts(self) -> None:
        policy_ids = torch.arange(120, device="cuda") % 12
        ready = torch.ones(120, dtype=torch.bool, device="cuda")
        routes, route_mask, counts = route_torch(policy_ids, ready, 12, 10)
        self.assertTrue(routes.is_cuda)
        self.assertEqual(tuple(routes.shape), (12, 10))
        self.assertTrue(bool(route_mask.all().item()))
        self.assertEqual(counts.tolist(), [10] * 12)
        expected = torch.arange(0, 120, 12, device="cuda")
        torch.testing.assert_close(routes[0], expected)

    def test_device_normalizer_removes_duplicates_and_fills_minimum(self) -> None:
        proposed = torch.tensor(
            [[2, 2, 8, -1], [3, -1, -1, -1]],
            dtype=torch.long,
            device="cuda",
        )
        lengths = torch.tensor([4, 1], dtype=torch.long, device="cuda")
        option_mask = torch.tensor(
            [[1, 1, 1, 1, 0], [1, 0, 1, 0, 1]],
            dtype=torch.bool,
            device="cuda",
        )
        minimum = torch.tensor([2, 2], dtype=torch.long, device="cuda")
        maximum = torch.tensor([3, 2], dtype=torch.long, device="cuda")
        actions, action_lengths = normalize_actions_device(
            proposed,
            lengths,
            option_mask,
            minimum,
            maximum,
            max_select=4,
        )
        self.assertTrue(actions.is_cuda)
        self.assertEqual(action_lengths.tolist(), [2, 2])
        self.assertEqual(actions[0, :2].tolist(), [2, 0])
        self.assertEqual(actions[1, :2].tolist(), [0, 2])

    def test_twelve_heterogeneous_slots_dispatch_without_host_counts(self) -> None:
        class FakeAdapter:
            def act_device(self, batch):
                proposed = (batch["global_cat"][:, :1] % 5).long()
                lengths = torch.ones(
                    proposed.shape[0], dtype=torch.long, device=proposed.device
                )
                return proposed, lengths

        policies = [
            {
                "policy_id": policy_id,
                "name": f"policy_{policy_id}",
                "deck": f"deck_{policy_id}",
                "checkpoint": f"unused_{policy_id}.pt",
                "adapter": f"adapter_family_{policy_id % 4}",
                "codec": "test_codec",
                "frozen": policy_id != 11,
                "dtype": "bf16",
                "parameter_mib": 1.0,
            }
            for policy_id in range(12)
        ]
        manifest = PolicyPoolManifest.from_dict(
            {
                "name": "test_pool",
                "max_policies": 32,
                "policies": policies,
            }
        )
        pool = GPUResidentPolicyPool(
            manifest,
            {policy_id: FakeAdapter() for policy_id in range(12)},
            capacity=10,
            max_select=4,
        )
        batch_size = 120
        global_cat = torch.arange(batch_size, device="cuda").view(-1, 1)
        encoded = {
            "global_cat": global_cat,
            "option_mask": torch.ones(batch_size, 5, dtype=torch.bool, device="cuda"),
            "min_count": torch.ones(batch_size, dtype=torch.long, device="cuda"),
            "max_count": torch.ones(batch_size, dtype=torch.long, device="cuda"),
        }
        result = pool.act(
            {"test_codec": encoded},
            torch.arange(batch_size, device="cuda") % 12,
            torch.ones(batch_size, dtype=torch.bool, device="cuda"),
        )
        self.assertTrue(result.indices.is_cuda)
        torch.testing.assert_close(result.indices[:, 0], torch.arange(batch_size, device="cuda") % 5)
        torch.testing.assert_close(
            result.lengths, torch.ones(batch_size, dtype=torch.long, device="cuda")
        )
        self.assertEqual(result.route_counts.tolist(), [10] * 12)


if __name__ == "__main__":
    unittest.main()
