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
    MAX_POLICIES,
    PolicyPoolManifest,
    extend_normalized_actions_device,
    normalize_actions_device,
    route_torch,
)
from ptcg_cuda_engine.policy_adapters import StaticBatchFieldsDeviceAdapter  # noqa: E402


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

    def test_route_torch_supports_deck40_policy_ids(self) -> None:
        policy_ids = torch.arange(160, device="cuda") % 40
        ready = torch.ones(160, dtype=torch.bool, device="cuda")
        routes, route_mask, counts = route_torch(policy_ids, ready, 40, 4)
        self.assertEqual(tuple(routes.shape), (40, 4))
        self.assertTrue(bool(route_mask.all().item()))
        self.assertEqual(counts.tolist(), [4] * 40)
        torch.testing.assert_close(routes[39], torch.tensor([39, 79, 119, 159], device="cuda"))

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
                "max_policies": MAX_POLICIES,
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

    def test_pool_uses_action_contract_beyond_legacy_model_horizon(self) -> None:
        class SixteenStepAdapter:
            def act_device(self, batch):
                proposed = torch.arange(16, device="cuda").view(1, 16)
                lengths = torch.tensor([16], dtype=torch.long, device="cuda")
                return proposed, lengths

        manifest = PolicyPoolManifest.from_dict(
            {
                "name": "legacy_action_contract",
                "max_policies": 1,
                "policies": [
                    {
                        "policy_id": 0,
                        "name": "legacy_policy",
                        "deck": "unused.deck",
                        "checkpoint": "unused.pt",
                        "adapter": "legacy_adapter",
                        "codec": "legacy_codec",
                        "frozen": True,
                    }
                ],
            }
        )
        pool = GPUResidentPolicyPool(
            manifest,
            {0: SixteenStepAdapter()},
            capacity=1,
            max_select=24,
        )
        encoded = {
            "global_cat": torch.zeros(1, 1, dtype=torch.long, device="cuda"),
            "option_mask": torch.ones(1, 20, dtype=torch.bool, device="cuda"),
            "min_count": torch.tensor([16], dtype=torch.long, device="cuda"),
            "max_count": torch.tensor([16], dtype=torch.long, device="cuda"),
            "action_min_count": torch.tensor([20], dtype=torch.long, device="cuda"),
            "action_max_count": torch.tensor([20], dtype=torch.long, device="cuda"),
        }
        result = pool.act(
            {"legacy_codec": encoded},
            torch.zeros(1, dtype=torch.long, device="cuda"),
            torch.ones(1, dtype=torch.bool, device="cuda"),
        )
        self.assertEqual(result.lengths.tolist(), [20])
        self.assertEqual(result.indices[0, :20].tolist(), list(range(20)))

    def test_pool_batches_shared_static_foundation_slots_once(self) -> None:
        class SharedFakeAdapter:
            outputs_normalized = True

            def __init__(self) -> None:
                self.calls = 0
                self.deck_id_shape = None
                self.deck_id_rows = None

            @property
            def shared_batch_key(self):
                return ("shared_fake", id(self), 1)

            def act_device(self, batch):
                self.calls += 1
                self.deck_id_shape = tuple(batch["deck_ids"].shape)
                self.deck_id_rows = batch["deck_ids"].detach().cpu().tolist()
                proposed = batch["deck_scalar"].long().view(-1, 1)
                lengths = torch.ones(
                    proposed.shape[0], dtype=torch.long, device=proposed.device
                )
                return proposed, lengths

        policies = [
            {
                "policy_id": policy_id,
                "name": f"shared_policy_{policy_id}",
                "deck": f"deck_{policy_id}",
                "checkpoint": "unused.pt",
                "adapter": "shared_fake",
                "codec": "shared_codec",
                "frozen": True,
            }
            for policy_id in range(3)
        ]
        manifest = PolicyPoolManifest.from_dict(
            {
                "name": "shared_static_pool",
                "max_policies": MAX_POLICIES,
                "policies": policies,
            }
        )
        shared = SharedFakeAdapter()
        adapters = {
            policy_id: StaticBatchFieldsDeviceAdapter(
                shared,
                {
                    "deck_scalar": torch.tensor(
                        [policy_id + 1], dtype=torch.long, device="cuda"
                    ),
                    "deck_ids": torch.arange(
                        1,
                        policy_id + 3,
                        dtype=torch.long,
                        device="cuda",
                    ).view(1, -1),
                },
            )
            for policy_id in range(3)
        }
        pool = GPUResidentPolicyPool(
            manifest,
            adapters,
            capacity=4,
            max_select=4,
        )
        batch_size = 9
        encoded = {
            "global_cat": torch.arange(batch_size, device="cuda").view(-1, 1),
            "option_mask": torch.ones(batch_size, 5, dtype=torch.bool, device="cuda"),
            "min_count": torch.ones(batch_size, dtype=torch.long, device="cuda"),
            "max_count": torch.ones(batch_size, dtype=torch.long, device="cuda"),
        }
        profile = {}
        result = pool.act(
            {"shared_codec": encoded},
            torch.arange(batch_size, device="cuda") % 3,
            torch.ones(batch_size, dtype=torch.bool, device="cuda"),
            policy_profile=profile,
        )

        self.assertEqual(shared.calls, 1)
        self.assertEqual(profile["calls"], {"shared:shared_codec:3": 1})
        self.assertEqual(shared.deck_id_shape, (9, 4))
        self.assertEqual(shared.deck_id_rows[0], [1, 2, 0, 0])
        self.assertEqual(shared.deck_id_rows[1], [1, 2, 3, 0])
        self.assertEqual(shared.deck_id_rows[2], [1, 2, 3, 4])
        torch.testing.assert_close(
            result.indices[:, 0],
            (torch.arange(batch_size, device="cuda") % 3) + 1,
        )
        torch.testing.assert_close(
            result.lengths, torch.ones(batch_size, dtype=torch.long, device="cuda")
        )

    def test_normalized_fast_path_extends_legacy_action_contract(self) -> None:
        proposed = torch.arange(16, device="cuda").view(1, 16)
        lengths = torch.tensor([16], dtype=torch.long, device="cuda")
        option_mask = torch.ones(1, 20, dtype=torch.bool, device="cuda")
        action_min = torch.tensor([20], dtype=torch.long, device="cuda")
        action_max = torch.tensor([20], dtype=torch.long, device="cuda")
        actions, action_lengths = extend_normalized_actions_device(
            proposed,
            lengths,
            option_mask,
            action_min,
            action_max,
            max_select=24,
        )
        self.assertEqual(action_lengths.tolist(), [20])
        self.assertEqual(actions[0, :20].tolist(), list(range(20)))

    def test_pool_fast_path_uses_normalized_adapter_contract(self) -> None:
        class NormalizedSixteenStepAdapter:
            outputs_normalized = True

            def act_device(self, batch):
                proposed = torch.arange(16, device="cuda").view(1, 16)
                lengths = torch.tensor([16], dtype=torch.long, device="cuda")
                return proposed, lengths

        manifest = PolicyPoolManifest.from_dict(
            {
                "name": "legacy_action_contract_fast_path",
                "max_policies": 1,
                "policies": [
                    {
                        "policy_id": 0,
                        "name": "legacy_policy",
                        "deck": "unused.deck",
                        "checkpoint": "unused.pt",
                        "adapter": "legacy_adapter",
                        "codec": "legacy_codec",
                        "frozen": True,
                    }
                ],
            }
        )
        pool = GPUResidentPolicyPool(
            manifest,
            {0: NormalizedSixteenStepAdapter()},
            capacity=1,
            max_select=24,
        )
        encoded = {
            "global_cat": torch.zeros(1, 1, dtype=torch.long, device="cuda"),
            "option_mask": torch.ones(1, 20, dtype=torch.bool, device="cuda"),
            "min_count": torch.tensor([16], dtype=torch.long, device="cuda"),
            "max_count": torch.tensor([16], dtype=torch.long, device="cuda"),
            "action_min_count": torch.tensor([20], dtype=torch.long, device="cuda"),
            "action_max_count": torch.tensor([20], dtype=torch.long, device="cuda"),
        }
        result = pool.act(
            {"legacy_codec": encoded},
            torch.zeros(1, dtype=torch.long, device="cuda"),
            torch.ones(1, dtype=torch.bool, device="cuda"),
        )
        self.assertEqual(result.lengths.tolist(), [20])
        self.assertEqual(result.indices[0, :20].tolist(), list(range(20)))


if __name__ == "__main__":
    unittest.main()
