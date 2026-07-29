from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import torch


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))

from ptcg_cuda_engine.policy_adapters import (  # noqa: E402
    EntityPointerPolicyV1DeviceAdapter,
    IDOnlyPointerPolicyDeviceAdapter,
)


class FakePointerPolicy:
    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size, option_count = batch["option_mask"].shape
        options = torch.arange(
            option_count,
            dtype=torch.float32,
            device=batch["option_mask"].device,
        )
        option_repr = options.view(1, option_count, 1).expand(batch_size, -1, -1)
        return {
            "option_repr": option_repr,
            "state_repr": torch.zeros(
                (batch_size, 1),
                dtype=torch.float32,
                device=option_repr.device,
            ),
        }

    def initial_decoder_state(self, state_repr: torch.Tensor) -> list[torch.Tensor]:
        return [state_repr]

    def pointer_logits(
        self,
        option_repr: torch.Tensor,
        hidden: list[torch.Tensor],
        valid_options: torch.Tensor,
        stop_valid: torch.Tensor,
    ) -> torch.Tensor:
        del hidden
        options = option_repr.squeeze(-1).masked_fill(~valid_options, -1.0e9)
        stop = torch.where(
            stop_valid,
            torch.full_like(stop_valid, 100, dtype=option_repr.dtype),
            torch.full_like(stop_valid, -1.0e9, dtype=option_repr.dtype),
        )
        return torch.cat([options, stop.unsqueeze(1)], dim=1)

    def advance_decoder(
        self,
        selected_repr: torch.Tensor,
        hidden: list[torch.Tensor],
        update_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        return [torch.where(update_mask.unsqueeze(1), selected_repr, hidden[0])]


class EntityPointerPolicyV1DeviceAdapterTest(unittest.TestCase):
    def test_greedy_decode_and_route_padding_stay_on_device(self) -> None:
        batch = {
            "option_mask": torch.tensor(
                [[True, True, True], [True, True, False], [True, False, False]]
            ),
            "min_count": torch.tensor([2, 1, 0]),
            "max_count": torch.tensor([3, 2, 2]),
            "_route_mask": torch.tensor([True, False, True]),
        }
        adapter = EntityPointerPolicyV1DeviceAdapter(FakePointerPolicy(), max_select=4)

        actions, lengths = adapter.act_device(batch)

        self.assertEqual(actions.device, batch["option_mask"].device)
        self.assertEqual(lengths.device, batch["option_mask"].device)
        self.assertEqual(actions.tolist(), [[2, 1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
        self.assertEqual(lengths.tolist(), [2, 0, 0])

    def test_hot_path_has_no_host_tensor_materialization(self) -> None:
        source = inspect.getsource(EntityPointerPolicyV1DeviceAdapter.act_device)
        for forbidden in (".cpu(", ".item(", ".tolist(", "bool(active.any())"):
            self.assertNotIn(forbidden, source)


class FakeIDOnlyConfig:
    d_model = 1
    decoder_layers = 1


class FakeIDOnlyPolicy:
    config = FakeIDOnlyConfig()

    def encode(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, option_count = batch["option_mask"].shape
        options = torch.arange(
            option_count,
            dtype=torch.float32,
            device=batch["option_mask"].device,
        ).view(1, option_count, 1).expand(batch_size, -1, -1)
        return torch.zeros((batch_size, 1), device=options.device), options

    def pointer_key(self, options: torch.Tensor) -> torch.Tensor:
        return options

    def decoder_init(self, state: torch.Tensor) -> torch.Tensor:
        return state + 1

    def pointer_query(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden

    def option_bias(self, options: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(options)

    def stop(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.full_like(hidden, 100)

    def decoder(self, selected: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return selected + hidden


class IDOnlyPointerPolicyDeviceAdapterTest(unittest.TestCase):
    def test_greedy_decode_uses_min_max_and_route_mask(self) -> None:
        batch = {
            "option_mask": torch.tensor(
                [[True, True, True], [True, True, False], [True, False, False]]
            ),
            "min_count": torch.tensor([2, 1, 0]),
            "max_count": torch.tensor([3, 2, 2]),
            "_route_mask": torch.tensor([True, False, True]),
        }
        adapter = IDOnlyPointerPolicyDeviceAdapter(FakeIDOnlyPolicy(), max_select=4)

        actions, lengths = adapter.act_device(batch)

        self.assertEqual(actions.tolist(), [[2, 1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
        self.assertEqual(lengths.tolist(), [2, 0, 0])

    def test_hot_path_has_no_host_tensor_materialization(self) -> None:
        source = inspect.getsource(IDOnlyPointerPolicyDeviceAdapter.act_device)
        for forbidden in (".cpu(", ".item(", ".tolist(", "bool(active.any())"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
