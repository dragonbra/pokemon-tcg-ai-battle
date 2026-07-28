from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from ..opponent_inference.resident_pool import (
    OpponentRequest,
    _add_deck_conditioning,
    _decoder_hidden,
    _id_only_collate,
    _selection_bounds,
)


class OpponentInferenceContractTest(unittest.TestCase):
    def test_deck_conditioning_uses_exact_runtime_deck(self) -> None:
        row: dict[str, object] = {}
        deck = [1] * 4 + list(range(2, 58))
        _add_deck_conditioning(row, deck)
        self.assertEqual(row["deck_ids"], deck)
        self.assertEqual(row["deck_counts"][:4], [4, 4, 4, 4])

    def test_id_only_collate_accepts_new_policy_module_api(self) -> None:
        expected = lambda rows: rows
        policy_module = SimpleNamespace(collate_examples=expected)
        package_module = SimpleNamespace(_load_policy_module=lambda: policy_module)
        self.assertIs(_id_only_collate(package_module), expected)

    def test_id_only_collate_accepts_legacy_train_module_api(self) -> None:
        expected = lambda rows: rows
        train_module = SimpleNamespace(collate_examples=expected)
        package_module = SimpleNamespace(_load_train_module=lambda: train_module)
        self.assertIs(_id_only_collate(package_module), expected)

    def test_layered_decoder_uses_last_hidden_slice_for_pointer_query(self) -> None:
        model = SimpleNamespace(
            config=SimpleNamespace(decoder_layers=2, d_model=3),
            decoder_init=nn.Linear(3, 6, bias=False),
        )
        state = torch.ones((1, 3))
        hidden, top = _decoder_hidden(model, state)
        self.assertEqual(tuple(hidden.shape), (1, 2, 3))
        self.assertTrue(torch.equal(top, hidden[:, -1]))

    def test_selection_bounds_come_from_runtime_observations(self) -> None:
        requests = [
            OpponentRequest("a", "p", {"select": {"minCount": 1, "maxCount": 3}}),
            OpponentRequest("b", "p", {"select": {"minCount": 0, "maxCount": 9}}),
        ]
        minimum, maximum = _selection_bounds(requests, torch.device("cpu"), 4)
        self.assertEqual(minimum.tolist(), [1, 0])
        self.assertEqual(maximum.tolist(), [3, 4])


if __name__ == "__main__":
    unittest.main()
