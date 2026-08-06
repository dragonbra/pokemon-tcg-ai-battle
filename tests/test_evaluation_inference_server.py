from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from evaluation.runner.batch import _candidate_socket_path
from evaluation.runner.inference_server import (
    _AbilityRepeatGuard,
    _apply_ability_repeat_guard,
    _inject_source_id_if_required,
    _normalize_request_deck,
    _resolve_inference_dtype,
    _stack_batches,
)
from evaluation.runner.worker import _remote_policy_agent


class CandidateInferenceServerTest(unittest.TestCase):
    def test_repeat_guard_ends_turn_after_eight_identical_ability_entries(self) -> None:
        observation = {
            "select": {
                "type": 0,
                "option": [
                    {"type": 10, "area": 5, "index": 1},
                    {"type": 13, "attackId": 120},
                    {"type": 14},
                ],
            },
            "current": {"turn": 10, "yourIndex": 1},
        }
        guard = _AbilityRepeatGuard(limit=8)
        for _ in range(8):
            self.assertEqual(_apply_ability_repeat_guard(observation, [0], guard), [0])
        self.assertEqual(_apply_ability_repeat_guard(observation, [0], guard), [2])

    def test_repeat_guard_does_not_change_attacks_or_disabled_contract(self) -> None:
        observation = {
            "select": {"type": 0, "option": [{"type": 13}, {"type": 14}]},
            "current": {"turn": 4, "yourIndex": 0},
        }
        self.assertEqual(
            _apply_ability_repeat_guard(observation, [0], _AbilityRepeatGuard(limit=8)),
            [0],
        )
        ability = {
            "select": {"type": 0, "option": [{"type": 10}, {"type": 14}]},
            "current": {"turn": 4, "yourIndex": 0},
        }
        guard = _AbilityRepeatGuard(limit=0)
        for _ in range(20):
            self.assertEqual(_apply_ability_repeat_guard(ability, [0], guard), [0])

    def test_source_identity_is_only_injected_for_declared_policies(self) -> None:
        canonical = type("Canonical", (), {"requires_source_id": False})()
        legacy = type("Legacy", (), {})()
        canonical_row = {"option_mask": torch.tensor([[True]])}
        legacy_row = {"option_mask": torch.tensor([[True]])}
        _inject_source_id_if_required(torch, canonical_row, canonical)
        _inject_source_id_if_required(torch, legacy_row, legacy)
        self.assertNotIn("source_id", canonical_row)
        self.assertEqual(legacy_row["source_id"].tolist(), [0])

    def test_canonical_policy_declares_fail_closed_inference(self) -> None:
        module = __import__(
            "train.0025_semantic_foundation_pretraining.deployment.canonical_inference",
            fromlist=["PortableCanonicalPolicy"],
        )
        self.assertTrue(module.PortableCanonicalPolicy.fail_closed_inference_errors)

    def test_remote_policy_agent_routes_exact_deck_and_role(self) -> None:
        class Connection:
            def __init__(self) -> None:
                self.requests = []

            def send(self, payload) -> None:
                self.requests.append(payload)

            @staticmethod
            def recv():
                return {"ok": True, "action": [7]}

        connection = Connection()
        deck = list(range(1, 61))
        with patch("evaluation.runner.worker.Client", return_value=connection):
            agent = _remote_policy_agent("/tmp/frozen.sock", deck=deck, role="opponent")
            self.assertEqual(agent({"select": None}), [7])

        self.assertEqual(
            connection.requests,
            [{"observation": {"select": None}, "deck": tuple(deck)}],
        )

    def test_remote_policy_agent_error_names_the_routed_role(self) -> None:
        class Connection:
            @staticmethod
            def send(_payload) -> None:
                return None

            @staticmethod
            def recv():
                return {"ok": False, "error": "boom"}

        with patch("evaluation.runner.worker.Client", return_value=Connection()):
            agent = _remote_policy_agent(
                "/tmp/frozen.sock", deck=list(range(1, 61)), role="opponent"
            )
            with self.assertRaisesRegex(RuntimeError, "shared opponent inference"):
                agent({"select": None})

    def test_socket_path_fits_linux_af_unix_limit(self) -> None:
        path = _candidate_socket_path()
        self.assertEqual(path.parent, Path(tempfile.gettempdir()))
        self.assertLessEqual(len(str(path).encode()), 107)
        self.assertFalse(path.exists())
        self.assertNotEqual(path, _candidate_socket_path())

    def test_stack_batches_pads_variable_tensor_axes(self) -> None:
        rows = [
            {
                "option_mask": torch.tensor([[True, True]]),
                "option_cat": torch.tensor([[[1, 2], [3, 4]]]),
                "targets": torch.tensor([[0, 2]]),
            },
            {
                "option_mask": torch.tensor([[True]]),
                "option_cat": torch.tensor([[[5, 6]]]),
                "targets": torch.tensor([[0]]),
            },
        ]
        batch = _stack_batches(torch, rows, torch.device("cpu"))
        self.assertEqual(tuple(batch["option_cat"].shape), (2, 2, 2))
        self.assertEqual(batch["option_mask"].tolist(), [[True, True], [True, False]])
        self.assertEqual(batch["targets"].tolist(), [[0, 2], [0, -100]])

    def test_request_deck_is_exact_60_positive_integer_cards(self) -> None:
        expected = tuple(range(1, 61))
        self.assertEqual(_normalize_request_deck(list(expected), (999,) * 60), expected)
        self.assertEqual(_normalize_request_deck(None, expected), expected)
        for invalid in ([1] * 59, [1] * 61, [0] * 60, [True] * 60, "1,2"):
            with self.subTest(invalid=type(invalid).__name__, length=len(invalid)):
                with self.assertRaisesRegex(ValueError, "60 positive integer"):
                    _normalize_request_deck(invalid, expected)

    def test_inference_dtype_requires_supported_cuda_precision(self) -> None:
        self.assertIs(
            _resolve_inference_dtype(torch, "fp32", torch.device("cpu")),
            torch.float32,
        )
        self.assertIs(
            _resolve_inference_dtype(torch, "fp16", torch.device("cuda:0")),
            torch.float16,
        )
        with self.assertRaisesRegex(ValueError, "requires a CUDA device"):
            _resolve_inference_dtype(torch, "fp16", torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "unsupported inference dtype"):
            _resolve_inference_dtype(torch, "tf32", torch.device("cuda:0"))


if __name__ == "__main__":
    unittest.main()
