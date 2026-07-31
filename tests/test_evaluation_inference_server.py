from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from evaluation.runner.batch import _candidate_socket_path
from evaluation.runner.inference_server import _normalize_request_deck, _stack_batches
from evaluation.runner.worker import _remote_policy_agent


class CandidateInferenceServerTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
