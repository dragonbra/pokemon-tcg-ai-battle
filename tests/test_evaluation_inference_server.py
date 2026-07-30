from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from evaluation.runner.batch import _candidate_socket_path
from evaluation.runner.inference_server import _stack_batches


class CandidateInferenceServerTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
