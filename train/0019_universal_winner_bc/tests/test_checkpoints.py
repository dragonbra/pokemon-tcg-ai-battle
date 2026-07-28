from __future__ import annotations

from importlib import import_module
import tempfile
import unittest
from pathlib import Path

import torch


CHECKPOINTS = import_module("train.0019_universal_winner_bc.training.checkpoints")


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_is_model_only(self) -> None:
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            manifest = CHECKPOINTS.save_checkpoint(
                Path(directory),
                model=model,
                optimizer=optimizer,
                epoch=1,
                global_step=2,
                metadata={"project_id": "0019_universal_winner_bc"},
                criteria=("latest",),
            )
            payload = torch.load(
                Path(directory) / manifest["path"], map_location="cpu", weights_only=True
            )
        self.assertEqual(set(payload), {"model", "epoch", "global_step", "metadata"})
        self.assertFalse(manifest["optimizer_state_saved"])
        self.assertFalse(manifest["resumable_training_state_saved"])


if __name__ == "__main__":
    unittest.main()
