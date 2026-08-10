from __future__ import annotations

import importlib
from pathlib import Path
import tempfile
import unittest

import torch


storage = importlib.import_module("train.0042_full_model_design.training.storage")
model_module = importlib.import_module("train.0042_full_model_design.model")
ModelConfig = model_module.ModelConfig
PodNativeActorCritic = model_module.PodNativeActorCritic


class StorageTest(unittest.TestCase):
    def test_round_trip_is_model_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "update-000001.pt"
            model = torch.nn.Linear(3, 2)
            storage.save_model_only(model, path, update=1, metadata={"source": "epoch2"})
            payload = storage.load_model_only(path)
            self.assertEqual(payload["update"], 1)
            self.assertEqual(set(payload), {"schema_version", "update", "state_dict", "metadata"})
            self.assertTrue(path.with_suffix(".pt.sha256").is_file())

    def test_trainable_checkpoint_excludes_frozen_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "update-000001.pt"
            source = PodNativeActorCritic(ModelConfig())
            storage.save_trainable_heads(
                source,
                path,
                update=1,
                metadata={"foundation": "epoch2"},
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(
                payload["schema_version"], storage.TRAINABLE_HEADS_SCHEMA
            )
            self.assertTrue(payload["state_dict"])
            self.assertTrue(
                all(
                    name.startswith("action_decoder.")
                    or name.startswith("value_head.")
                    for name in payload["state_dict"]
                )
            )
            target = PodNativeActorCritic(ModelConfig())
            storage.load_trainable_heads(target, path)
            for name, value in payload["state_dict"].items():
                self.assertTrue(torch.equal(target.state_dict()[name], value))


if __name__ == "__main__":
    unittest.main()
