from __future__ import annotations

import importlib
import tempfile
from pathlib import Path
import unittest

import torch


PROJECT = "train.0037_dragapult_value_initialized_rl"


class AdaptationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        policy = importlib.import_module(f"{PROJECT}.policy")
        run = importlib.import_module(f"{PROJECT}.training.run_full_semantic")
        cls.policy = policy
        cls.run_module = run

    def _load(self, layernorm: bool):
        return self.policy.load_actor_critic(
            self.run_module.SOURCE_CHECKPOINT,
            self.run_module.focal_deck(),
            "cpu",
            adaptation=self.policy.AdaptationConfig(
                lora=True, layernorm_tuning=layernorm
            ),
        )[0]

    def test_lora_targets_first_two_and_last_board_layers(self) -> None:
        model = self._load(False)
        inventory = model.adaptation_inventory
        self.assertEqual(inventory["board_layers"], [0, 1, 3])
        self.assertEqual(inventory["adapter_parameters"], 107_520)
        self.assertEqual(inventory["layernorm_parameters"], 0)
        names = [name for name, value in model.named_parameters() if value.requires_grad]
        self.assertFalse(any("board_encoder.layers.2" in name for name in names))
        self.assertFalse(any("prototype_encoder" in name for name in names))
        for name, value in model.named_parameters():
            if ".parametrizations." in name and name.endswith(".b"):
                self.assertTrue(torch.count_nonzero(value) == 0, name)

        parity = importlib.import_module(f"{PROJECT}.parity")
        base, _ = self.policy.load_actor_critic(
            self.run_module.SOURCE_CHECKPOINT, self.run_module.focal_deck(), "cpu"
        )
        adapted_base = parity._base_actor_state(model.actor.state_dict())
        self.assertEqual(set(adapted_base), set(base.actor.state_dict()))
        for name, value in base.actor.state_dict().items():
            self.assertTrue(torch.equal(adapted_base[name], value), name)

    def test_layernorm_arm_excludes_shared_prototype_encoder(self) -> None:
        model = self._load(True)
        self.assertEqual(model.adaptation_inventory["layernorm_parameters"], 12_160)
        names = [name for name, value in model.named_parameters() if value.requires_grad]
        self.assertTrue(any("state_encoder" in name and "norm" in name for name in names))
        self.assertTrue(any("option_encoder" in name and "norm" in name for name in names))
        self.assertFalse(any("prototype_encoder" in name for name in names))

    def test_adapted_checkpoint_contains_only_trainable_tensors(self) -> None:
        storage = importlib.import_module(f"{PROJECT}.training.storage_full_semantic")
        model = self._load(True)
        expected = {name for name, value in model.named_parameters() if value.requires_grad}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "update.pt"
            storage.save_model_only(model, path, update=1, metadata={})
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["schema_version"], "0037_value_initialized_adapted_model_only_v2")
            self.assertEqual(set(payload["state_dict"]), expected)
            self.assertFalse(any(name.endswith(".original") for name in payload["state_dict"]))
            target = self._load(True)
            storage.load_adapted_model_only(target, path)
            for name, value in payload["state_dict"].items():
                self.assertTrue(torch.equal(dict(target.named_parameters())[name], value), name)


if __name__ == "__main__":
    unittest.main()
