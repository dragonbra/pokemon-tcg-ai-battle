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

    def test_lora_targets_only_last_option_block_q_and_v(self) -> None:
        model = self._load(False)
        inventory = model.adaptation_inventory
        self.assertEqual(inventory["option_block"], 1)
        self.assertEqual(inventory["attention_targets"], ["self_attn.qv", "cross_attn.qv"])
        self.assertEqual(inventory["adapter_parameters"], 10_240)
        self.assertEqual(inventory["layernorm_parameters"], 0)
        names = [name for name, value in model.named_parameters() if value.requires_grad]
        adapter_names = [name for name in names if ".parametrizations." in name]
        self.assertTrue(adapter_names)
        self.assertTrue(all("cross_attention_transformer.layers.1" in name for name in adapter_names))
        self.assertTrue(all("in_proj_weight" in name for name in adapter_names))
        self.assertFalse(any("out_proj" in name or "linear1" in name or "linear2" in name for name in adapter_names))
        self.assertFalse(any("board_encoder" in name and ".parametrizations." in name for name in names))
        self.assertFalse(any("prototype_encoder" in name for name in names))
        for name, value in model.named_parameters():
            if ".parametrizations." in name and name.endswith((".q_b", ".v_b")):
                self.assertTrue(torch.count_nonzero(value) == 0, name)

        parity = importlib.import_module(f"{PROJECT}.parity")
        base, _ = self.policy.load_actor_critic(
            self.run_module.SOURCE_CHECKPOINT, self.run_module.focal_deck(), "cpu"
        )
        adapted_base = parity._base_actor_state(model.actor.state_dict())
        self.assertEqual(set(adapted_base), set(base.actor.state_dict()))
        for name, value in base.actor.state_dict().items():
            self.assertTrue(torch.equal(adapted_base[name], value), name)

    def test_layernorm_arm_tunes_only_option_output_norm(self) -> None:
        model = self._load(True)
        self.assertEqual(model.adaptation_inventory["layernorm_parameters"], 640)
        names = [name for name, value in model.named_parameters() if value.requires_grad]
        norm_names = model.adaptation_inventory["layernorm_parameter_names"]
        self.assertEqual(norm_names, [
            "option_encoder.cross_attention_transformer.norm.weight",
            "option_encoder.cross_attention_transformer.norm.bias",
        ])
        self.assertFalse(any("state_encoder" in name and value.requires_grad for name, value in model.named_parameters()))
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
