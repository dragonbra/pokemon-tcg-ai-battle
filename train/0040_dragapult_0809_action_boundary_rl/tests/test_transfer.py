from __future__ import annotations

import importlib
from pathlib import Path
import tempfile
import unittest

import torch


model_module = importlib.import_module("train.0040_dragapult_0809_action_boundary_rl.model")
transfer = importlib.import_module("train.0040_dragapult_0809_action_boundary_rl.transfer")


class WeightTransferTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.checkpoint = Path(
            "archive/pretrained/0031_friend_0809_gsb_v5_value_v9/model.pt"
        )

    def test_rejects_wrong_checkpoint_hash(self) -> None:
        model = model_module.PodNativeActorCritic()
        with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
            transfer.load_0031_initialization(model, self.checkpoint, expected_sha256="0" * 64)

    def test_real_transfer_is_explicit_and_elementwise_correct(self) -> None:
        torch.manual_seed(11)
        model = model_module.PodNativeActorCritic()
        before_value = model.value_head[-1].weight.detach().clone()
        report = transfer.load_0031_initialization(model, self.checkpoint)
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        source = payload["state_dict"]
        torch.testing.assert_close(
            model.action_decoder.query.weight,
            source["action_decoder.query.weight"],
        )
        rows = source["prototype_encoder.card_identity.weight"].shape[0]
        torch.testing.assert_close(
            model.entity_cat_embeddings.fields[0].weight[:rows],
            source["prototype_encoder.card_identity.weight"],
        )
        torch.testing.assert_close(model.value_head[-1].weight, before_value)
        self.assertFalse(report["strict_feature_parity_claimed"])
        self.assertGreater(report["transferred_parameter_count"], 0)
        self.assertLess(report["transferred_parameter_fraction"], 1.0)
        destinations = [row["destination"] for row in report["copied_tensors"]]
        self.assertEqual(len(destinations), len(set(destinations)))

    def test_rejects_missing_source_tensor(self) -> None:
        model = model_module.PodNativeActorCritic(
            model_module.ModelConfig(d_model=32, heads=4, state_layers=1, option_layers=1)
        )
        with self.assertRaisesRegex(KeyError, "required source tensor is missing"):
            transfer.transfer_state_dict(
                model, {}, source_checkpoint="synthetic", source_sha256="synthetic"
            )


if __name__ == "__main__":
    unittest.main()
