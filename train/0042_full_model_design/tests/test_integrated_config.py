from __future__ import annotations

import importlib
import unittest

import torch


config = importlib.import_module("train.0042_full_model_design.integrated.config")
losses = importlib.import_module("train.0042_full_model_design.integrated.loss_registry")
presets = importlib.import_module("train.0042_full_model_design.integrated.presets")


class IntegratedConfigTest(unittest.TestCase):
    def test_all_0042_presets_validate(self) -> None:
        self.assertEqual(set(presets.PRESETS), {"BASE", "FULL_MODEL"})
        for value in presets.PRESETS.values():
            value.validate()
        self.assertFalse(presets.preset("FULL_MODEL").enable_tempo_aux_loss)

    def test_rejects_contract_mixing(self) -> None:
        with self.assertRaises(ValueError):
            config.IntegratedFlags(enable_action_boundary=False).validate()
        with self.assertRaises(ValueError):
            config.IntegratedFlags(enable_prize_aux=True, prize_aux_mode="off").validate()

    def test_loss_registry_has_independent_terms_and_groups(self) -> None:
        registry = losses.LossRegistry()
        registry.register(losses.LossTerm("L_policy_win", torch.tensor(2.0), 1.0, "actor"))
        registry.register(losses.LossTerm("L_value_prize", torch.tensor(3.0), 0.5, "prize", False))
        self.assertEqual(float(registry.total()), 2.0)
        self.assertEqual(registry.optimizer_groups(), {"actor": ("L_policy_win",)})
        self.assertEqual(registry.metrics()["loss_weight/L_value_prize"], 0.0)


if __name__ == "__main__":
    unittest.main()
