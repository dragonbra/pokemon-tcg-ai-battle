from __future__ import annotations

import importlib
from pathlib import Path
import tempfile
import unittest

import torch

HELPERS = importlib.import_module(
    "train.0026_raging_bolt_canonical_decoder_rl.tests.helpers"
)
BASE = HELPERS.BASE
observation = HELPERS.observation


class PolicyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = importlib.import_module(f"{BASE}.policy")
        cls.distribution = importlib.import_module(f"{BASE}.policy.action_distribution")
        cls.online = importlib.import_module(f"{BASE}.focal.deployment.canonical_online_runtime")
        cls.checkpoint = importlib.import_module(f"{BASE}.checkpoint")
        cls.league = importlib.import_module(f"{BASE}.league")

    def _model_batch(self):
        model, _ = self.policy.load_actor_critic("cpu")
        deck = next(
            item.deck for item in self.league.load_frozen_catalog()
            if item.deck_id == "raging_bolt_ex_james_cox_henry_chao_001"
        )
        encoder = self.online.OnlineCausalEncoder(0, deck, model.actor.config)
        return model, encoder.encode(observation())

    def test_only_action_decoder_and_value_head_are_trainable(self) -> None:
        model, _ = self.policy.load_actor_critic("cpu")
        model.assert_trainable_contract()
        names = model.trainable_parameter_names()
        self.assertTrue(names)
        self.assertTrue(all(
            name.startswith("actor.action_decoder.") or name.startswith("value_head.")
            for name in names
        ))
        self.assertFalse(any(name.startswith("actor.option_encoder.") for name in names))
        self.assertFalse(any(name.startswith("actor.state_encoder.") for name in names))

    def test_optimizer_step_changes_decoder_not_representation(self) -> None:
        model, batch = self._model_batch()
        before_representation = model.representation_sha256()
        before_decoder = model.decoder_sha256()
        sampled = self.distribution.greedy_actions(model, batch)
        sequences = torch.tensor([sampled[0].indices], dtype=torch.long)
        lengths = torch.tensor([len(sampled[0].indices)], dtype=torch.long)
        stopped = torch.tensor([sampled[0].stopped], dtype=torch.bool)
        evaluated = self.distribution.evaluate_actions(model, batch, sequences, lengths, stopped)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4
        )
        loss = -evaluated.log_prob.mean() + evaluated.value.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        self.assertEqual(model.representation_sha256(), before_representation)
        self.assertNotEqual(model.decoder_sha256(), before_decoder)

    def test_decoder_checkpoint_round_trip_is_model_only(self) -> None:
        model, _ = self._model_batch()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "update-1.pt"
            self.checkpoint.save_model_checkpoint(path, model, policy_version="V1_test", update=1)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            forbidden = {"optimizer", "scheduler", "scaler", "rng", "rollout", "replay"}
            self.assertFalse(forbidden & set(payload))
            restored, _ = self.policy.load_actor_critic("cpu")
            audit = self.checkpoint.load_model_checkpoint(path, restored)
            self.assertEqual(audit["decoder_sha256"], model.decoder_sha256())


if __name__ == "__main__":
    unittest.main()
