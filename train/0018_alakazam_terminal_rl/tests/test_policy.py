from __future__ import annotations

import hashlib
import gzip
import json
import unittest

import torch

from ..constants import (
    REPOSITORY_ROOT,
    SOURCE_CHECKPOINT,
    TARGET_DECK,
    TARGET_DECK_PATH,
    TARGET_DECK_SHA256,
    TARGET_SOURCE_ID,
)
from ..policy.actor_critic import load_source_actor_critic
from ..policy.portable_inference import PortablePolicy, _fixed_config
from ..policy.action_distribution import evaluate_actions, sample_actions
from ..policy.online_runtime import OnlineCausalEncoder


class PolicyContractTest(unittest.TestCase):
    def test_portable_policy_reconstructs_frozen_source_contract(self) -> None:
        model_config, source_config = _fixed_config()
        self.assertEqual(model_config.ac.base.max_action_steps, 64)
        self.assertEqual(source_config.vocabulary_size, 25)
        self.assertEqual(source_config.initial_scale, 0.05)
        self.assertEqual(PortablePolicy.__init__.__kwdefaults__["source_id"], 0)

    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.model, cls.metadata = load_source_actor_critic()

    def test_source_actor_is_exact_r15_checkpoint(self) -> None:
        self.assertEqual(self.metadata["version"], "V2_win_multideck_r15_gradual_option")
        self.assertEqual(self.metadata["schema_version"], "0016_source_r15_training_v1")
        self.assertEqual(self.model.actor_parameter_count(), 17_601_282)
        self.assertEqual(
            hashlib.sha256(SOURCE_CHECKPOINT.read_bytes()).hexdigest(),
            "9db3b4d37a4d4f5cc483df1ac78ad4c5d39fed320c5325ebaa5fa61fd1114ebf",
        )

    def test_target_deck_is_exactly_frozen(self) -> None:
        self.assertEqual(len(TARGET_DECK), 60)
        self.assertEqual(
            hashlib.sha256(TARGET_DECK_PATH.read_bytes()).hexdigest(),
            TARGET_DECK_SHA256,
        )
        self.assertEqual(TARGET_SOURCE_ID, 0)

    def test_value_head_starts_neutral_and_actor_can_be_frozen(self) -> None:
        output = self.model.value_head[-2]
        self.assertTrue(torch.equal(output.weight, torch.zeros_like(output.weight)))
        self.assertTrue(torch.equal(output.bias, torch.zeros_like(output.bias)))
        self.model.freeze_actor()
        self.assertFalse(
            any(parameter.requires_grad for parameter in self.model.actor.parameters())
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in self.model.value_head.parameters()
            )
        )

    def test_sampled_action_log_probability_replays_exactly(self) -> None:
        raw_root = (
            REPOSITORY_ROOT
            / "rl_runs/0016_alakazam_multideck_bc/dataset/V1_win_multideck_raw"
        )
        row = None
        for path in sorted(raw_root.glob("*.jsonl.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    break
            if row is not None:
                break
        self.assertIsNotNone(row)
        observation = row["actor_observation"]
        actor = int(observation["current"]["yourIndex"])
        encoder = OnlineCausalEncoder(actor, TARGET_DECK, self.model.actor.config)
        batch = encoder.encode(observation)
        batch["source_id"] = torch.tensor([TARGET_SOURCE_ID])
        self.model.eval()
        sampled = sample_actions(self.model, batch)[0]
        width = max(1, len(sampled.indices))
        sequences = torch.full((1, width), -1, dtype=torch.long)
        if sampled.indices:
            sequences[0, : len(sampled.indices)] = torch.tensor(sampled.indices)
        replayed = evaluate_actions(
            self.model,
            batch,
            sequences,
            torch.tensor([len(sampled.indices)]),
            torch.tensor([sampled.stopped]),
        )
        self.assertAlmostEqual(
            sampled.log_prob, float(replayed.log_prob[0].detach()), places=5
        )
        self.assertGreaterEqual(len(sampled.indices), int(batch["min_count"][0]))
        self.assertLessEqual(len(sampled.indices), int(batch["max_count"][0]))


if __name__ == "__main__":
    unittest.main()
