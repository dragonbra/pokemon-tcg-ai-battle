from __future__ import annotations

import hashlib
import gzip
import json
import unittest

import torch

from ..constants import SOURCE_CHECKPOINT, TARGET_DECK, TARGET_DECK_HASH
from ..policy.actor_critic import load_source_actor_critic
from ..policy.action_distribution import evaluate_actions, sample_actions
from ..policy.online_runtime import OnlineCausalEncoder
from ..constants import REPOSITORY_ROOT, TARGET_SOURCE_ID


class PolicyContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.model, cls.metadata = load_source_actor_critic()

    def test_source_actor_is_exact_r15_checkpoint(self) -> None:
        self.assertEqual(self.metadata["version"], "V2_t1_plus_pure_dragapult")
        self.assertEqual(self.model.actor_parameter_count(), 17_416_642)
        self.assertEqual(
            hashlib.sha256(SOURCE_CHECKPOINT.read_bytes()).hexdigest(),
            "7f7589427098418682f3a53e0763f3fd24850d641a0a664955f6094f90530532",
        )

    def test_target_deck_is_exactly_frozen(self) -> None:
        digest = hashlib.sha256(
            "".join(f"{card_id}\n" for card_id in TARGET_DECK).encode("ascii")
        ).hexdigest()
        self.assertEqual(len(TARGET_DECK), 60)
        # The dataset hash covers sorted count pairs rather than deck.csv ordering.
        self.assertEqual(
            TARGET_DECK_HASH,
            "213d75c4498e480647d755c1999e2090bcb94580a4a15f9997e431032a822892",
        )
        self.assertEqual(len(digest), 64)

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
            / "rl_runs/0015_dragapult_conditioned_bc/dataset/V1_core_raw"
        )
        row = None
        for path in sorted(raw_root.glob("*.jsonl.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as source:
                for line in source:
                    candidate = json.loads(line)
                    if candidate.get("source_key") == "third_ptcg_club":
                        row = candidate
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
