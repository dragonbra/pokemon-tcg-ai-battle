from __future__ import annotations

import unittest

from rl.core.promotion import has_minimum_evaluation_coverage, is_promotable
from rl.core.reward import RewardProfile, potential_difference, terminal_reward
from rl.demo.toy_env import LEFT, RIGHT, ToyLineEnv
from rl.ptcg.features import PTCGFeatureConfig, encode_observation

try:
    import torch

    from rl.core.batch import collate_encoded
    from rl.core.losses import masked_cross_entropy
    from rl.core.model import CandidatePolicyValueNet, ModelConfig
except ImportError:  # pragma: no cover - exercised by environments without torch
    torch = None


class RLFrameworkTests(unittest.TestCase):
    def test_toy_environment_exposes_only_legal_actions(self) -> None:
        env = ToyLineEnv(target=2)
        env.reset(position=0)
        self.assertNotIn(LEFT, env.legal_actions())
        self.assertIn(RIGHT, env.legal_actions())
        with self.assertRaises(ValueError):
            env.step(LEFT)

    def test_reward_profile_is_explicit_and_potential_shaping_is_local(self) -> None:
        profile = RewardProfile(name="test", win=1.0, loss=-1.0)
        self.assertEqual(terminal_reward(0, 0, profile), 1.0)
        self.assertEqual(terminal_reward(1, 0, profile), -1.0)
        self.assertEqual(terminal_reward(2, 0, profile), 0.0)
        self.assertAlmostEqual(potential_difference(0.2, 0.8, 0.9), 0.52)

    def test_promotion_requires_primary_gain_and_guardrails(self) -> None:
        champion = {"win_rate": 0.50}
        self.assertTrue(
            is_promotable(
                {"win_rate": 0.52, "invalid_action_rate": 0.0, "error_rate": 0.0},
                champion,
            )
        )
        self.assertFalse(
            is_promotable(
                {"win_rate": 0.52, "invalid_action_rate": 0.01, "error_rate": 0.0},
                champion,
            )
        )

    def test_promotion_requires_seventeen_opponents_and_ten_games_each(self) -> None:
        summary = {"by_opponent": {str(index): {"games": 10} for index in range(17)}}
        self.assertTrue(has_minimum_evaluation_coverage(summary))
        summary["by_opponent"]["0"] = {"games": 9}
        self.assertFalse(has_minimum_evaluation_coverage(summary))

    def test_ptcg_encoder_has_stable_shapes(self) -> None:
        config = PTCGFeatureConfig()
        observation = {
            "current": {
                "turn": 2,
                "yourIndex": 0,
                "firstPlayer": 0,
                "players": [
                    {
                        "active": [{"id": 741, "hp": 60, "energies": [5]}],
                        "bench": [{"id": 741}],
                        "hand": [{"id": 742}],
                        "discard": [],
                        "deckCount": 40,
                        "handCount": 1,
                        "prize": [{"id": 741}] * 6,
                    },
                    {
                        "active": [{"id": 1072, "hp": 120}],
                        "bench": [],
                        "hand": [],
                        "discard": [],
                        "deckCount": 42,
                        "handCount": 4,
                        "prize": [{"id": 1}] * 6,
                    },
                ],
            },
            "select": {
                "type": 0,
                "context": 0,
                "minCount": 1,
                "maxCount": 1,
                "option": [
                    {"type": 13, "attackId": 1072},
                    {"type": 8, "cardId": 5, "inPlayArea": 4, "inPlayIndex": 0},
                ],
            },
        }
        encoded = encode_observation(observation, config)
        self.assertEqual(len(encoded["state_numeric"]), config.state_numeric_dim)
        self.assertEqual(len(encoded["state_card_ids"]), config.state_token_count)
        self.assertEqual(len(encoded["action_mask"]), config.max_candidates)
        self.assertEqual(sum(encoded["action_mask"]), 2)

    @unittest.skipUnless(torch is not None, "PyTorch is an optional RL dependency")
    def test_collate_encoded_uses_expected_dtypes(self) -> None:
        sample = {
            "state_numeric": [0.0] * 24,
            "state_card_ids": [0] * 24,
            "action_type_ids": [1] + [0] * 63,
            "action_card_ids": [0] * 64,
            "action_target_ids": [0] * 64,
            "action_numeric": [[0.0] * 10 for _ in range(64)],
            "action_mask": [True] + [False] * 63,
        }
        batch = collate_encoded([sample])
        self.assertEqual(batch["action_mask"].dtype, torch.bool)
        self.assertEqual(batch["action_type_ids"].dtype, torch.long)
        self.assertEqual(batch["state_numeric"].dtype, torch.float32)

    @unittest.skipUnless(torch is not None, "PyTorch is an optional RL dependency")
    def test_model_masks_padding_and_computes_loss(self) -> None:
        config = ModelConfig(
            state_numeric_dim=2,
            state_token_count=1,
            candidate_numeric_dim=2,
            max_candidates=3,
            card_vocab_size=4,
            action_type_vocab_size=3,
            d_model=16,
            hidden_dim=32,
            num_heads=2,
        )
        model = CandidatePolicyValueNet(config)
        inputs = {
            "state_numeric": torch.zeros((1, 2)),
            "state_card_ids": torch.zeros((1, 1), dtype=torch.long),
            "action_type_ids": torch.tensor([[1, 2, 0]]),
            "action_card_ids": torch.zeros((1, 3), dtype=torch.long),
            "action_target_ids": torch.zeros((1, 3), dtype=torch.long),
            "action_numeric": torch.zeros((1, 3, 2)),
            "action_mask": torch.tensor([[True, True, False]]),
        }
        value, logits = model(**inputs)
        self.assertEqual(tuple(value.shape), (1,))
        self.assertEqual(tuple(logits.shape), (1, 3))
        self.assertLess(logits[0, 2].item(), -1e10)
        loss = masked_cross_entropy(logits, torch.tensor([1]), inputs["action_mask"])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    @unittest.skipUnless(torch is not None, "PyTorch is an optional RL dependency")
    def test_ptcg_inference_bridge_returns_a_legal_option(self) -> None:
        from rl.ptcg.inference import PTCGCandidatePolicy
        from tempfile import TemporaryDirectory

        feature_config = PTCGFeatureConfig()
        model_config = ModelConfig(
            state_numeric_dim=feature_config.state_numeric_dim,
            state_token_count=feature_config.state_token_count,
            candidate_numeric_dim=feature_config.candidate_numeric_dim,
            max_candidates=feature_config.max_candidates,
            card_vocab_size=feature_config.card_vocab_size,
            action_type_vocab_size=feature_config.action_type_vocab_size,
            d_model=16,
            hidden_dim=32,
            num_heads=2,
        )
        model = CandidatePolicyValueNet(model_config)
        policy = PTCGCandidatePolicy(model, feature_config=feature_config)
        observation = {
            "current": {
                "turn": 2,
                "yourIndex": 0,
                "firstPlayer": 0,
                "players": [{"active": [{"id": 741}], "bench": []}, {"active": [], "bench": []}],
            },
            "select": {
                "type": 0,
                "context": 0,
                "minCount": 1,
                "maxCount": 1,
                "option": [{"type": 13}, {"type": 14}],
            },
        }
        index, value = policy.select(observation)
        self.assertIn(index, (0, 1))
        self.assertTrue(-1.0 <= value <= 1.0)
        _, _, confidence = policy.select_with_confidence(observation)
        self.assertTrue(0.0 <= confidence <= 1.0)

        old_feature_config = PTCGFeatureConfig(
            state_numeric_dim=24,
            state_token_count=24,
        )
        old_model_config = ModelConfig(
            state_numeric_dim=old_feature_config.state_numeric_dim,
            state_token_count=old_feature_config.state_token_count,
            candidate_numeric_dim=old_feature_config.candidate_numeric_dim,
            max_candidates=old_feature_config.max_candidates,
            card_vocab_size=old_feature_config.card_vocab_size,
            action_type_vocab_size=old_feature_config.action_type_vocab_size,
            d_model=16,
            hidden_dim=32,
            num_heads=2,
        )
        old_model = CandidatePolicyValueNet(old_model_config)
        with TemporaryDirectory() as directory:
            checkpoint = f"{directory}/old.pt"
            torch.save(
                {
                    "model": old_model.state_dict(),
                    "metadata": {
                        "model_config": old_model_config.to_dict(),
                        "feature_config": old_feature_config.__dict__,
                    },
                },
                checkpoint,
            )
            restored = PTCGCandidatePolicy.from_checkpoint(checkpoint)
        self.assertEqual(restored.feature_config.state_token_count, 24)


if __name__ == "__main__":
    unittest.main()
