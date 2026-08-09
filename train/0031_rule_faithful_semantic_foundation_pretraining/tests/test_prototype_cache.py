from __future__ import annotations

import importlib
import unittest
from unittest import mock

import torch


PROJECT = "train.0031_rule_faithful_semantic_foundation_pretraining"
MODEL = importlib.import_module(f"{PROJECT}.model")
DOMAIN = importlib.import_module(f"{PROJECT}.domain.prototypes")


def tiny_policy():
    return MODEL.SemanticPolicy(
        MODEL.ModelConfig(
            d_model=32,
            heads=4,
            state_layers=1,
            event_layers=1,
            option_layers=1,
            dropout=0.0,
            max_card_id=8,
            max_attack_id=8,
            max_skill_id=8,
            max_effect_id=8,
            max_options=8,
            max_action_steps=4,
        ),
        DOMAIN.PrototypeIndex.empty(),
    )


class PrototypeCacheTest(unittest.TestCase):
    def test_eval_builds_one_detached_non_checkpoint_cache(self) -> None:
        policy = tiny_policy().eval()
        keys_before = tuple(policy.state_dict())
        with mock.patch.object(
            policy.prototype_encoder,
            "encode_all",
            wraps=policy.prototype_encoder.encode_all,
        ) as encode_all:
            first = policy.prototype_memory()
            second = policy.prototype_memory()
        self.assertEqual(encode_all.call_count, 1)
        self.assertIs(first, second)
        for tensor in (first.cards, first.attacks, first.skills, first.effects):
            self.assertFalse(tensor.requires_grad)
        self.assertEqual(tuple(policy.state_dict()), keys_before)
        self.assertEqual(
            policy.prototype_cache_stats(),
            {"builds": 1, "hits": 1, "invalidations": 0},
        )

    def test_device_and_state_load_invalidate_cache(self) -> None:
        policy = tiny_policy().eval()
        policy.prototype_memory()
        policy.to(dtype=torch.float64)
        self.assertEqual(policy.prototype_cache_stats()["invalidations"], 1)
        policy.prototype_memory()
        state = {name: value.clone() for name, value in policy.state_dict().items()}
        policy.load_state_dict(state, strict=True)
        self.assertEqual(policy.prototype_cache_stats()["invalidations"], 2)

    def test_trainable_prototypes_bypass_but_frozen_prototypes_cache(self) -> None:
        trainable = tiny_policy().train()
        with mock.patch.object(
            trainable.prototype_encoder,
            "encode_all",
            wraps=trainable.prototype_encoder.encode_all,
        ) as encode_all:
            trainable.prototype_memory()
            trainable.prototype_memory()
        self.assertEqual(encode_all.call_count, 2)
        self.assertEqual(trainable.prototype_cache_stats()["builds"], 0)

        frozen = tiny_policy()
        frozen.prototype_encoder.requires_grad_(False)
        frozen.train()
        with mock.patch.object(
            frozen.prototype_encoder,
            "encode_all",
            wraps=frozen.prototype_encoder.encode_all,
        ) as encode_all:
            first = frozen.prototype_memory()
            second = frozen.prototype_memory()
        self.assertEqual(encode_all.call_count, 1)
        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
