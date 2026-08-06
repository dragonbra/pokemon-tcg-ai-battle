from __future__ import annotations

import importlib
from pathlib import Path
import tempfile
import unittest

import torch


PROJECT = "train.0033_dragapult_third_ptcg_club_rl"
ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "archive/pretrained/0031_friend_pt0805_epoch13_best_validation_loss/model.pt"
DECK = ROOT / "train/0033_dragapult_third_ptcg_club_rl/league/decks/dragapult_third_ptcg_club/deck.csv"


class FullSemanticPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        policy = importlib.import_module(f"{PROJECT}.policy")
        cls.deck = tuple(map(int, DECK.read_text().splitlines()))
        cls.model, cls.identity = policy.load_actor_critic(CHECKPOINT, cls.deck, "cpu")

    def test_exact_pt0805_inventory_and_trainable_boundary(self) -> None:
        self.assertEqual(self.identity.actor_parameter_count, 56_352_322)
        self.assertEqual(self.identity.actor_tensor_count, 293)
        self.assertEqual(len(self.model.actor.state_dict()), 293)
        trainable = [name for name, value in self.model.named_parameters() if value.requires_grad]
        self.assertTrue(any(name.startswith("actor.action_decoder.") for name in trainable))
        self.assertTrue(any(name.startswith("value_head.") for name in trainable))
        self.assertFalse(any("state_encoder" in name for name in trainable))
        self.assertFalse(any("option_encoder" in name for name in trainable))
        self.assertFalse(any("prototype_encoder" in name for name in trainable))

    def test_model_only_checkpoint_excludes_recovery_and_frozen_state(self) -> None:
        storage = importlib.import_module(f"{PROJECT}.training.storage_full_semantic")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "update.pt"
            digest = storage.save_model_only(
                self.model,
                path,
                update=0,
                metadata={"source_checkpoint_sha256": self.identity.checkpoint_sha256},
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(len(digest), 64)
            self.assertFalse(storage.FORBIDDEN_KEYS & set(payload))
            self.assertTrue(all(
                name.startswith(("action_decoder.", "value_head."))
                for name in payload["state_dict"]
            ))
            self.assertFalse(any("state_encoder" in name for name in payload["state_dict"]))


class FullSchemaTest(unittest.TestCase):
    def test_exact_full_schema_inventory(self) -> None:
        parity = importlib.import_module(f"{PROJECT}.parity")
        fields = importlib.import_module(f"{PROJECT}.semantic_policy.contracts.fields")
        parity.assert_full_schema()
        self.assertEqual(fields.SCHEMA_VERSION, "0031_rule_faithful_semantic_decision_v2")
        self.assertEqual(len(fields.ACTOR_KEYS | fields.MASK_KEYS | {"targets"}), 39)
        self.assertEqual(parity.EXPECTED_WIDTHS["resource_num"], 15)
        self.assertEqual(parity.EXPECTED_WIDTHS["event_cat"], 31)
        self.assertEqual(parity.EXPECTED_WIDTHS["option_cat"], 19)


if __name__ == "__main__":
    unittest.main()
