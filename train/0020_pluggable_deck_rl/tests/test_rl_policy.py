from __future__ import annotations

import ast
import hashlib
from importlib import import_module
from pathlib import Path
import unittest

import torch


RL = import_module("train.0020_pluggable_deck_rl.rl")
CONSTANTS = import_module("train.0020_pluggable_deck_rl.rl.constants")
ACTOR_CRITIC = import_module("train.0020_pluggable_deck_rl.rl.policy.actor_critic")


class RLPolicyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.model, cls.metadata = ACTOR_CRITIC.load_source_actor_critic()

    def test_frozen_dragapult_identity(self) -> None:
        self.assertEqual(RL.PROJECT_ID, "0020_pluggable_deck_rl")
        self.assertEqual(len(CONSTANTS.TARGET_DECK), 60)
        self.assertEqual(CONSTANTS.TARGET_SOURCE_ID, 0)
        self.assertEqual(
            hashlib.sha256(CONSTANTS.TARGET_DECK_PATH.read_bytes()).hexdigest(),
            "00468e64b7c5eefb1cc4586ba9351a7f2a5bce223ef209d2b9e4b0bb0ca42e17",
        )

    def test_source_actor_is_exact_0019_epoch13(self) -> None:
        self.assertEqual(self.metadata["version"], "V1_universal_winner_r15")
        self.assertEqual(
            self.metadata["schema_version"],
            "0019_universal_winner_training_v1",
        )
        self.assertEqual(self.model.actor_parameter_count(), 17_756_162)
        self.assertEqual(
            hashlib.sha256(CONSTANTS.SOURCE_CHECKPOINT.read_bytes()).hexdigest(),
            "da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb",
        )

    def test_value_head_is_neutral_and_decoder_only_boundary_is_explicit(self) -> None:
        output = self.model.value_head[-2]
        self.assertTrue(torch.equal(output.weight, torch.zeros_like(output.weight)))
        self.assertTrue(torch.equal(output.bias, torch.zeros_like(output.bias)))
        self.model.unfreeze_decoder()
        trainable = {
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(any(name.startswith("value_head.") for name in trainable))
        self.assertTrue(any(name.startswith("actor.decoder.") for name in trainable))
        self.assertFalse(any("scenario_encoder" in name for name in trainable))

    def test_rl_runtime_has_no_other_numbered_project_imports(self) -> None:
        root = Path(RL.__file__).resolve().parent
        forbidden: list[tuple[str, str]] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                for name in names:
                    if name.startswith("train.00"):
                        forbidden.append((str(path), name))
        self.assertEqual(forbidden, [])


if __name__ == "__main__":
    unittest.main()
