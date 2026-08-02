from __future__ import annotations

from collections import Counter
import importlib
import tempfile
import unittest
from pathlib import Path

import torch


BASE = "train.0025_semantic_foundation_pretraining"
DEPLOY = importlib.import_module(f"{BASE}.deployment.canonical_inference")
ONLINE = importlib.import_module(f"{BASE}.deployment.canonical_online_runtime")
MODEL = importlib.import_module(f"{BASE}.model.canonical")
PROTOTYPES = importlib.import_module(f"{BASE}.features.prototypes")
EXPORT = importlib.import_module(f"{BASE}.export_candidate")


def _pokemon(card_id: int, player: int) -> dict:
    return {
        "id": card_id,
        "playerIndex": player,
        "serial": card_id + player * 10000,
        "hp": 200,
        "maxHp": 200,
        "appearThisTurn": False,
        "energyCards": [],
        "tools": [],
        "preEvolution": [],
    }


def _observation() -> dict:
    players = []
    for player, card_id in ((0, 96), (1, 63)):
        players.append(
            {
                "active": [_pokemon(card_id, player)],
                "bench": [],
                "hand": [],
                "discard": [],
                "prize": [None] * 6,
                "deckCount": 40,
                "handCount": 0,
                "asleep": False,
                "burned": False,
                "confused": False,
                "paralyzed": False,
                "poisoned": False,
            }
        )
    return {
        "current": {
            "yourIndex": 0,
            "firstPlayer": 0,
            "turn": 3,
            "turnActionCount": 1,
            "players": players,
            "stadium": [],
            "looking": [],
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
        },
        "select": {
            "type": 0,
            "context": 0,
            "contextCard": None,
            "effect": None,
            "deck": [],
            "option": [{"type": 13, "attackId": 120}, {"type": 14}],
            "minCount": 1,
            "maxCount": 1,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
        },
        "logs": [],
    }


class CanonicalDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prototypes = PROTOTYPES.PrototypeIndex.load(
            Path(f"{BASE.replace('.', '/')}/assets/official_public_prototypes_v1.json")
        )

    def test_online_batch_has_exact_canonical_actor_contract(self) -> None:
        config = MODEL.CanonicalModelConfig(
            d_model=32, heads=4, state_layers=1, option_layers=1, ffn_multiplier=2
        )
        encoder = ONLINE.OnlineCausalEncoder(0, [1] * 60, config)
        batch = encoder.encode(_observation())
        self.assertNotIn("legacy", batch)
        self.assertNotIn("source_id", batch)
        self.assertEqual(batch["option_cat"].shape[1], 2)
        self.assertEqual(batch["min_count"].tolist(), [1])

    def test_checkpoint_strict_load_and_legal_selection(self) -> None:
        torch.manual_seed(3)
        config = MODEL.CanonicalModelConfig(
            d_model=32, heads=4, state_layers=1, option_layers=1, ffn_multiplier=2,
            dropout=0.0,
        )
        model = MODEL.CanonicalSemanticPolicy(config, self.prototypes)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save(
                {
                    "schema_version": "0025_model_only_checkpoint_v1",
                    "state_dict": model.state_dict(),
                    "metadata": {
                        "arm": "canonical_semantic",
                        "epoch": 1,
                        "model_config": {"config": config.to_dict()},
                        "validation": {"loss": 1.0},
                    },
                },
                checkpoint,
            )
            policy = DEPLOY.PortableCanonicalPolicy.from_checkpoint(checkpoint, [1] * 60)
            action = policy.select(_observation())
            self.assertEqual(len(action), 1)
            self.assertIn(action[0], (0, 1))
            policy.reset()
            self.assertIsNone(policy.encoder)

    def test_canonical_export_is_self_contained(self) -> None:
        config = MODEL.CanonicalModelConfig(
            d_model=32, heads=4, state_layers=1, option_layers=1, ffn_multiplier=2
        )
        model = MODEL.CanonicalSemanticPolicy(config, self.prototypes)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            torch.save(
                {
                    "schema_version": "0025_model_only_checkpoint_v1",
                    "state_dict": model.state_dict(),
                    "metadata": {
                        "arm": "canonical_semantic",
                        "epoch": 1,
                        "model_config": {"config": config.to_dict()},
                        "validation": {"loss": 1.0},
                    },
                }, checkpoint,
            )
            deck = root / "deck.csv"
            deck.write_text("1\n" * 60, encoding="ascii")
            cg = root / "cg"
            cg.mkdir()
            (cg / "__init__.py").write_text("", encoding="ascii")
            output = root / "candidate"
            manifest = EXPORT.export_candidate(
                checkpoint=checkpoint,
                deck_path=deck,
                cg_source=cg,
                output=output,
                deck_id="test",
            )
            self.assertEqual(manifest["arm"], "canonical_semantic")
            self.assertTrue((output / "strategy/deployment/official_full_engine_prototypes_v1.json").is_file())
            self.assertFalse(any(path.is_symlink() for path in output.rglob("*")))
            self.assertEqual(Counter(map(int, (output / "deck.csv").read_text().split())), Counter({1: 60}))


if __name__ == "__main__":
    unittest.main()
