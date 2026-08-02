from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

import torch


BASE = importlib.import_module(
    "train.0025_semantic_foundation_pretraining.legacy.base_model"
)
DEPLOY = importlib.import_module(
    "train.0025_semantic_foundation_pretraining.deployment.portable_inference"
)
EXPORT = importlib.import_module(
    "train.0025_semantic_foundation_pretraining.export_candidate"
)


def _observation(option_count: int, minimum: int, maximum: int) -> dict:
    return {
        "current": {
            "yourIndex": 0,
            "firstPlayer": 0,
            "turn": 3,
            "players": [
                {"active": [{"id": 1, "hp": 100, "maxHp": 100}], "bench": [], "hand": [], "discard": [], "prize": [], "deckCount": 40},
                {"active": [{"id": 3, "hp": 100, "maxHp": 100}], "bench": [], "discard": [], "prize": [], "deckCount": 40},
            ],
        },
        "select": {
            "type": 1,
            "context": 0,
            "minCount": minimum,
            "maxCount": maximum,
            "option": [
                {"type": 1, "playerIndex": 0, "area": 4, "index": 0, "number": index}
                for index in range(option_count)
            ],
        },
    }


class DeploymentTests(unittest.TestCase):
    def test_batched_decode_matches_single_observation_decode(self) -> None:
        torch.manual_seed(7)
        config = BASE.IDOnlyConfig(d_model=32, heads=4, encoder_layers=1)
        model = DEPLOY.InferenceIDOnlyPointerPolicy(config).eval()
        codec = BASE.IDOnlyCodec(config)
        rows = []
        for observation, minimum, maximum in (
            (_observation(3, 0, 2), 0, 2),
            (_observation(4, 1, 3), 1, 3),
        ):
            row = codec.encode(observation, list(range(minimum)))
            self.assertIsNotNone(row)
            rows.append(row)
        batch = BASE.collate_id_only(rows)
        with torch.inference_mode():
            decoded = model.deterministic_action_tensors(batch)
        for index, row in enumerate(rows):
            single = BASE.collate_id_only([row])
            expected = model.greedy_action(single)
            actual = decoded.sequences[index, : decoded.lengths[index]].tolist()
            self.assertTrue(bool(decoded.legal[index]))
            self.assertEqual(actual, expected)

    def test_export_is_model_only_and_self_contained(self) -> None:
        config = BASE.IDOnlyConfig(d_model=32, heads=4, encoder_layers=1)
        model = BASE.IDOnlyPointerPolicy(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            torch.save(
                {
                    "schema_version": "0025_model_only_checkpoint_v1",
                    "state_dict": model.state_dict(),
                    "metadata": {
                        "arm": "legacy_default",
                        "epoch": 1,
                        "model_config": {"config": config.to_dict()},
                        "validation": {"loss": 1.0},
                    },
                },
                checkpoint,
            )
            deck = root / "deck.csv"
            deck.write_text("1\n" * 60, encoding="ascii")
            cg = root / "cg_source"
            cg.mkdir()
            (cg / "api.py").write_text("", encoding="ascii")
            output = root / "candidate"
            manifest = EXPORT.export_candidate(
                checkpoint=checkpoint,
                deck_path=deck,
                cg_source=cg,
                output=output,
                deck_id="test_deck",
            )
            packaged = torch.load(
                output / "strategy/model.bin", map_location="cpu", weights_only=True
            )
            self.assertTrue(manifest["model_only"])
            self.assertEqual(
                set(packaged), {"schema_version", "state_dict", "metadata"}
            )
            self.assertFalse(any(path.is_symlink() for path in output.rglob("*")))
            self.assertEqual(len((output / "deck.csv").read_text().splitlines()), 60)


if __name__ == "__main__":
    unittest.main()
