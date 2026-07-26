from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch

from archive.train_legacy.alakazam_bc_rl.id_only_pointer import (
    IDOnlyCodec as ReferenceCodec,
    IDOnlyConfig as ReferenceConfig,
    IDOnlyPointerPolicy as ReferencePolicy,
)
_config = importlib.import_module("train.0010_alakazam_sota_model.config")
_dataset = importlib.import_module("train.0010_alakazam_sota_model.dataset")
_model = importlib.import_module("train.0010_alakazam_sota_model.model")
_training = importlib.import_module("train.0010_alakazam_sota_model.training")
TrainingConfig = _config.TrainingConfig
build_id_only_dataset = _dataset.build_id_only_dataset
IDOnlyCodec = _model.IDOnlyCodec
IDOnlyConfig = _model.IDOnlyConfig
IDOnlyPointerPolicy = _model.IDOnlyPointerPolicy
collate_id_only = _model.collate_id_only
train = _training.train


def _observation() -> dict[str, object]:
    return {
        "current": {
            "yourIndex": 0,
            "firstPlayer": 0,
            "turn": 3,
            "turnActionCount": 4,
            "supporterPlayed": True,
            "players": [
                {
                    "active": [
                        {
                            "id": 741,
                            "hp": 60,
                            "maxHp": 60,
                            "energyCards": [{"id": 5}],
                            "tools": [],
                            "preEvolution": [],
                        }
                    ],
                    "bench": [{"id": 305, "hp": 50, "maxHp": 50}],
                    "hand": [{"id": 743}, {"id": 1079}],
                    "discard": [{"id": 1120}],
                    "deckCount": 42,
                    "handCount": 2,
                    "prize": [1, 2, 3, 4, 5, 6],
                },
                {
                    "active": [{"id": 306, "hp": 120, "maxHp": 150}],
                    "bench": [],
                    "discard": [],
                    "deckCount": 44,
                    "handCount": 6,
                    "prize": [1, 2, 3, 4, 5, 6],
                },
            ],
            "stadium": [],
            "looking": [],
        },
        "select": {
            "type": 1,
            "context": 3,
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {"type": 2, "playerIndex": 0, "area": 2, "index": 0, "cardId": 743},
                {"type": 3, "playerIndex": 0, "area": 2, "index": 1, "cardId": 1079},
            ],
        },
    }


class AlakazamSotaModelTests(unittest.TestCase):
    def test_codec_matches_frozen_reference(self) -> None:
        config = IDOnlyConfig(d_model=32, heads=4, encoder_layers=1)
        reference = ReferenceCodec(ReferenceConfig(**config.to_dict()))
        expected = reference.encode(_observation(), [1])
        actual = IDOnlyCodec(config).encode(_observation(), [1])
        self.assertEqual(actual, expected)
        batch = collate_id_only([actual])  # type: ignore[list-item]
        self.assertEqual(tuple(batch["global_cat"].shape), (1, 4))
        self.assertEqual(tuple(batch["entity_cat"].shape[2:]), (7,))
        self.assertEqual(tuple(batch["option_cat"].shape), (1, 2, 12))
        self.assertEqual(batch["targets"].tolist(), [[1, 2]])

    def test_model_initialization_and_logits_match_reference(self) -> None:
        config = IDOnlyConfig(d_model=32, heads=4, encoder_layers=1, dropout=0.0)
        torch.manual_seed(17)
        actual = IDOnlyPointerPolicy(config)
        torch.manual_seed(17)
        expected = ReferencePolicy(ReferenceConfig(**config.to_dict()))
        self.assertEqual(actual.state_dict().keys(), expected.state_dict().keys())
        for name, tensor in actual.state_dict().items():
            self.assertTrue(torch.equal(tensor, expected.state_dict()[name]), name)
        row = IDOnlyCodec(config).encode(_observation(), [1])
        batch = collate_id_only([row])  # type: ignore[list-item]
        actual.eval()
        expected.eval()
        with torch.inference_mode():
            self.assertTrue(
                torch.equal(actual.teacher_logits(batch), expected.teacher_logits(batch))
            )

    def test_reference_training_config(self) -> None:
        config = TrainingConfig.from_dict(
            {
                "schema_version": "alakazam_sota_id_only_bc_v1",
                "model": {"d_model": 320, "heads": 8, "encoder_layers": 4},
            }
        )
        self.assertEqual(config.learning_rate, 3e-4)
        self.assertEqual(config.epochs, 6)
        self.assertEqual(config.batch_size, 96)
        self.assertEqual(config.model.d_model, 320)

    def test_collate_and_model_support_empty_public_entities(self) -> None:
        config = IDOnlyConfig(d_model=32, heads=4, encoder_layers=1, dropout=0.0)
        row = IDOnlyCodec(config).encode(_observation(), [1])
        self.assertIsNotNone(row)
        row["entity_cat"] = []  # type: ignore[index]
        row["entity_num"] = []  # type: ignore[index]
        for option in row["option_cat"]:  # type: ignore[index]
            option[9] = 0
            option[10] = 0
        batch = collate_id_only([row])  # type: ignore[list-item]
        self.assertEqual(tuple(batch["entity_cat"].shape), (1, 1, 7))
        self.assertEqual(batch["entity_mask"].tolist(), [[False]])
        model = IDOnlyPointerPolicy(config).eval()
        with torch.inference_mode():
            logits = model.teacher_logits(batch)
        self.assertEqual(tuple(logits.shape), (1, 2, 3))
        self.assertTrue(torch.isfinite(logits).all())

    def test_live_encoding_and_greedy_decode_enforce_count_bounds(self) -> None:
        config = IDOnlyConfig(d_model=32, heads=4, encoder_layers=1, dropout=0.0)
        row = IDOnlyCodec(config).encode(_observation(), None)
        self.assertIsNotNone(row)
        batch = collate_id_only([row])  # type: ignore[list-item]
        self.assertEqual(batch["min_count"].tolist(), [1])
        self.assertEqual(batch["max_count"].tolist(), [1])
        model = IDOnlyPointerPolicy(config).eval()
        with torch.inference_mode():
            action = model.greedy_action(batch)
        self.assertEqual(len(action), 1)
        self.assertIn(action[0], (0, 1))

    def test_dataset_builder_uses_frozen_identity_and_replay_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            archive_root = root / "archives"
            archive_root.mkdir()
            output = root / "dataset"
            replay = {
                "info": {"EpisodeId": 42},
                "steps": [[{"visualize": [{"obs": _observation(), "selected": [1]}]}]],
            }
            with zipfile.ZipFile(archive_root / "daily.zip", "w") as bundle:
                bundle.writestr("episodes/42.json", json.dumps(replay))
            source_row = {
                "episode_id": 42,
                "player_index": 0,
                "episode_step": 0,
                "source": "/removed/episodes/42.json",
                "split": "train",
                "targets": [1],
            }
            source.write_text(json.dumps(source_row) + "\n", encoding="utf-8")
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            audit = build_id_only_dataset(
                source,
                archive_root,
                output,
                model_config=IDOnlyConfig(d_model=32, heads=4, encoder_layers=1),
                expected_source_sha256=source_hash,
            )
            self.assertEqual(audit["records_by_split"], {"train": 1, "validation": 0, "test": 0})
            with gzip.open(output / "train.jsonl.gz", "rt", encoding="utf-8") as handle:
                row = json.loads(handle.readline())
            self.assertEqual(row["episode_id"], 42)
            self.assertEqual(row["action"], [1])
            self.assertEqual(row["global_cat"], [2, 4, 1, 1])

    def test_cpu_training_smoke_writes_checkpoint_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = root / "dataset"
            dataset_root.mkdir()
            model_config = IDOnlyConfig(
                d_model=32,
                heads=4,
                encoder_layers=1,
                dropout=0.0,
            )
            row = IDOnlyCodec(model_config).encode(_observation(), [1])
            self.assertIsNotNone(row)
            for split, count in (("train", 2), ("validation", 1), ("test", 0)):
                with gzip.open(dataset_root / f"{split}.jsonl.gz", "wt") as handle:
                    for _ in range(count):
                        handle.write(json.dumps(row) + "\n")
            audit = {
                "schema_version": "alakazam_sota_id_only_dataset_v1",
                "records_by_split": {"train": 2, "validation": 1, "test": 0},
            }
            (dataset_root / "dataset_audit.json").write_text(
                json.dumps(audit), encoding="utf-8"
            )
            config = TrainingConfig(
                model=model_config,
                batch_size=2,
                epochs=1,
                learning_rate=1e-3,
                train_eval_interval=1,
                device="cpu",
                amp=False,
                max_model_mib=5.0,
                storage_path=str(root),
                min_free_gib=0.0,
            )
            output = root / "smoke"
            summary = train(dataset_root, output, config)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["best_epoch"], 1)
            self.assertTrue((output / "checkpoints" / "best_validation.pt").is_file())
            self.assertTrue((output / "metrics.jsonl").is_file())
            resumed = train(
                dataset_root,
                root / "resumed",
                config,
                resume_checkpoint=output / "checkpoints" / "latest.pt",
            )
            self.assertEqual(resumed["resume_start_epoch"], 1)
            self.assertEqual(resumed["best_epoch"], 2)


if __name__ == "__main__":
    unittest.main()
