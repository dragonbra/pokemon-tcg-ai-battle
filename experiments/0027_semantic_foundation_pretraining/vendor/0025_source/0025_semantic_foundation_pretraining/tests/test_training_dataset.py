from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import tempfile
import unittest
from pathlib import Path


DATASET = importlib.import_module(
    "train.0025_semantic_foundation_pretraining.training.dataset"
)


def _record(option_count: int, action: list[int], episode_id: int) -> dict:
    actor = {
        "legacy": {
            "global_cat": [1, 1, 1, 1],
            "global_num": [0.0] * 12,
            "entity_cat": [],
            "entity_num": [],
            "option_cat": [[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, index + 1] for index in range(option_count)],
            "min_count": 1,
            "max_count": 2,
        },
        "prototype_card_refs": [],
        "prototype_attack_refs": [],
        "prototype_skill_refs": [],
        "prototype_effect_refs": [],
        "semantic_option_cat": [[1] + [0] * 15 for _ in range(option_count)],
        "semantic_option_num": [[0.0] * 14 for _ in range(option_count)],
        "semantic_option_state": [[3] * 14 for _ in range(option_count)],
        "deck_card_ids": [1],
        "deck_multiplicity": [60],
        "ledger_cat": [],
        "ledger_num": [],
        "event_cat": [],
        "event_num": [],
        "known_opponent_hand_card_ids": [],
        "unknown_opponent_hand_count": 0,
        "turn_budget": [0.0] * 5,
    }
    return {
        "actor": actor,
        "target": {"ordered_action": action, "termination": "forced_max"},
        "audit": {
            "identity": {"episode_id": episode_id},
            "split": "train",
            "source_id": 73,
            "source_payload_sha256": "a" * 64,
        },
    }


class TrainingDatasetTests(unittest.TestCase):
    def test_shared_targets_use_padded_stop_and_exclude_audit(self) -> None:
        batch = DATASET.collate_training_records(
            [_record(2, [1], 1), _record(4, [2, 0], 2)]
        )
        self.assertEqual(tuple(batch["targets"].shape), (2, 3))
        self.assertEqual(batch["targets"].tolist(), [[1, 4, -100], [2, 0, 4]])
        self.assertEqual(tuple(batch["option_mask"].shape), (2, 4))
        self.assertNotIn("source_id", batch)
        self.assertNotIn("audit", batch)

    def test_verified_split_iteration_is_deterministic_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [_record(2, [index % 2], index) for index in range(5)]
            shard = root / "train-00000.jsonl.gz"
            with gzip.open(shard, "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            digest = hashlib.sha256(shard.read_bytes()).hexdigest()
            manifest = {
                "schema_version": "0025_semantic_decision_v1",
                "status": "complete",
                "split_counts": {"train": 5, "validation": 0},
                "shards": {
                    "train": [{"path": shard.name, "count": 5, "sha256": digest}],
                    "validation": [],
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest) + "\n")
            dataset = DATASET.SemanticDecisionDataset(root)
            first = [
                [row["audit"]["identity"]["episode_id"] for row in batch]
                for batch in dataset.iter_record_batches("train", 2, seed=9)
            ]
            second = [
                [row["audit"]["identity"]["episode_id"] for row in batch]
                for batch in dataset.iter_record_batches("train", 2, seed=9)
            ]
            self.assertEqual(first, second)
            self.assertEqual(sorted(value for batch in first for value in batch), list(range(5)))
            self.assertEqual(dataset.batch_count("train", 2), 3)

    def test_actor_label_copy_is_rejected(self) -> None:
        row = _record(2, [1], 1)
        row["actor"]["legacy"]["action"] = [1]
        with self.assertRaisesRegex(ValueError, "actor payload contains target action"):
            DATASET.collate_training_records([row])


if __name__ == "__main__":
    unittest.main()
