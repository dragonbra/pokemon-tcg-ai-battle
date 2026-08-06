from __future__ import annotations

import gzip
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest


BASE = "train.0033_effect_summary_semantic_foundation_pretraining"
SCHEMA = importlib.import_module(f"{BASE}.contracts.fields")
DATASET = importlib.import_module(f"{BASE}.training.dataset")
COMPILER = importlib.import_module(f"{BASE}.features.compiler")
PROTOTYPES = importlib.import_module(f"{BASE}.domain.prototypes")
_row = importlib.import_module(f"{BASE}.tests.test_features")._row


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CanonicalDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prototypes = PROTOTYPES.PrototypeIndex.load(
            Path(
                "train/0033_effect_summary_semantic_foundation_pretraining/assets/"
                "official_public_prototypes_v1.json"
            )
        )

    def _dataset(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        shards: dict[str, list[dict]] = {"train": [], "validation": []}
        for split, energy in (("train", 4), ("validation", 1)):
            source, snapshot = _row(energy)
            source["split"] = split
            record = COMPILER.compile_canonical_row(source, snapshot, self.prototypes)
            stored = {
                "schema_version": SCHEMA.SCHEMA_VERSION,
                "actor": record["actor"],
                "target": record["target"],
                "audit": {"split": split, "source_id": 73},
            }
            path = root / f"{split}-00000.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(stored, sort_keys=True) + "\n")
            shards[split].append(
                {
                    "path": path.name,
                    "count": 1,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        manifest = {
            "schema_version": SCHEMA.SCHEMA_VERSION,
            "status": "complete",
            "split_counts": {"train": 1, "validation": 1},
            "shards": shards,
            "feature_input_audit": {
                "status": "passed",
                "audited_decisions": 2,
            },
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        return root

    def test_verified_iteration_is_complete_and_actor_is_target_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = DATASET.CanonicalDecisionDataset(
                self._dataset(Path(directory))
            )
            first = list(dataset.iter_record_batches("train", 1, seed=17))
            second = list(dataset.iter_record_batches("train", 1, seed=17))
            self.assertEqual(first, second)
            self.assertEqual(dataset.split_counts, {"train": 1, "validation": 1})
            actor = first[0][0]["actor"]
            self.assertEqual(set(actor), SCHEMA.ACTOR_KEYS)
            self.assertNotIn("legacy", actor)
            self.assertNotIn("ordered_action", actor)
            batch = next(dataset.iter_batches("validation", 1, seed=17))
            self.assertEqual(set(batch), SCHEMA.EXPECTED_BATCH_KEYS)

    def test_modified_shard_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._dataset(Path(directory))
            with (root / "train-00000.jsonl.gz").open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "commitment mismatch"):
                DATASET.CanonicalDecisionDataset(root)

    def test_combined_dataset_covers_every_partition_once_per_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = self._dataset(root / "left")
            right = self._dataset(root / "right")
            combined = DATASET.CombinedCanonicalDecisionDataset([left, right])

            first = list(combined.iter_record_batches("train", 1, seed=23))
            second = list(combined.iter_record_batches("train", 1, seed=23))

            self.assertEqual(first, second)
            self.assertEqual(combined.split_counts, {"train": 2, "validation": 2})
            self.assertEqual(combined.batch_count("train", 1), 2)
            self.assertEqual(combined.manifest["partition_count"], 2)
            self.assertEqual(len(first), 2)


if __name__ == "__main__":
    unittest.main()
