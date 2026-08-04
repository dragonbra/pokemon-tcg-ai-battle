from __future__ import annotations

import copy
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock


BASE = "train.0031_rule_faithful_semantic_foundation_pretraining"
SCHEMA = importlib.import_module(f"{BASE}.contracts.fields")
COMPILER = importlib.import_module(f"{BASE}.features.compiler")
PROTOTYPES = importlib.import_module(f"{BASE}.domain.prototypes")
DATASET = importlib.import_module(f"{BASE}.training.dataset")
COLLATE = importlib.import_module(f"{BASE}.features.collate")
_row = importlib.import_module(f"{BASE}.tests.test_features")._row


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repeat(rows: list, count: int) -> list:
    if not rows:
        return []
    return [copy.deepcopy(rows[index % len(rows)]) for index in range(count)]


class TrainingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prototypes = PROTOTYPES.PrototypeIndex.load(
            Path(
                "train/0031_rule_faithful_semantic_foundation_pretraining/assets/"
                "official_public_prototypes_v1.json"
            )
        )

    def _record(self, identity: int, scale: int) -> dict:
        source, snapshot = _row(4 if identity % 2 else 1)
        source["split"] = "train"
        compiled = COMPILER.compile_canonical_row(source, snapshot, self.prototypes)
        actor = copy.deepcopy(compiled["actor"])
        lengths = {
            "card": 4 + scale * 3,
            "resource": 3 + scale,
            "event": 2 + scale * 4,
            "option": 2 + scale * 2,
            "option_skill": 1 + scale,
            "option_effect": 1 + scale * 5,
        }
        for prefix in ("card", "resource", "event", "option"):
            for suffix in ("cat", "num"):
                name = f"{prefix}_{suffix}"
                actor[name] = _repeat(actor[name], lengths[prefix])
            state_name = f"{prefix}_state"
            actor[state_name] = _repeat(actor[state_name], lengths[prefix])
        actor["card_parent"] = [0] * lengths["card"]
        actor["option_source"] = [0] * lengths["option"]
        actor["option_target"] = [0] * lengths["option"]
        actor["option_context"] = [0] * lengths["option"]
        actor["option_effect_card"] = [0] * lengths["option"]
        for prefix in ("option_skill", "option_effect"):
            count = lengths[prefix]
            for suffix in ("id", "role"):
                source_values = actor[f"{prefix}_{suffix}"]
                actor[f"{prefix}_{suffix}"] = (
                    _repeat(source_values, count) if source_values else [0] * count
                )
            actor[f"{prefix}_parent"] = [1] * count
        actor["min_count"] = 1
        actor["max_count"] = 1
        return {
            "schema_version": SCHEMA.SCHEMA_VERSION,
            "actor": actor,
            "target": {"ordered_action": [0]},
            "audit": {"identity": {"episode_id": identity}, "split": "train"},
        }

    def _dataset(self, root: Path) -> DATASET.CanonicalDecisionDataset:
        shards = {"train": [], "validation": []}
        records = [self._record(index, 1 + index % 8) for index in range(64)]
        for split in shards:
            split_records = copy.deepcopy(records if split == "train" else records[:8])
            for record in split_records:
                record["audit"]["split"] = split
            path = root / f"{split}-00000.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for record in split_records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            shards[split].append(
                {
                    "path": path.name,
                    "count": len(split_records),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        manifest = {
            "schema_version": SCHEMA.SCHEMA_VERSION,
            "status": "complete",
            "split_counts": {"train": 64, "validation": 8},
            "shards": shards,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        return DATASET.CanonicalDecisionDataset(root)

    @staticmethod
    def _identities(batches: list[list[dict]]) -> list[int]:
        return [
            int(record["audit"]["identity"]["episode_id"])
            for batch in batches
            for record in batch
        ]

    @staticmethod
    def _padded_cells(batches: list[list[dict]], field: str) -> int:
        return sum(
            len(batch)
            * max(1, max(len(record["actor"][field]) for record in batch))
            for batch in batches
        )

    def test_length_bucketing_is_deterministic_and_exact_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            first = list(
                dataset.iter_record_batches(
                    "train", 8, seed=17, length_bucketed=True
                )
            )
            second = list(
                dataset.iter_record_batches(
                    "train", 8, seed=17, length_bucketed=True
                )
            )
            different = list(
                dataset.iter_record_batches(
                    "train", 8, seed=19, length_bucketed=True
                )
            )
            self.assertEqual(self._identities(first), self._identities(second))
            self.assertNotEqual(self._identities(first), self._identities(different))
            self.assertEqual(sorted(self._identities(first)), list(range(64)))

    def test_length_bucketing_reduces_option_and_effect_padding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            plain = list(
                dataset.iter_record_batches(
                    "train", 8, seed=17, length_bucketed=False
                )
            )
            bucketed = list(
                dataset.iter_record_batches(
                    "train", 8, seed=17, length_bucketed=True
                )
            )
            self.assertLess(
                self._padded_cells(bucketed, "option_cat"),
                self._padded_cells(plain, "option_cat"),
            )
            self.assertLess(
                self._padded_cells(bucketed, "option_effect_id"),
                self._padded_cells(plain, "option_effect_id"),
            )

    def test_fixed_bucket_padding_produces_bounded_shapes_without_truncation(self) -> None:
        records = [self._record(1, 1), self._record(2, 3)]
        batch = COLLATE.collate_canonical_records(
            records, bucket_padding=COLLATE.BucketPadding()
        )
        bounds = COLLATE.BucketPadding()
        for family, key in (
            ("card", "card_cat"),
            ("event", "event_cat"),
            ("option", "option_cat"),
            ("effect", "option_effect_id"),
            ("skill", "option_skill_id"),
        ):
            expected = bounds.upper_bound(
                family,
                max(1, max(len(record["actor"][key]) for record in records)),
            )
            self.assertEqual(batch[key].shape[1], expected)
        self.assertEqual(batch["targets"].shape[1], 2)
        self.assertEqual(
            batch["card_mask"].sum(dim=1).tolist(),
            [len(record["actor"]["card_cat"]) for record in records],
        )

    def test_fixed_bucket_padding_fails_closed_above_maximum(self) -> None:
        record = self._record(1, 1)
        actor = record["actor"]
        overflow = COLLATE.BucketPadding().effect[-1] + 1
        actor["option_effect_id"] = [1] * overflow
        actor["option_effect_role"] = [1] * overflow
        actor["option_effect_parent"] = [1] * overflow
        with self.assertRaisesRegex(ValueError, f"effect length {overflow} exceeds"):
            COLLATE.collate_canonical_records(
                [record], bucket_padding=COLLATE.BucketPadding()
            )

    def test_fixed_bucket_defaults_cover_materialized_dataset_maxima(self) -> None:
        bounds = COLLATE.BucketPadding()
        audited_maxima = {
            "card": 134,
            "event": 64,
            "option": 78,
            "effect": 317,
            "skill": 82,
        }
        for family, maximum in audited_maxima.items():
            self.assertGreaterEqual(bounds.upper_bound(family, maximum), maximum)

    def test_prefetch_preserves_order_and_reports_counts(self) -> None:
        prefetch = importlib.import_module(f"{BASE}.training.prefetch")
        with prefetch.PrefetchIterator(range(20), depth=2) as iterator:
            self.assertEqual(list(iterator), list(range(20)))
            self.assertEqual(iterator.produced, 20)
            self.assertEqual(iterator.consumed, 20)
            self.assertGreaterEqual(iterator.wait_seconds, 0.0)

    def test_prefetch_reraises_producer_exception(self) -> None:
        prefetch = importlib.import_module(f"{BASE}.training.prefetch")

        def broken():
            yield 1
            raise RuntimeError("fixture producer failed")

        with self.assertRaisesRegex(RuntimeError, "fixture producer failed"):
            with prefetch.PrefetchIterator(broken(), depth=1) as iterator:
                list(iterator)

    def test_prefetch_early_close_joins_thread(self) -> None:
        prefetch = importlib.import_module(f"{BASE}.training.prefetch")
        before = set(threading.enumerate())
        with prefetch.PrefetchIterator(range(10000), depth=1) as iterator:
            next(iterator)
        leaked = [
            thread
            for thread in threading.enumerate()
            if thread not in before and thread.name.startswith("0031-prefetch")
        ]
        self.assertEqual(leaked, [])

    def test_benchmark_result_is_finite_and_json_serializable(self) -> None:
        benchmark_module = importlib.import_module(f"{BASE}.benchmark_training")
        fake_metrics = {
            "seconds": 2.0,
            "decisions": 8,
            "decisions_per_second": 4.0,
            "data_wait_seconds": 0.1,
            "data_wait_fraction": 0.05,
            "cuda_peak_allocated_bytes": 1024,
            "cuda_peak_reserved_bytes": 2048,
        }
        fake_model = mock.Mock()
        fake_model.parameters.return_value = []
        with (
            mock.patch.object(benchmark_module.torch.cuda, "is_available", return_value=True),
            mock.patch.object(benchmark_module.torch.cuda, "manual_seed_all"),
            mock.patch.object(benchmark_module, "CanonicalDecisionDataset") as dataset_type,
            mock.patch.object(benchmark_module, "PrototypeIndex") as prototype_type,
            mock.patch.object(benchmark_module, "SemanticPolicy", return_value=fake_model),
            mock.patch.object(benchmark_module.torch.optim, "AdamW", return_value=mock.Mock()),
            mock.patch.object(benchmark_module, "_train_epoch", return_value=(fake_metrics, 2)),
        ):
            dataset_type.return_value.manifest = {}
            dataset_type.return_value.manifest_sha256 = "a" * 64
            dataset_type.return_value.split_counts = {"train": 8, "validation": 0}
            prototype_type.load.return_value = mock.Mock()
            fake_model.cuda.return_value = fake_model
            result = benchmark_module.benchmark(
                dataset_root=Path("fixture"),
                batches=2,
                batch_size=4,
                seed=17,
                prefetch_depth=2,
                length_bucketed=True,
                shared_prototypes=True,
            )
        self.assertTrue(result["acceptance"]["cpu_pipeline_gate_passed"])
        self.assertEqual(result["consumer_active_fraction"], 0.95)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
