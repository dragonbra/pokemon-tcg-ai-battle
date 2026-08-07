from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import unittest

import torch


BASE = "train.0035_lifetime_aware_feature_compiler"
BENCHMARK = importlib.import_module(f"{BASE}.benchmark_incremental_features")
FIELDS = importlib.import_module(f"{BASE}.contracts.fields")
PROTOTYPES = importlib.import_module(f"{BASE}.domain.prototypes")
COLLATE = importlib.import_module(f"{BASE}.features.collate")
COMPILER = importlib.import_module(f"{BASE}.features.compiler")
STATE = importlib.import_module(f"{BASE}.knowledge.state")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "tests/fixtures/incremental_feature_trajectory.json.gz"
REFERENCE_PATH = PROJECT_ROOT / "tests/fixtures/canonical_reference_hashes.json"
REFERENCE_SCHEMA = "0035_canonical_semantic_reference_v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_record_sha256(record: object) -> str:
    encoded = json.dumps(
        record,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_batch_sha256(batch: dict[str, torch.Tensor]) -> str:
    """Commit names, dtypes, shapes, and exact values for all 39 tensors."""
    digest = hashlib.sha256()
    for name, tensor in sorted(batch.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class SemanticLineageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = json.loads(REFERENCE_PATH.read_text(encoding="ascii"))
        assets = PROJECT_ROOT / "assets"
        cls.prototypes = PROTOTYPES.PrototypeIndex.load(
            assets / "official_public_prototypes_v1.json",
            assets / "official_full_engine_prototypes_v2.json",
        )

    def test_frozen_fixture_recompiles_to_exact_0031_semantics(self) -> None:
        reference = self.reference
        self.assertEqual(reference["schema_version"], REFERENCE_SCHEMA)
        self.assertEqual(reference["actor_schema"], FIELDS.SCHEMA_VERSION)
        self.assertEqual(reference["fixture_file"], FIXTURE_PATH.name)
        self.assertEqual(reference["fixture_sha256"], _file_sha256(FIXTURE_PATH))

        expected_keys = sorted(FIELDS.EXPECTED_BATCH_KEYS)
        self.assertEqual(reference["tensor_keys"], expected_keys)
        self.assertEqual(len(expected_keys), 39)

        actual_commitments: list[dict[str, object]] = []
        for trajectory in BENCHMARK.load_parity_trajectories(FIXTURE_PATH):
            knowledge = STATE.CausalKnowledge(trajectory.actor, trajectory.deck)
            for decision_index, decision in enumerate(trajectory.decisions):
                snapshot = knowledge.consume(decision.observation, decision.event_cursor)
                record = COMPILER.compile_canonical_row(
                    decision.row(), snapshot, self.prototypes
                )
                batch = COLLATE.collate_canonical_records([record])

                self.assertEqual(sorted(batch), expected_keys)
                self.assertTrue(all(isinstance(value, torch.Tensor) for value in batch.values()))
                actual_commitments.append(
                    {
                        "episode_id": decision.identity["episode_id"],
                        "player_index": decision.identity["player_index"],
                        "episode_step": decision.identity["episode_step"],
                        "decision_index": decision_index,
                        "record_sha256": _canonical_record_sha256(record),
                        "tensor_sha256": _tensor_batch_sha256(batch),
                    }
                )

        self.assertEqual(len(actual_commitments), 35)
        self.assertEqual(actual_commitments, reference["decisions"])


if __name__ == "__main__":
    unittest.main()
