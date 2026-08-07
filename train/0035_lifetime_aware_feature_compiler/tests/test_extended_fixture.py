from __future__ import annotations

import gzip
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest


BASE = "train.0035_lifetime_aware_feature_compiler"
EXTENDED = importlib.import_module(f"{BASE}.tests.extended_fixture")
COMPILER = importlib.import_module(f"{BASE}.features.compiler")
INCREMENTAL = importlib.import_module(f"{BASE}.features.incremental")
COLLATE = importlib.import_module(f"{BASE}.features.collate")
FIELDS = importlib.import_module(f"{BASE}.contracts.fields")
PROTOTYPES = importlib.import_module(f"{BASE}.domain.prototypes")
STATE = importlib.import_module(f"{BASE}.knowledge.state")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


class ExtendedSemanticFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assets = PROJECT_ROOT / "assets"
        cls.prototypes = PROTOTYPES.PrototypeIndex.load(
            assets / "official_public_prototypes_v1.json",
            assets / "official_full_engine_prototypes_v2.json",
        )

    def test_fixture_has_committed_actor_deck_and_chronology_coverage(self) -> None:
        trajectories = EXTENDED.load_extended_trajectories()
        self.assertEqual(len(trajectories), 10)
        self.assertEqual(sum(len(item.decisions) for item in trajectories), 525)
        self.assertEqual(
            {actor: sum(item.actor == actor for item in trajectories) for actor in (0, 1)},
            {0: 5, 1: 5},
        )
        self.assertEqual(
            len({item.deck_manifest["sha256"] for item in trajectories}),
            7,
        )

        select_types: set[int] = set()
        log_types: set[int] = set()
        maximum_turn = 0
        for trajectory in trajectories:
            steps = [
                decision.raw_row["identity"]["episode_step"]
                for decision in trajectory.decisions
            ]
            indexes = [
                decision.event_cursor["actor_decision_index"]
                for decision in trajectory.decisions
            ]
            self.assertEqual(indexes, list(range(len(trajectory.decisions))))
            self.assertTrue(all(left < right for left, right in zip(steps, steps[1:])))
            for decision in trajectory.decisions:
                observation = decision.observation
                select_types.add(observation["select"]["type"])
                log_types.update(
                    item["type"] for item in observation["logs"] if "type" in item
                )
                maximum_turn = max(maximum_turn, observation["current"]["turn"])

        self.assertTrue({0, 1, 4, 5, 6, 7, 8, 9}.issubset(select_types))
        self.assertTrue({0, 5, 8, 10, 12, 15, 16, 22}.issubset(log_types))
        self.assertGreaterEqual(maximum_turn, 15)

    def test_full_0035_compiler_processes_every_extended_decision(self) -> None:
        compiled = 0
        for trajectory in EXTENDED.load_extended_trajectories():
            knowledge = STATE.CausalKnowledge(trajectory.actor, trajectory.deck)
            records = []
            for decision_index, decision in enumerate(trajectory.decisions):
                snapshot = knowledge.consume(decision.observation, decision.event_cursor)
                self.assertEqual(snapshot.decision_index, decision_index)
                record = COMPILER.compile_canonical_row(
                    decision.row(), snapshot, self.prototypes
                )
                self.assertEqual(record["schema_version"], FIELDS.SCHEMA_VERSION)
                records.append(record)
                compiled += 1
            batch = COLLATE.collate_canonical_records(records)
            self.assertEqual(frozenset(batch), FIELDS.EXPECTED_BATCH_KEYS)
            self.assertEqual(len(batch), 39)
            self.assertTrue(all(value.shape[0] == len(records) for value in batch.values()))
        self.assertEqual(compiled, 525)

    def test_incremental_compiler_has_exact_extended_record_and_tensor_parity(self) -> None:
        compiled = 0
        for trajectory in EXTENDED.load_extended_trajectories():
            full_knowledge = STATE.CausalKnowledge(trajectory.actor, trajectory.deck)
            incremental_knowledge = STATE.CausalKnowledge(
                trajectory.actor, trajectory.deck
            )
            incremental = INCREMENTAL.IncrementalCanonicalCompiler(self.prototypes)
            full_records = []
            incremental_records = []
            for decision in trajectory.decisions:
                full_snapshot = full_knowledge.consume(
                    decision.observation, decision.event_cursor
                )
                incremental_snapshot = incremental_knowledge.consume(
                    decision.observation, decision.event_cursor
                )
                row = decision.row()
                expected = COMPILER.compile_canonical_row(
                    row, full_snapshot, self.prototypes
                )
                actual = incremental.compile(row, incremental_snapshot)
                self.assertEqual(actual, expected)
                full_records.append(expected)
                incremental_records.append(actual)
                compiled += 1
            full_batch = COLLATE.collate_canonical_records(full_records)
            incremental_batch = COLLATE.collate_canonical_records(
                incremental_records
            )
            self.assertEqual(set(incremental_batch), set(full_batch))
            for name in full_batch:
                self.assertTrue(
                    (incremental_batch[name] == full_batch[name]).all().item(), name
                )
            self.assertEqual(incremental.stats.snapshot()["fallbacks"], 0)
        self.assertEqual(compiled, 525)

    def test_loader_rejects_content_and_provenance_drift(self) -> None:
        with gzip.open(EXTENDED.FIXTURE_PATH, "rt", encoding="ascii") as handle:
            original = json.load(handle)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json.gz"

            content_drift = json.loads(json.dumps(original))
            content_drift["trajectories"][0]["actor"] = 1
            with gzip.open(path, "wt", encoding="ascii") as handle:
                json.dump(content_drift, handle)
            with self.assertRaisesRegex(ValueError, "content commitment"):
                EXTENDED.load_extended_trajectories(path)

            provenance_drift = json.loads(json.dumps(original))
            provenance_drift["source"]["shard_sha256"] = "0" * 64
            unhashed = dict(provenance_drift)
            unhashed.pop("content_sha256")
            provenance_drift["content_sha256"] = hashlib.sha256(
                _canonical_bytes(unhashed)
            ).hexdigest()
            with gzip.open(path, "wt", encoding="ascii") as handle:
                json.dump(provenance_drift, handle)
            with self.assertRaisesRegex(ValueError, "audited source commitment"):
                EXTENDED.load_extended_trajectories(path)


if __name__ == "__main__":
    unittest.main()
