from __future__ import annotations

import importlib
import json
import unittest
from pathlib import Path

split = importlib.import_module("train.0013_semantic_goal_policy.data.split")
protocol = importlib.import_module("train.0013_semantic_goal_policy.protocol")
SplitGroup = split.SplitGroup
assign_groups = split.assign_groups
PROTOCOL = protocol.PreRunProtocol.from_json(Path(__file__).parents[1] / "configs" / "pre_run_protocol.json")


def groups(*sizes: int, same_date: bool = False) -> list[SplitGroup]:
    result: list[SplitGroup] = []
    for stratum, size in enumerate(sizes):
        date = "2026-07-24" if same_date else f"2026-07-{24 + stratum:02d}"
        result.extend(SplitGroup(date, f"{stratum:x}" * 64, stratum * 1000 + item, 0) for item in range(size))
    return result


class DeterministicSplitTest(unittest.TestCase):
    def test_known_frozen_digest_vector(self) -> None:
        item = SplitGroup("2026-07-24", "a" * 64, 42, 1)
        self.assertEqual(split._digest(item), "8c7596e5f29d432593c7d378358062c45464c5fe3e9ef4be5caefeacd69bc3b3")

    def test_private_assignments_are_canonical_and_committed_by_compact_audit(self) -> None:
        first = assign_groups(groups(10, 9), PROTOCOL)
        second = assign_groups(list(reversed(groups(10, 9))), PROTOCOL.data)
        self.assertEqual(first.private_canonical_bytes(), second.private_canonical_bytes())
        self.assertEqual(first.private_sha256(), second.private_sha256())
        self.assertEqual(first.private_sha256(), first.audit.private_assignments_sha256)
        self.assertEqual(first.audit.private_assignments_count, 19)
        self.assertEqual(first.assignments, second.assignments)
        self.assertTrue(first.private_canonical_bytes().endswith(b"\n"))

    def test_compact_audit_has_no_deck_or_group_hashes(self) -> None:
        manifest = assign_groups(groups(10, 9), PROTOCOL)
        text = manifest.canonical_bytes().decode("utf-8")
        self.assertNotIn("000000000000", text)
        self.assertNotIn("\"group_hash\"", text)
        self.assertNotIn("2026-07-", text)
        self.assertNotIn("\"strata\"", text)
        self.assertNotIn("000000000000", text)

    def test_joint_quota_cases_and_large_stratum_both_splits(self) -> None:
        for sizes, expected in (((14, 14), 3), ((15, 15), 3), ((10, 9), 2), ((9, 9), 2)):
            with self.subTest(sizes=sizes):
                manifest = assign_groups(groups(*sizes), PROTOCOL)
                self.assertEqual(manifest.audit.validation_target, expected)
                self.assertEqual(manifest.audit.validation_actual, expected)
                self.assertEqual(manifest.audit.validation_delta, 0)
                for values in manifest.private_strata.values():
                    if values["groups"] >= 10:
                        self.assertGreater(values["train_groups"], 0)
                        self.assertGreater(values["validation_groups"], 0)

    def test_ten_large_strata_same_date_and_membership_commitment(self) -> None:
        manifest = assign_groups(groups(*(11 for _ in range(10)), same_date=True), PROTOCOL)
        self.assertEqual(manifest.audit.validation_actual, 11)
        validation_counts = [values["validation_groups"] for values in manifest.private_strata.values()]
        self.assertEqual(sum(validation_counts), 11)
        self.assertEqual(sorted(validation_counts), [1] * 9 + [2])
        changed = assign_groups(groups(*(11 for _ in range(10)), same_date=True)[:-1], PROTOCOL)
        self.assertNotEqual(manifest.audit.private_assignments_sha256, changed.audit.private_assignments_sha256)

    def test_half_up_ties_are_explicit(self) -> None:
        self.assertEqual(split._half_up_tenth(5), 1)
        self.assertEqual(assign_groups(groups(5), PROTOCOL).audit.validation_target, 1)
        self.assertEqual(assign_groups(groups(25), PROTOCOL).audit.validation_target, 3)

    def test_empty_singleton_and_two_group_contracts(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            assign_groups([], PROTOCOL)
        self.assertEqual(assign_groups(groups(1), PROTOCOL).audit.counts, {"train": 1, "validation": 0, "test": 0})
        self.assertEqual(assign_groups(groups(2), PROTOCOL).audit.counts, {"train": 1, "validation": 1, "test": 0})

    def test_frozen_protocol_and_trajectory_contracts(self) -> None:
        changed = json.loads(PROTOCOL.canonical_bytes())
        changed["split"]["seed"] = 7
        with self.assertRaisesRegex(ValueError, "split.seed"):
            assign_groups(groups(2), changed)
        duplicate = [SplitGroup("2026-07-24", "a" * 64, 1, 0), SplitGroup("2026-07-24", "b" * 64, 1, 0)]
        with self.assertRaisesRegex(ValueError, "duplicate episode-player identity"):
            assign_groups(duplicate, PROTOCOL)


if __name__ == "__main__":
    unittest.main()
