from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rl_environment.streaming import aligned_records, chunks, iter_jsonl, shuffle_buffer


class RLStreamingTests(unittest.TestCase):
    def test_jsonl_chunks_and_shuffle_are_bounded_and_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text(
                "".join(json.dumps({"id": index}) + "\n" for index in range(10)),
                encoding="utf-8",
            )
            first = list(shuffle_buffer(iter_jsonl(path), seed=7, buffer_size=3))
            second = list(shuffle_buffer(iter_jsonl(path), seed=7, buffer_size=3))
        self.assertEqual(first, second)
        self.assertEqual(sorted(row["id"] for row in first), list(range(10)))
        self.assertEqual([len(batch) for batch in chunks(first, 4)], [4, 4, 2])

    def test_aligned_records_fail_closed_on_shifted_sidecar(self) -> None:
        primary = ({"id": value} for value in (1, 2))
        sidecar = ({"id": value} for value in (1, 3))
        with self.assertRaisesRegex(ValueError, "stream identity mismatch"):
            list(aligned_records(primary, sidecar, identity=lambda row: row["id"]))


if __name__ == "__main__":
    unittest.main()
