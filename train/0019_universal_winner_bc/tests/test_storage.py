from __future__ import annotations

import tempfile
from importlib import import_module
import unittest
from pathlib import Path


STORAGE = import_module("train.0019_universal_winner_bc.storage")
directory_bytes = STORAGE.directory_bytes
guard_storage = STORAGE.guard_storage


class StorageGuardTests(unittest.TestCase):
    def test_counts_files_and_enforces_dataset_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "payload").write_bytes(b"1234")
            self.assertEqual(directory_bytes(root), 4)
            status = guard_storage(
                root,
                dataset_limit_bytes=5,
                linux_free_floor_bytes=0,
                c_free_floor_bytes=0,
                c_mount=root,
            )
            self.assertEqual(status["dataset_bytes"], 4)
            with self.assertRaisesRegex(RuntimeError, "dataset hard limit"):
                guard_storage(
                    root,
                    dataset_limit_bytes=4,
                    linux_free_floor_bytes=0,
                    c_free_floor_bytes=0,
                    c_mount=root,
                )


if __name__ == "__main__":
    unittest.main()
