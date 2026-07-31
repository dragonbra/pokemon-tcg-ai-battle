from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ..storage import GIB, preflight_storage, prune_latest, runtime_storage


class StorageTest(unittest.TestCase):
    def test_launch_and_runtime_thresholds_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            usage = mock.Mock(free=90 * GIB)
            with mock.patch("shutil.disk_usage", return_value=usage):
                with self.assertRaisesRegex(RuntimeError, "requires 100 GiB"):
                    preflight_storage(root)
                self.assertFalse(runtime_storage(root).stop_requested)
            usage.free = 79 * GIB
            with mock.patch("shutil.disk_usage", return_value=usage):
                self.assertTrue(runtime_storage(root).stop_requested)

    def test_pruning_preserves_latest_and_protected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / f"update-{index:06d}.pt" for index in range(5)]
            for path in paths: path.write_bytes(b"x")
            removed = prune_latest(root, keep=2, protected={paths[0]})
            self.assertEqual(set(removed), set(paths[1:3]))
            self.assertTrue(paths[0].exists())
            self.assertTrue(all(path.exists() for path in paths[3:]))


if __name__ == "__main__":
    unittest.main()
