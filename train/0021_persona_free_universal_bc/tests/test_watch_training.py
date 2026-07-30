from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
import tempfile
import unittest


latest_checkpoint = import_module(
    "train.0021_persona_free_universal_bc.watch_training"
).latest_checkpoint


class WatchTrainingTests(unittest.TestCase):
    def test_latest_checkpoint_must_resolve_inside_version_checkpoint_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "criteria").mkdir()
            checkpoint = root / "epoch-0001.pt"
            checkpoint.write_bytes(b"checkpoint")
            (root / "criteria/latest.json").write_text(
                json.dumps({"path": checkpoint.name}), encoding="utf-8"
            )
            self.assertEqual(latest_checkpoint(root), checkpoint)
            (root / "criteria/latest.json").write_text(
                json.dumps({"path": "../outside.pt"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "escapes or is absent"):
                latest_checkpoint(root)


if __name__ == "__main__":
    unittest.main()
