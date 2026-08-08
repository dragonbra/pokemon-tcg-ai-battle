from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ..export_value import EXPORT_SCHEMA_VERSION, export_value_checkpoint
from ..model.source import SOURCE_CHECKPOINT_SHA256
from ..model.value_network import LatentQueryValueHead
from ..training.checkpoints import save_value_checkpoint


class ExportTests(unittest.TestCase):
    def test_export_is_hash_committed_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "input.pt"
            save_value_checkpoint(
                checkpoint,
                LatentQueryValueHead(16, heads=4, layers=1),
                {
                    "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
                    "value_config": {
                        "architecture": "latent_queries", "queries": 8, "layers": 1, "dropout": 0.0
                    },
                },
            )
            output = root / "export"
            manifest = export_value_checkpoint(checkpoint, output)
            self.assertEqual(manifest["schema_version"], EXPORT_SCHEMA_VERSION)
            self.assertTrue((output / "value_head.pt").is_file())
            self.assertEqual(
                json.loads((output / "manifest.json").read_text())["value_checkpoint_sha256"],
                manifest["value_checkpoint_sha256"],
            )
            with self.assertRaises(FileExistsError):
                export_value_checkpoint(checkpoint, output)


if __name__ == "__main__":
    unittest.main()
