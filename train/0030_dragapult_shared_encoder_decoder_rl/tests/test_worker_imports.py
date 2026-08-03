from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
WORKER_MODULE = "train.0030_dragapult_shared_encoder_decoder_rl.rollout.worker"


class WorkerImportTest(unittest.TestCase):
    def test_worker_import_does_not_load_torch(self):
        code = (
            "import importlib, sys; "
            f"importlib.import_module({WORKER_MODULE!r}); "
            "raise SystemExit(1 if 'torch' in sys.modules else 0)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
