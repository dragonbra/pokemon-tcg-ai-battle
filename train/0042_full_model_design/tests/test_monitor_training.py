from __future__ import annotations

import importlib
from pathlib import Path
import tempfile
import unittest


monitor = importlib.import_module("train.0042_full_model_design.monitor_training")


class MonitorTrainingTest(unittest.TestCase):
    def test_historical_failed_status_does_not_kill_resume(self) -> None:
        self.assertFalse(monitor._is_current_failed_status("failed", 99.0, 100.0))

    def test_new_failed_status_is_fatal(self) -> None:
        self.assertTrue(monitor._is_current_failed_status("failed", 100.0, 100.0))
        self.assertTrue(monitor._is_current_failed_status("failed_cuda", 101.0, 100.0))

    def test_running_status_is_not_fatal(self) -> None:
        self.assertFalse(monitor._is_current_failed_status("running", 101.0, 100.0))

    def test_tail_can_ignore_historical_attempt_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training.log"
            path.write_text("Traceback (most recent call last)\n")
            offset = path.stat().st_size
            with path.open("a") as handle:
                handle.write("fresh attempt started\n")
            self.assertEqual(
                monitor._tail(path, start_offset=offset),
                "fresh attempt started\n",
            )


if __name__ == "__main__":
    unittest.main()
