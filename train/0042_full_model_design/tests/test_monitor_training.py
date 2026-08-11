from __future__ import annotations

import importlib
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


if __name__ == "__main__":
    unittest.main()
