from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


monitor = importlib.import_module(
    "train.0024_marnies_grimmsnarl_ex_froslass_001_league_training.monitor_training"
)


def _args(version: str, pid: int, log: Path, *, min_hours: float) -> argparse.Namespace:
    return argparse.Namespace(
        version=version,
        pid=pid,
        log=log,
        min_hours=min_hours,
        interval_seconds=0.01,
        metrics_stale_minutes=45.0,
        first_metrics_grace_minutes=45.0,
        checkpoint_stale_minutes=90.0,
    )


class TrainingMonitorTest(unittest.TestCase):
    def test_process_exit_emits_alert_and_writes_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            version = "V7_test"
            artifact = root / version / "artifact"
            artifact.mkdir(parents=True)
            (artifact / "status.json").write_text(
                json.dumps({"state": "running"}), encoding="utf-8"
            )
            log = root / "train.log"
            log.write_text("", encoding="utf-8")
            output = io.StringIO()
            with (
                mock.patch.object(monitor, "RUN_ROOT", root),
                mock.patch.object(monitor, "MONITOR_ROOT", root / "monitor"),
                mock.patch.object(monitor, "_nvidia_smi", return_value={"available": False}),
                contextlib.redirect_stdout(output),
            ):
                result = monitor.monitor(_args(version, 999_999_999, log, min_hours=24.0))
            self.assertEqual(result, 2)
            events = output.getvalue().splitlines()
            self.assertTrue(any(line.startswith("TRAINING_MONITOR_ALERT ") for line in events))
            alert = json.loads(next(line.split(" ", 1)[1] for line in events if line.startswith("TRAINING_MONITOR_ALERT ")))
            self.assertEqual(alert["alert_reason"], "process_exited")
            self.assertTrue((root / "monitor" / version / "alert.json").is_file())

    def test_minimum_watch_completion_emits_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            version = "V7_test"
            artifact = root / version / "artifact"
            artifact.mkdir(parents=True)
            (artifact / "status.json").write_text(
                json.dumps({"state": "running"}), encoding="utf-8"
            )
            log = root / "train.log"
            log.write_text("", encoding="utf-8")
            output = io.StringIO()
            with (
                mock.patch.object(monitor, "RUN_ROOT", root),
                mock.patch.object(monitor, "MONITOR_ROOT", root / "monitor"),
                mock.patch.object(monitor, "_nvidia_smi", return_value={"available": False}),
                contextlib.redirect_stdout(output),
            ):
                result = monitor.monitor(_args(version, __import__("os").getpid(), log, min_hours=0.0))
            self.assertEqual(result, 0)
            self.assertIn("TRAINING_MONITOR_COMPLETE ", output.getvalue())


if __name__ == "__main__":
    unittest.main()
