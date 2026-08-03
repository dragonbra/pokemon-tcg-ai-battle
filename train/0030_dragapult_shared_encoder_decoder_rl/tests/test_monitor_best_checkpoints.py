from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from ..monitor_best_checkpoints import (
    AuditConfig,
    AuditState,
    EvaluationPoint,
    audit_checkpoint,
    evaluation_record,
    next_audit_version,
    select_new_bests,
)


def metric(update: int, win_rate: float) -> dict[str, float]:
    return {
        "trainer/update": float(update),
        "eval/checkpoint_update": float(update),
        "eval/foundation_0019/win_rate": win_rate,
    }


class BestCheckpointMonitorTest(unittest.TestCase):
    def test_evaluation_record_ignores_rollout_only_metrics(self):
        self.assertIsNone(evaluation_record({"trainer/update": 7.0}))
        self.assertEqual(
            evaluation_record(metric(10, 57 / 102)),
            EvaluationPoint(update=10, win_rate=57 / 102),
        )

    def test_selects_only_strict_new_best(self):
        state = AuditState.initial(5, 62 / 102)
        points = select_new_bests(
            [metric(10, 57 / 102), metric(15, 62 / 102), metric(20, 63 / 102)],
            state,
        )
        self.assertEqual(points, [EvaluationPoint(update=20, win_rate=63 / 102)])

    def test_attempted_update_is_not_selected_after_restart(self):
        point = EvaluationPoint(update=20, win_rate=63 / 102)
        state = AuditState.initial(5, 62 / 102).mark_attempted(point)
        self.assertEqual(select_new_bests([metric(20, 63 / 102)], state), [])

    def test_state_round_trip_preserves_attempted_update(self):
        point = EvaluationPoint(update=20, win_rate=63 / 102)
        state = AuditState.initial(5, 62 / 102).mark_attempted(point)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state.write(path)
            restored = AuditState.read(path)
        self.assertEqual(restored, state)

    def test_missing_checkpoint_returns_failure_without_export_or_evaluation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = AuditConfig(
                repository_root=root,
                training_version="V3_lightweight_engine_workers",
                state_path=root / "state.json",
            )
            runner = mock.Mock()
            result = audit_checkpoint(
                EvaluationPoint(update=20, win_rate=63 / 102),
                config,
                command_runner=runner,
            )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "checkpoint_missing")
        runner.assert_not_called()

    def test_next_version_is_monotonic(self):
        self.assertEqual(
            next_audit_version(["V1_a", "V3_c", "not_a_version"], 20),
            "V4_update20_frozen0019_audit",
        )

    def test_failed_result_can_be_recorded_idempotently(self):
        point = EvaluationPoint(update=20, win_rate=63 / 102)
        state = AuditState.initial(5, 62 / 102).mark_attempted(point)
        failed = state.mark_finished(point.update, "failed", reason="checkpoint_missing")
        payload = json.loads(json.dumps(failed.to_dict()))
        restored = AuditState.from_dict(payload)
        self.assertEqual(restored.attempts["20"]["status"], "failed")
        self.assertEqual(select_new_bests([metric(20, 63 / 102)], restored), [])


if __name__ == "__main__":
    unittest.main()
