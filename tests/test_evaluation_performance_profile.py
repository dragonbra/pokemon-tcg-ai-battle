from __future__ import annotations

import unittest
import os
from types import SimpleNamespace

from evaluation.performance_profile import (
    _ProcessTreeRssMonitor,
    _workload,
    aggregate_worker_performance,
)


class EvaluationPerformanceProfileTest(unittest.TestCase):
    def test_process_tree_rss_monitor_includes_current_process(self) -> None:
        monitor = _ProcessTreeRssMonitor(os.getpid(), interval_seconds=0.001)
        monitor.start()
        peak = monitor.stop()
        self.assertGreater(peak, 0)

    def test_workload_distributes_balanced_pairs_across_catalog(self) -> None:
        catalog = SimpleNamespace(opponents=("a", "b", "c"))

        opponents, counts = _workload(catalog, 16)

        self.assertEqual(opponents, ("a", "b", "c"))
        self.assertEqual(counts, (6, 6, 4))
        self.assertTrue(all(count % 2 == 0 for count in counts))

    def test_aggregate_worker_performance_sums_only_valid_numeric_fields(self) -> None:
        records = (
            {
                "steps": 2,
                "performance": {
                    "worker_wall_seconds": 2.0,
                    "engine_start_seconds": 0.1,
                    "engine_select_seconds": 0.4,
                    "agent_seconds": 1.0,
                    "agent_calls": 2,
                    "engine_select_calls": 2,
                },
            },
            {
                "steps": 3,
                "performance": {
                    "worker_wall_seconds": 3.0,
                    "engine_start_seconds": 0.2,
                    "engine_select_seconds": 0.6,
                    "agent_seconds": 1.5,
                    "agent_calls": 3,
                    "engine_select_calls": 3,
                },
            },
        )

        result = aggregate_worker_performance(records, wall_seconds=2.5, workers=2)

        self.assertEqual(result["games"], 2)
        self.assertEqual(result["engine_select_calls"], 5)
        self.assertEqual(result["agent_calls"], 5)
        self.assertAlmostEqual(result["engine_select_seconds"], 1.0)
        self.assertAlmostEqual(result["engine_seconds_per_selection"], 0.2)
        self.assertAlmostEqual(result["worker_capacity_utilization"], 1.0)

    def test_aggregate_worker_performance_handles_empty_input(self) -> None:
        result = aggregate_worker_performance((), wall_seconds=1.0, workers=1)
        self.assertEqual(result["games"], 0)
        self.assertEqual(result["engine_seconds_per_selection"], 0.0)


if __name__ == "__main__":
    unittest.main()
