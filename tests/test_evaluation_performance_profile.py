from __future__ import annotations

import unittest
import os
from types import SimpleNamespace

from evaluation.performance_profile import (
    _ProcessTreeRssMonitor,
    _workload,
    aggregate_worker_performance,
    derive_pipeline_diagnostics,
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

    def test_pipeline_diagnostics_separates_batch_and_request_costs(self) -> None:
        inference = {
            "batches": 2,
            "mean_batch_size": 3.0,
            "batch_histogram": {"2": 1, "4": 1},
            "counters": {"batches_deadline": 2},
            "request_latency_ms": {"count": 6, "total": 60.0},
            "latency_ms": {
                "ipc_ingress": {"mean": 2.0},
                "handler_wakeup": {"mean": 3.0},
                "queue_wait": {"mean": 4.0},
            },
            "seconds": {
                "batch_cycle_seconds": 0.2,
                "inference_dispatch_seconds": 0.1,
                "dispatch_profile_bookkeeping_seconds": 0.01,
                "gpu_model_seconds": 0.04,
                "collate_cpu_seconds": 0.02,
                "h2d_gpu_seconds": 0.01,
                "ipc_egress_send_seconds": 0.03,
            },
        }
        worker = {
            "ipc_calls": 6,
            "compiler_seconds": 0.006,
            "ipc_roundtrip_seconds": 0.12,
        }

        result = derive_pipeline_diagnostics(
            inference, worker, wall_seconds=1.0, max_live_environments=4
        )

        self.assertEqual(result["batching"]["deadline_fraction"], 1.0)
        self.assertEqual(result["per_request_ms"]["compiler"], 1.0)
        self.assertEqual(result["per_request_ms"]["server_request_mean"], 10.0)
        self.assertAlmostEqual(
            result["dispatch_handoff_residual"]["milliseconds_per_batch"], 45.0
        )


if __name__ == "__main__":
    unittest.main()
