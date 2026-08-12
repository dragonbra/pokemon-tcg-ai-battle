from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from engine_cuda_2_0.tools.run_official_semantic0031_v2_parity_scaffold import (
    build_and_run_trace,
)


class ImmutableTraceReuseTest(unittest.TestCase):
    def test_switch_recovery_retains_the_departing_active_identity(self) -> None:
        source = Path(
            "engine_cuda_2_0/include/ptcg_cuda/official_core_pod.cuh"
        ).read_text(encoding="utf-8")
        begin = source.index("inline bool official_pod_switch_active(")
        end = source.index("template <typename T>", begin)
        switch = source[begin:end]
        history = switch.index("OfficialSemanticLogType::kSwitch")
        recovery = switch.index(
            "official_pod_clear_special_conditions_for_ref(state, player, active)"
        )
        exchange = switch.index("ps->active.values[0] = bench")
        self.assertLess(history, recovery)
        self.assertLess(recovery, exchange)

    def test_reuse_trace_neither_builds_nor_overwrites_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / "trace.jsonl"
            payload = {
                "actor_observation": {"select": {"type": 0, "list": []}},
                "ordered_action": [],
            }
            trace.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            before = trace.read_bytes()
            args = argparse.Namespace(
                source=root,
                trace=trace,
                reuse_trace=True,
                skip_trace_build=False,
            )

            report = build_and_run_trace(args, root / "unused.bin", root, root)

            self.assertEqual(trace.read_bytes(), before)
            self.assertEqual(report["semantic_trace_records"], 1)
            self.assertEqual(report["trace_sha256"], hashlib.sha256(before).hexdigest())
            self.assertTrue(report["reused"])


if __name__ == "__main__":
    unittest.main()
