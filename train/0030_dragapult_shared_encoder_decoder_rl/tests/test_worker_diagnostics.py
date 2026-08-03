from __future__ import annotations

import unittest

from ..worker_diagnostics import classify_maps, parse_kib_field


class WorkerDiagnosticsTest(unittest.TestCase):
    def test_parse_kib_field_converts_to_bytes(self):
        text = "Rss: 900 kB\nPss: 321 kB\nSwap: 12 kB\n"
        self.assertEqual(parse_kib_field(text, "Pss"), 321 * 1024)
        self.assertEqual(parse_kib_field(text, "Swap"), 12 * 1024)
        self.assertEqual(parse_kib_field(text, "Missing"), 0)

    def test_classify_maps_detects_forbidden_worker_runtimes(self):
        text = "\n".join((
            "/opt/python/site-packages/torch/lib/libtorch_cpu.so",
            "/usr/lib/wsl/lib/libcuda.so.1",
            "/repo/evaluation/arena/opponents/example/cg/libcg.so",
        ))
        result = classify_maps(text)
        self.assertTrue(result.torch)
        self.assertTrue(result.cuda)
        self.assertTrue(result.engine)

    def test_classify_maps_accepts_engine_only_worker(self):
        result = classify_maps("/repo/cg/libcg.so\n/usr/lib/libstdc++.so")
        self.assertFalse(result.torch)
        self.assertFalse(result.cuda)
        self.assertTrue(result.engine)


if __name__ == "__main__":
    unittest.main()
