from __future__ import annotations

import json
from pathlib import Path
import unittest

from engine_cuda.tools.benchmark_0037_cuda_resident_refill import json_ready


class JsonReadyTest(unittest.TestCase):
    def test_recursively_converts_paths_before_report_serialization(self) -> None:
        payload = {
            "checkpoint": Path("/tmp/checkpoint.pt"),
            "nested": [Path("/tmp/model.pt"), {"deck": Path("deck.csv")}],
        }

        normalized = json_ready(payload)

        self.assertEqual(normalized["checkpoint"], "/tmp/checkpoint.pt")
        self.assertEqual(normalized["nested"][1]["deck"], "deck.csv")
        json.dumps(normalized)


if __name__ == "__main__":
    unittest.main()
