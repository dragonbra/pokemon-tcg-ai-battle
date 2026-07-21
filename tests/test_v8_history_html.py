from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.render_v8_history import build_history_pages


ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = ROOT / "work" / "auto-iteration" / "history_iterations" / "v8-semantic-recovery"


class V8HistoryHtmlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads((HISTORY_DIR / "summary.json").read_text(encoding="utf-8"))

    def test_build_history_pages_renders_full_and_focused_records(self) -> None:
        pages = build_history_pages(self.summary, HISTORY_DIR)

        self.assertIn("index.html", pages)
        self.assertIn("target-baseline/index.html", pages)
        self.assertIn("start-baseline/index.html", pages)
        self.assertIn("iteration-004/index.html", pages)
        self.assertIn("118/50/2", pages["target-baseline/index.html"])
        self.assertIn("8/128/34", pages["start-baseline/index.html"])
        self.assertIn("17×10", pages["target-baseline/index.html"])
        self.assertIn("focused", pages["iteration-004/index.html"])
        self.assertIn("attackId=1072", pages["index.html"])


if __name__ == "__main__":
    unittest.main()
