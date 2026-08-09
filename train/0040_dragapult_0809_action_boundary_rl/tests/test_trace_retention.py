from __future__ import annotations

import importlib
import unittest

worker = importlib.import_module("train.0040_dragapult_0809_action_boundary_rl.rollout.pool_worker")


class TraceRetentionTest(unittest.TestCase):
    def test_compaction_keeps_protocol_and_drops_bulk_state(self):
        event = {"primitive_select": [1], "gate": "FORCED", "observation": {
            "current": {"turn": 3, "yourIndex": 0, "players": [{"hand": list(range(30))}]},
            "select": {"type": 0, "context": 14, "option": list(range(100)),
                       "remainDamageCounter": 4},
        }}
        compact = worker._compact_event(event)
        self.assertEqual(compact["primitive_select"], [1])
        self.assertEqual(compact["observation"]["select"]["remainDamageCounter"], 4)
        self.assertNotIn("players", compact["observation"]["current"])
        self.assertNotIn("option", compact["observation"]["select"])


if __name__ == "__main__":
    unittest.main()
