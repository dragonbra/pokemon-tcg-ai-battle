from __future__ import annotations

import importlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
REPLAY = ROOT / "data/replays/0016_alakazam_multideck_bc/goonew/submission-54960905/episode-88310066-replay.json"
compiler_module = importlib.import_module("train.0040_dragapult_0809_action_boundary_rl.rollout.worker_compiler")


class ObserveOnlyReplayParityTest(unittest.TestCase):
    def test_macro_shortcut_preserves_next_history_and_features(self) -> None:
        payload = json.loads(REPLAY.read_text(encoding="utf-8"))
        decks = payload["steps"][0][0]["visualize"][0]["action"]
        observations = []
        macro_actor = None
        for step in payload["steps"]:
            for agent in step:
                observation = agent.get("observation") or {}
                current, select = observation.get("current"), observation.get("select")
                if not isinstance(current, dict) or not isinstance(select, dict):
                    continue
                if select.get("remainDamageCounter") == 6:
                    macro_actor = current.get("yourIndex")
                observations.append(observation)
        self.assertIn(macro_actor, (0, 1))
        relevant = [
            item for item in observations
            if (item.get("current") or {}).get("yourIndex") == macro_actor
            and isinstance((item.get("select") or {}).get("option"), list)
            and (item.get("select") or {}).get("option")
        ]
        old = compiler_module.WorkerLocalCompiler(macro_actor, decks[macro_actor])
        shortcut = compiler_module.WorkerLocalCompiler(macro_actor, decks[macro_actor])
        inside = False
        compared = False
        for observation in relevant:
            remain = observation["select"].get("remainDamageCounter")
            old_record = old.compile(observation)
            if remain == 6:
                inside = True
            if inside and isinstance(remain, int) and 1 <= remain <= 6:
                shortcut.observe_only(observation)
                continue
            shortcut_record = shortcut.compile(observation)
            if inside:
                self.assertEqual(old_record, shortcut_record)
                compared = True
                break
        self.assertTrue(compared, "replay has no post-macro decision for parity comparison")


if __name__ == "__main__":
    unittest.main()
