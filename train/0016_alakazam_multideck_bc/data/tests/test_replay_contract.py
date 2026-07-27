from __future__ import annotations

from importlib import import_module
import unittest


CONTRACT = import_module("train.0016_alakazam_multideck_bc.data.replay_contract")


def _observation() -> dict[str, object]:
    return {
        "current": {
            "yourIndex": 0,
            "firstPlayer": 0,
            "players": [
                {"active": [], "bench": [], "discard": [], "hand": [], "prize": []},
                {"active": [], "bench": [], "discard": [], "hand": [], "prize": []},
            ],
        },
        "logs": [],
        "select": {"option": [{"type": 0}], "minCount": 1, "maxCount": 1},
    }


class ReplayContractTest(unittest.TestCase):
    def test_accepts_missing_nondecision_terminal_visual_tail(self) -> None:
        observation = _observation()
        payload = {
            "steps": [
                [
                    {
                        "visualize": [
                            {"selected": None},
                            {
                                "selected": [0],
                                "action": [[0], []],
                                "obs": observation,
                            },
                        ]
                    },
                    {},
                ],
                [
                    {"status": "ACTIVE", "observation": observation},
                    {"status": "INACTIVE"},
                ],
                [{"action": [0], "duration": 1.0}, {"action": []}],
                [{"status": "DONE", "action": []}, {"status": "TIMEOUT"}],
            ]
        }
        frames = list(CONTRACT.decision_frames(payload, 0))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][2], (0,))


if __name__ == "__main__":
    unittest.main()
