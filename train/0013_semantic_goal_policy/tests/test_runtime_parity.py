from __future__ import annotations

import copy
import importlib
import unittest

session = importlib.import_module("train.0013_semantic_goal_policy.runtime.session")


def obs(turn, *, logs=None, deck=None):
    return {
        "current": {
            "yourIndex": 0, "firstPlayer": 0, "turn": turn, "turnActionCount": 1,
            "players": [
                {"active": [], "bench": [], "benchMax": 5, "deckCount": 54, "discard": [], "hand": [], "handCount": 0, "prize": []},
                {"active": [], "bench": [], "benchMax": 5, "deckCount": 50, "discard": [], "hand": [], "handCount": 3, "prize": []},
            ], "stadium": [], "looking": [],
        },
        "select": {"type": 1, "context": 0, "minCount": 1, "maxCount": 1, "option": [{"type": 1, "cardId": 1}], **({"deck": deck} if deck is not None else {})},
        "logs": logs or [],
    }


class RuntimeParityTest(unittest.TestCase):
    def test_offline_and_online_typed_hashes_match_each_decision(self):
        observations = [obs(1), obs(1, deck=[{"id": 1}] * 54), obs(2, logs=[{"type": "shuffle", "playerIndex": 0}])]
        actions = [[0], [0], [0]]
        offline = session.replay_session(0, [1] * 60, observations, actions)
        online = session.PolicySession.new_game(0, [1] * 60)
        hashes = []
        for observation, action in zip(observations, actions):
            encoded = online.observe(observation)
            hashes.append(encoded.sha256)
            online.record_action(action)
        self.assertEqual([item.sha256 for item in offline], hashes)

    def test_pending_timing_reset_schema_and_unknown_log(self):
        runtime = session.PolicySession.new_game(0, [1] * 60)
        first = runtime.observe(obs(1))
        runtime.record_action([0])
        second = runtime.observe(obs(1))
        self.assertNotEqual(first.decision_index, second.decision_index)
        reset = session.PolicySession.new_game(0, [1] * 60).observe(obs(1))
        self.assertEqual(reset.decision_index, 0)
        malformed = obs(1, logs=[{"type": "unknown_new_log"}])
        with self.assertRaisesRegex(ValueError, "unclassified"):
            session.PolicySession.new_game(0, [1] * 60).observe(malformed)
        with self.assertRaisesRegex(ValueError, "schema"):
            session.TypedInputEnvelope.from_dict({"schema_version": "wrong", "payload": {}})


if __name__ == "__main__":
    unittest.main()
