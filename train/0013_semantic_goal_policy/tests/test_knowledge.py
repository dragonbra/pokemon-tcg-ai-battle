from __future__ import annotations

import importlib
import unittest

state = importlib.import_module("train.0013_semantic_goal_policy.knowledge.state")
visibility = importlib.import_module("train.0013_semantic_goal_policy.knowledge.visibility")


def observation(*, deck_view=None, hand_count=3, logs=None):
    return {
        "current": {
            "yourIndex": 0,
            "turn": 2,
            "players": [
                {"active": [], "bench": [], "discard": [], "hand": [], "handCount": 0, "prize": [], "deckCount": 54},
                {"active": [], "bench": [], "discard": [], "hand": [], "handCount": hand_count, "prize": [], "deckCount": 50},
            ],
        },
        "select": {"type": 1, "context": 0, "minCount": 1, "maxCount": 1, "option": [], **({"deck": deck_view} if deck_view is not None else {})},
        "logs": logs or [],
    }


class CausalKnowledgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.knowledge = state.CausalKnowledgeState.new_game(0, [1] * 30 + [2] * 30)

    def test_prize_unknown_before_full_view_and_unlocks_only_next_decision(self) -> None:
        before = self.knowledge.consume(observation())
        self.assertEqual(before.self_ledger[1].prize.state, state.KnowledgeStage.UNKNOWN)
        current = self.knowledge.consume(observation(deck_view=[{"id": 1}] * 27 + [{"id": 2}] * 27))
        self.assertEqual(current.self_ledger[1].prize.state, state.KnowledgeStage.UNKNOWN)
        after = self.knowledge.consume(observation())
        self.assertEqual(after.self_ledger[1].prize.state, state.KnowledgeStage.INFERRED_EXACT)
        self.assertEqual(after.self_ledger[1].prize.value, 3)

    def test_pending_action_does_not_mutate_current_state(self) -> None:
        snapshot = self.knowledge.consume(observation())
        self.knowledge.record_pending([0], snapshot.decision_index)
        same = self.knowledge.encode_current()
        self.assertEqual(snapshot.self_ledger, same.self_ledger)
        with self.assertRaisesRegex(ValueError, "pending"):
            self.knowledge.record_pending([0], snapshot.decision_index)

    def test_opponent_known_hand_lifecycle_and_hidden_draw(self) -> None:
        self.knowledge.consume(observation(hand_count=2))
        viewed = self.knowledge.consume(observation(hand_count=2, logs=[{"type":"hand_view","playerIndex":1,"cards":[{"id":30,"serial":8},{"id":1,"serial":9}]}]))
        self.assertEqual(len(viewed.opponent_hand.known), 2)
        moved = self.knowledge.consume(observation(hand_count=2, logs=[{"type":"move","playerIndex":1,"serial":8,"cardId":30,"fromArea":"hand","toArea":"discard"},{"type":"draw","playerIndex":1}]))
        self.assertEqual([card.card_id for card in moved.opponent_hand.known], [1])
        self.assertEqual(moved.opponent_hand.unknown_slots, 1)

    def test_shuffle_degrades_order_not_membership_and_new_game_resets(self) -> None:
        self.knowledge.consume(observation(deck_view=[{"id": 1}] * 27 + [{"id": 2}] * 27))
        self.knowledge.consume(observation())
        shuffled = self.knowledge.consume(observation(logs=[{"type":"shuffle","playerIndex":0,"area":"deck"}]))
        self.assertFalse(shuffled.deck_order_known)
        self.assertTrue(shuffled.deck_membership_known)
        reset = state.CausalKnowledgeState.new_game(0, [1] * 60).encode_current()
        self.assertEqual(reset.decision_index, 0)
        self.assertFalse(reset.deck_membership_known)

    def test_visibility_contexts_fail_closed(self) -> None:
        self.assertEqual(visibility.classify_deck_view({"deck":[{"id":1}],"option":[],"context":0}), visibility.ViewKind.ORDERED_VIEW)
        with self.assertRaisesRegex(ValueError, "unclassified"):
            visibility.classify_log({"type":"brand_new_hidden_event"})


if __name__ == "__main__":
    unittest.main()
