from __future__ import annotations

import copy
import importlib
from types import MappingProxyType
import unittest


BASE = "train.0035_lifetime_aware_feature_compiler"
VERSIONS = importlib.import_module(f"{BASE}.features.observation_versions")
STATE = importlib.import_module(f"{BASE}.knowledge.state")


def _row() -> dict:
    def player(index: int, serial: int) -> dict:
        return {
            "active": [{
                "id": 96 if index == 0 else 63,
                "serial": serial,
                "playerIndex": index,
                "hp": 100,
                "maxHp": 100,
                "energyCards": [],
                "energies": [],
                "tools": [],
                "preEvolution": [],
            }],
            "bench": [], "hand": [], "discard": [],
            "asleep": False, "burned": False, "confused": False,
            "paralyzed": False, "poisoned": False,
        }
    return {
        "actor_observation": {
            "current": {
                "yourIndex": 0, "turn": 1, "turnActionCount": 0,
                "players": [player(0, 10), player(1, 20)],
                "stadium": [], "looking": [],
            },
            "select": {
                "type": 0, "context": 0, "contextCard": None,
                "effect": None, "deck": [],
                "option": [{"type": 14}], "minCount": 1, "maxCount": 1,
            },
        }
    }


def _snapshot(index: int, events: tuple = ()):
    return STATE.CausalSnapshot(
        decision_index=index,
        perspective_actor=0,
        self_ledger=MappingProxyType({}),
        known_opponent_hand=(),
        unknown_opponent_hand=0,
        recent_events=events,
        deck_membership_known=False,
        deck_order_known=False,
    )


class ObservationVersionTests(unittest.TestCase):
    def test_option_only_change_preserves_card_version(self) -> None:
        tracker = VERSIONS.ObservationVersionTracker(0)
        first = _row()
        v1 = tracker.observe(first, _snapshot(0))
        second = copy.deepcopy(first)
        second["actor_observation"]["select"]["option"] = [
            {"type": 13, "attackId": 120}
        ]
        v2 = tracker.observe(second, _snapshot(1))
        self.assertEqual(v2.cards, v1.cards)
        self.assertGreater(v2.selection, v1.selection)
        self.assertGreater(v2.globals, v1.globals)

    def test_hp_and_primitive_type_changes_advance_card_version(self) -> None:
        tracker = VERSIONS.ObservationVersionTracker(0)
        row = _row()
        first = tracker.observe(row, _snapshot(0))
        changed = copy.deepcopy(row)
        changed["actor_observation"]["current"]["players"][0]["active"][0]["hp"] = 90
        second = tracker.observe(changed, _snapshot(1))
        self.assertGreater(second.cards, first.cards)
        type_changed = copy.deepcopy(changed)
        type_changed["actor_observation"]["current"]["players"][0]["active"][0]["hp"] = True
        third = tracker.observe(type_changed, _snapshot(2))
        self.assertGreater(third.cards, second.cards)

    def test_same_object_mutation_is_not_hidden_by_identity(self) -> None:
        tracker = VERSIONS.ObservationVersionTracker(0)
        row = _row()
        first = tracker.observe(row, _snapshot(0))
        row["actor_observation"]["current"]["players"][0]["active"][0]["hp"] = 80
        second = tracker.observe(row, _snapshot(1))
        self.assertGreater(second.cards, first.cards)

    def test_event_identity_and_reset_are_session_owned(self) -> None:
        tracker = VERSIONS.ObservationVersionTracker(0)
        row = _row()
        event = STATE.TypedEvent(
            0, 0, 0, 2, "turn_start", 0, None, None, None, False,
            MappingProxyType({"type": 2, "playerIndex": 0}),
        )
        first = tracker.observe(row, _snapshot(0, (event,)))
        second = tracker.observe(copy.deepcopy(row), _snapshot(1, (event,)))
        self.assertEqual(second.events, first.events)
        tracker.reset()
        restarted = tracker.observe(copy.deepcopy(row), _snapshot(0, (event,)))
        self.assertEqual(restarted.cards, 1)
        self.assertEqual(restarted.events, 1)

    def test_non_chronological_input_fails_closed(self) -> None:
        tracker = VERSIONS.ObservationVersionTracker(0)
        tracker.observe(_row(), _snapshot(0))
        with self.assertRaises(VERSIONS.ObservationVersionError):
            tracker.observe(_row(), _snapshot(2))


if __name__ == "__main__":
    unittest.main()
