from __future__ import annotations

import copy
import importlib
import unittest

schema = importlib.import_module("train.0013_semantic_goal_policy.features.schema")
observation = importlib.import_module("train.0013_semantic_goal_policy.features.observation")


def actor_observation() -> dict[str, object]:
    return {
        "current": {
            "yourIndex": 0,
            "firstPlayer": 0,
            "turn": 3,
            "turnActionCount": 2,
            "energyAttached": False,
            "retreated": False,
            "stadiumPlayed": False,
            "supporterPlayed": True,
            "players": [
                {"active": [{"id": 30, "serial": 1, "hp": 200, "maxHp": 270, "energies": [2]}], "bench": [], "benchMax": 5, "deckCount": 45, "discard": [], "hand": [{"id": 1, "serial": 2}], "handCount": 1, "prize": []},
                {"active": [], "bench": [], "benchMax": 5, "deckCount": 50, "discard": [], "hand": [], "handCount": 4, "prize": []},
            ],
            "stadium": [],
            "looking": [],
        },
        "select": {"type": 1, "context": 0, "effect": 0, "minCount": 1, "maxCount": 1, "option": [{"type": 1, "cardId": 30, "serial": 1}]},
        "logs": [],
    }


class TypedSchemaTest(unittest.TestCase):
    def test_encodes_named_typed_tokens_and_relations(self) -> None:
        typed = observation.encode_observation(actor_observation(), registered_deck=[1] * 30 + [30] * 30)
        typed.validate()
        self.assertEqual(typed.schema_version, schema.SCHEMA_VERSION)
        self.assertEqual(len(typed.registered_deck), 2)
        self.assertTrue(any(relation.kind == schema.RelationType.LOCATED_IN for relation in typed.relations))
        self.assertEqual(len(typed.options), 1)

    def test_unknown_missing_not_applicable_padding_and_overflow_are_independent(self) -> None:
        value = schema.NumericFeature.unknown()
        self.assertEqual(value.epistemic, schema.EpistemicState.UNKNOWN)
        self.assertFalse(value.padding)
        clipped = schema.NumericFeature.observed(10_000, cap=100)
        self.assertEqual(clipped.value, 100)
        self.assertTrue(clipped.overflow)
        self.assertNotEqual(schema.NumericFeature.missing(), schema.NumericFeature.not_applicable())
        self.assertNotEqual(schema.NumericFeature.padding_value(), value)

    def test_rejects_outcome_future_opponent_deck_and_serial_as_numeric(self) -> None:
        for key in ("terminal_outcome", "future_frame", "opponent_deck"):
            raw = actor_observation()
            raw[key] = []
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "forbidden"):
                observation.encode_observation(raw, registered_deck=[1] * 60)
        typed = observation.encode_observation(actor_observation(), registered_deck=[1] * 60)
        entity = next(item for item in typed.entities if item.card_id is not None)
        self.assertNotIn("serial", entity.numeric)
        self.assertIsNotNone(entity.instance_key)

    def test_invalid_relation_endpoint_and_duplicate_deck_multiplicity_fail(self) -> None:
        typed = observation.encode_observation(actor_observation(), registered_deck=[1] * 60)
        invalid = copy.deepcopy(typed)
        invalid.relations += (schema.Relation(schema.RelationType.TARGETS, "missing", invalid.state.token_id),)
        with self.assertRaisesRegex(ValueError, "relation endpoint"):
            invalid.validate()


if __name__ == "__main__":
    unittest.main()
