from __future__ import annotations

import copy
import importlib
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
import unittest


BASE = "train.0031_rule_faithful_semantic_foundation_pretraining"
COMPILER = importlib.import_module(f"{BASE}.features.compiler")
INCREMENTAL = importlib.import_module(f"{BASE}.features.incremental")
PROTOTYPES = importlib.import_module(f"{BASE}.domain.prototypes")
STATE = importlib.import_module(f"{BASE}.knowledge.state")
BENCHMARK = importlib.import_module(f"{BASE}.benchmark_incremental_features")


def _pokemon(card_id: int, *, player: int, serial: int, hp: int) -> dict:
    return {
        "id": card_id,
        "playerIndex": player,
        "serial": serial,
        "hp": hp,
        "maxHp": hp,
        "appearThisTurn": False,
        "energyCards": [],
        "energies": [],
        "tools": [],
        "preEvolution": [],
    }


def _row() -> dict:
    players = []
    for player, active in enumerate((
        _pokemon(96, player=0, serial=10, hp=210),
        _pokemon(63, player=1, serial=20, hp=240),
    )):
        players.append({
            "active": [active],
            "bench": [],
            "hand": [],
            "discard": [],
            "prize": [None] * 6,
            "deckCount": 40,
            "handCount": 0,
            "benchMax": 5,
            "asleep": False,
            "burned": False,
            "confused": False,
            "paralyzed": False,
            "poisoned": False,
        })
    return {
        "actor_observation": {
            "current": {
                "yourIndex": 0,
                "firstPlayer": 0,
                "turn": 3,
                "turnActionCount": 2,
                "players": players,
                "stadium": [],
                "looking": [],
                "supporterPlayed": False,
                "stadiumPlayed": False,
                "energyAttached": False,
                "retreated": False,
            },
            "select": {
                "type": 0,
                "context": 0,
                "contextCard": None,
                "effect": None,
                "deck": [],
                "option": [{"type": 13, "attackId": 120}, {"type": 14}],
                "minCount": 1,
                "maxCount": 1,
                "remainDamageCounter": 0,
                "remainEnergyCost": 0,
            },
        },
        "ordered_action": [0],
        "action_termination": "forced_max",
        "identity": {"episode_id": "incremental-test", "player_index": 0},
        "split": "validation",
        "deck_manifest": {"counts": [[1, 59], [96, 1]]},
    }


def _snapshot(decision_index: int = 0, *, events: tuple = ()):
    return STATE.CausalSnapshot(
        decision_index=decision_index,
        perspective_actor=0,
        self_ledger=MappingProxyType({}),
        known_opponent_hand=(),
        unknown_opponent_hand=0,
        recent_events=events,
        deck_membership_known=False,
        deck_order_known=False,
    )


class IncrementalFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assets = Path(BASE.replace(".", "/")) / "assets"
        cls.prototypes = PROTOTYPES.PrototypeIndex.load(
            assets / "official_public_prototypes_v1.json",
            assets / "official_full_engine_prototypes_v2.json",
        )

    def assert_full_parity(self, incremental, row, snapshot) -> None:
        self.assertEqual(
            incremental.compile(row, snapshot),
            COMPILER.compile_canonical_row(row, snapshot, self.prototypes),
        )

    def test_dirty_graph_closes_card_relations_and_keeps_option_change_local(self) -> None:
        first_row = _row()
        option_row = copy.deepcopy(first_row)
        option_row["actor_observation"]["select"]["option"][0]["attackId"] = 121
        card_row = copy.deepcopy(option_row)
        card_row["actor_observation"]["current"]["players"][0]["active"][0]["hp"] = 180

        first = INCREMENTAL.project_layer_dependencies(first_row, _snapshot(0))
        option = INCREMENTAL.project_layer_dependencies(option_row, _snapshot(1))
        cards = INCREMENTAL.project_layer_dependencies(card_row, _snapshot(2))

        self.assertEqual(INCREMENTAL.dirty_layers(first, option), {"options"})
        self.assertEqual(
            INCREMENTAL.dirty_layers(option, cards),
            {"cards", "events", "options"},
        )

    def test_sequential_layer_reuse_preserves_complete_records(self) -> None:
        incremental = INCREMENTAL.IncrementalCanonicalCompiler(self.prototypes)
        rows = []
        rows.append(_row())
        rows.append(copy.deepcopy(rows[-1]))
        rows[-1]["actor_observation"]["select"]["option"][0]["attackId"] = 121
        rows.append(copy.deepcopy(rows[-1]))
        rows[-1]["actor_observation"]["current"]["turnActionCount"] = 3
        rows.append(copy.deepcopy(rows[-1]))
        rows[-1]["actor_observation"]["current"]["players"][0]["active"][0]["hp"] = 170

        for index, row in enumerate(rows):
            self.assert_full_parity(incremental, row, _snapshot(index))

        stats = incremental.stats.snapshot()
        self.assertEqual(stats["decisions"], len(rows))
        for layer in INCREMENTAL.LAYER_NAMES:
            self.assertEqual(
                stats[f"layer/{layer}/hit"] + stats[f"layer/{layer}/miss"],
                len(rows),
            )
            self.assertGreater(stats[f"layer/{layer}/hit"], 0)

    def test_causal_knowledge_chronology_has_full_record_parity(self) -> None:
        row = _row()
        knowledge = STATE.CausalKnowledge(0, [1] * 59 + [96])
        incremental = INCREMENTAL.IncrementalCanonicalCompiler(self.prototypes)

        for decision in range(3):
            current = copy.deepcopy(row)
            observation = current["actor_observation"]
            observation["current"]["turnActionCount"] = 2 + decision
            observation["logs"] = []
            snapshot = knowledge.consume(observation)
            self.assertEqual(snapshot.decision_index, decision)
            self.assert_full_parity(incremental, current, snapshot)

    def test_frozen_chronological_fixture_has_zero_fallback_and_exact_parity(self) -> None:
        total_decisions = 0
        for trajectory in BENCHMARK.load_parity_trajectories():
            knowledge = STATE.CausalKnowledge(trajectory.actor, trajectory.deck)
            incremental = INCREMENTAL.IncrementalCanonicalCompiler(self.prototypes)
            for decision in trajectory.decisions:
                snapshot = knowledge.consume(decision.observation, decision.event_cursor)
                self.assert_full_parity(incremental, decision.row(), snapshot)
                total_decisions += 1
            stats = incremental.stats.snapshot()
            self.assertEqual(stats["fallbacks"], 0)
            for layer in INCREMENTAL.LAYER_NAMES:
                self.assertEqual(
                    stats[f"layer/{layer}/hit"] + stats[f"layer/{layer}/miss"],
                    len(trajectory.decisions),
                )
        self.assertGreater(total_decisions, 0)
        # This frozen game changes every canonical layer on every decision.  The
        # synthetic transition test above independently proves hits for each
        # layer; this real chronology is the strict no-fallback parity gate.

    def test_card_insertion_rebuilds_relation_layers_and_preserves_targets(self) -> None:
        incremental = INCREMENTAL.IncrementalCanonicalCompiler(self.prototypes)
        first = _row()
        self.assert_full_parity(incremental, first, _snapshot(0))

        second = copy.deepcopy(first)
        second["actor_observation"]["current"]["players"][0]["bench"] = [
            _pokemon(96, player=0, serial=11, hp=210)
        ]
        second["actor_observation"]["select"]["option"] = [{
            "type": 1,
            "playerIndex": 0,
            "area": 2,
            "index": 0,
            "inPlayArea": 5,
            "inPlayIndex": 0,
        }]
        self.assert_full_parity(incremental, second, _snapshot(1))
        record = incremental.compile(copy.deepcopy(second), _snapshot(2))
        self.assertGreater(record["actor"]["option_target"][0], 0)

    def test_known_deck_order_transition_invalidates_every_consumer(self) -> None:
        row = _row()
        before = _snapshot(0)
        after = replace(
            _snapshot(1),
            deck_membership_known=True,
            deck_order_known=True,
            known_self_deck_order=(STATE.KnownDeckCard(1, 101),),
        )
        previous = INCREMENTAL.project_layer_dependencies(row, before)
        current = INCREMENTAL.project_layer_dependencies(row, after)
        self.assertEqual(
            INCREMENTAL.dirty_layers(previous, current),
            {"cards", "resources", "events", "options", "globals"},
        )
        incremental = INCREMENTAL.IncrementalCanonicalCompiler(self.prototypes)
        self.assert_full_parity(incremental, row, before)
        self.assert_full_parity(incremental, row, after)

    def test_new_event_only_rebuilds_event_layer_and_preserves_relations(self) -> None:
        row = _row()
        payload = MappingProxyType({
            "type": 8,
            "playerIndex": 0,
            "cardIdActive": 96,
            "serialActive": 10,
            "cardIdBench": 96,
            "serialBench": 11,
        })
        event = STATE.TypedEvent(
            0, 1, 0, 8, "switch", 0, None, None, None, False, payload
        )
        previous = INCREMENTAL.project_layer_dependencies(row, _snapshot(0))
        current = INCREMENTAL.project_layer_dependencies(row, _snapshot(1, events=(event,)))
        self.assertEqual(INCREMENTAL.dirty_layers(previous, current), {"events"})

        incremental = INCREMENTAL.IncrementalCanonicalCompiler(self.prototypes)
        self.assert_full_parity(incremental, row, _snapshot(0))
        self.assert_full_parity(incremental, row, _snapshot(1, events=(event,)))

    def test_skipped_chronology_falls_back_to_stateless_full_compiler(self) -> None:
        incremental = INCREMENTAL.IncrementalCanonicalCompiler(self.prototypes)
        row = _row()
        self.assert_full_parity(incremental, row, _snapshot(0))
        self.assert_full_parity(incremental, row, _snapshot(2))
        stats = incremental.stats.snapshot()
        self.assertEqual(stats["fallback/non_chronological_decision"], 1)
        self.assertEqual(stats["fallbacks"], 1)

    def test_rewound_chronology_falls_back_to_stateless_full_compiler(self) -> None:
        incremental = INCREMENTAL.IncrementalCanonicalCompiler(self.prototypes)
        row = _row()
        self.assert_full_parity(incremental, row, _snapshot(1))
        self.assert_full_parity(incremental, row, _snapshot(0))
        self.assertEqual(
            incremental.stats.snapshot()["fallback/non_chronological_decision"], 1
        )

    def test_unclassified_projection_value_falls_back_and_counts_reason(self) -> None:
        incremental = INCREMENTAL.IncrementalCanonicalCompiler(self.prototypes)
        row = _row()
        row["actor_observation"]["current"]["looking"] = object()
        self.assert_full_parity(incremental, row, _snapshot(0))
        stats = incremental.stats.snapshot()
        self.assertEqual(stats["fallback/projection_TypeError"], 1)
        self.assertEqual(stats["full_rebuilds"], 1)

    def test_actor_schema_mismatch_falls_back_to_stateless_authority(self) -> None:
        incremental = INCREMENTAL.IncrementalCanonicalCompiler(self.prototypes)
        row = _row()
        row["actor_schema_version"] = "unexpected_actor_schema"
        self.assert_full_parity(incremental, row, _snapshot(0))
        self.assertEqual(incremental.stats.snapshot()["fallback/schema_mismatch"], 1)

    def test_reset_clears_cache_without_erasing_monotonic_statistics(self) -> None:
        incremental = INCREMENTAL.IncrementalCanonicalCompiler(self.prototypes)
        row = _row()
        self.assert_full_parity(incremental, row, _snapshot(0))
        incremental.reset("test_boundary")
        self.assert_full_parity(incremental, row, _snapshot(1))
        stats = incremental.stats.snapshot()
        self.assertEqual(stats["reset/test_boundary"], 1)
        self.assertEqual(stats["layer/cards/miss"], 2)


if __name__ == "__main__":
    unittest.main()
