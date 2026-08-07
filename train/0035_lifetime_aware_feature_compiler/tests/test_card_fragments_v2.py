from __future__ import annotations

import copy
import importlib
from pathlib import Path
from types import MappingProxyType
import unittest


BASE = "train.0035_lifetime_aware_feature_compiler"
BENCHMARK = importlib.import_module(f"{BASE}.benchmark_incremental_features")
EXTENDED = importlib.import_module(f"{BASE}.tests.extended_fixture")
FRAGMENTS = importlib.import_module(f"{BASE}.features.fragments")
INCREMENTAL = importlib.import_module(f"{BASE}.features.incremental")
LAYERS = importlib.import_module(f"{BASE}.features.layers")
PROTOTYPES = importlib.import_module(f"{BASE}.domain.prototypes")
STATE = importlib.import_module(f"{BASE}.knowledge.state")


def _pokemon(card: int, player: int, serial: int) -> dict:
    return {
        "id": card,
        "playerIndex": player,
        "serial": serial,
        "hp": 100,
        "maxHp": 100,
        "appearThisTurn": False,
        "energyCards": [],
        "energies": [],
        "tools": [],
        "preEvolution": [],
    }


def _row(*, actor: int = 0) -> dict:
    players = []
    for player in (0, 1):
        players.append({
            "active": [_pokemon(1 + player, player, 10 + player)],
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
                "yourIndex": actor,
                "firstPlayer": 0,
                "turn": 1,
                "turnActionCount": 0,
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
                "option": [{"type": 14}],
                "minCount": 1,
                "maxCount": 1,
                "remainDamageCounter": 0,
                "remainEnergyCost": 0,
            },
        },
        "ordered_action": [0],
        "action_termination": "forced_max",
        "identity": {"episode_id": "card-fragment-v2", "player_index": actor},
        "split": "validation",
        "deck_manifest": {"counts": [[1, 60]]},
    }


def _snapshot(actor: int = 0, decision: int = 0):
    return STATE.CausalSnapshot(
        decision_index=decision,
        perspective_actor=actor,
        self_ledger=MappingProxyType({}),
        known_opponent_hand=(),
        unknown_opponent_hand=0,
        recent_events=(),
        deck_membership_known=False,
        deck_order_known=False,
    )


class CardFragmentV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assets = Path(BASE.replace(".", "/")) / "assets"
        cls.prototypes = PROTOTYPES.PrototypeIndex.load(
            assets / "official_public_prototypes_v1.json",
            assets / "official_full_engine_prototypes_v2.json",
        )

    def assert_card_parity(self, compiler, row, snapshot) -> None:
        self.assertEqual(
            compiler.compile(row, snapshot),
            LAYERS.compile_card_layer(row, snapshot, self.prototypes),
        )

    def test_exact_card_layer_parity_for_35_and_525_chronological_decisions(self) -> None:
        fixture_groups = (
            (BENCHMARK.load_parity_trajectories(), 35),
            (EXTENDED.load_extended_trajectories(), 525),
        )
        for trajectories, expected_count in fixture_groups:
            compiled = 0
            for trajectory in trajectories:
                knowledge = STATE.CausalKnowledge(trajectory.actor, trajectory.deck)
                compiler = FRAGMENTS.CardFragmentCompiler()
                for decision in trajectory.decisions:
                    snapshot = knowledge.consume(
                        decision.observation, decision.event_cursor
                    )
                    self.assert_card_parity(compiler, decision.row(), snapshot)
                    compiled += 1
                self.assertGreater(compiler.hits, 0)
            self.assertEqual(compiled, expected_count)

    def test_state_and_placement_transitions_do_not_reuse_stale_rows(self) -> None:
        compiler = FRAGMENTS.CardFragmentCompiler()
        first = _row()
        self.assert_card_parity(compiler, first, _snapshot(decision=0))

        damaged = copy.deepcopy(first)
        damaged["actor_observation"]["current"]["players"][0]["active"][0]["hp"] = 70
        self.assert_card_parity(compiler, damaged, _snapshot(decision=1))

        moved = copy.deepcopy(damaged)
        player = moved["actor_observation"]["current"]["players"][0]
        player["bench"] = player.pop("active")
        player["active"] = []
        self.assert_card_parity(compiler, moved, _snapshot(decision=2))

        aliased = copy.deepcopy(moved)
        aliased["actor_observation"]["select"]["contextCard"] = {
            "cardId": 3,
            "serial": 99,
            "playerIndex": 0,
        }
        self.assert_card_parity(compiler, aliased, _snapshot(decision=3))
        aliased["actor_observation"]["select"]["contextCard"]["cardId"] = True
        self.assert_card_parity(compiler, aliased, _snapshot(decision=4))

        # Reusing and mutating the identical Python object must not bypass the
        # cached semantic projection.
        active = aliased["actor_observation"]["current"]["players"][0]["bench"][0]
        active["hp"] = 55
        self.assert_card_parity(compiler, aliased, _snapshot(decision=5))

        typed = copy.deepcopy(aliased)
        typed["actor_observation"]["current"]["players"][0]["bench"][0]["hp"] = True
        self.assert_card_parity(compiler, typed, _snapshot(decision=6))

    def test_child_owner_and_unbound_relation_match_stateless_authority(self) -> None:
        row = _row(actor=1)
        root = row["actor_observation"]["current"]["players"][1]["active"][0]
        root.pop("playerIndex")
        root["energyCards"] = [
            {"id": 4, "serial": 201, "playerIndex": 1},
            {},
        ]
        snapshot = _snapshot(actor=1)
        actual = FRAGMENTS.CardFragmentCompiler().compile(row, snapshot)
        expected = LAYERS.compile_card_layer(row, snapshot, self.prototypes)
        self.assertEqual(actual, expected)
        energy_rows = [value for value in actual.cat if value[5] == 3]
        self.assertEqual(energy_rows[0][2], 1)
        root_index = actual.serial_locations[11]
        self.assertEqual(actual.child_locations[(root_index, "energy", 1)], -1)

    def test_slot_overflow_fails_at_the_same_boundary(self) -> None:
        row = _row()
        row["actor_observation"]["current"]["players"][0]["hand"] = [
            {"id": 1, "serial": 1000 + index} for index in range(257)
        ]
        with self.assertRaisesRegex(ValueError, "card zone slot"):
            LAYERS.compile_card_layer(row, _snapshot(), self.prototypes)
        with self.assertRaisesRegex(ValueError, "card zone slot"):
            FRAGMENTS.CardFragmentCompiler().compile(row, _snapshot())

    def test_duplicate_serial_fails_closed_to_full_compiler(self) -> None:
        row = _row()
        row["actor_observation"]["current"]["players"][1]["active"][0]["serial"] = 10
        snapshot = _snapshot()
        expected = INCREMENTAL.compile_canonical_row(row, snapshot, self.prototypes)
        compiler = INCREMENTAL.IncrementalCanonicalCompiler(self.prototypes)
        self.assertEqual(compiler.compile(row, snapshot), expected)
        stats = compiler.stats.snapshot()
        self.assertEqual(stats["fallback/layer_ValueError"], 1)
        self.assertEqual(stats["fallbacks"], 1)


if __name__ == "__main__":
    unittest.main()
