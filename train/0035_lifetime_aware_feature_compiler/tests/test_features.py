from __future__ import annotations

import importlib
import copy
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
import unittest


BASE = "train.0035_lifetime_aware_feature_compiler"
PROTOTYPES = importlib.import_module(f"{BASE}.domain.prototypes")
COMPILER = importlib.import_module(f"{BASE}.features.compiler")
LAYERS = importlib.import_module(f"{BASE}.features.layers")
FIELDS = importlib.import_module(f"{BASE}.contracts.fields")
STATE = importlib.import_module(f"{BASE}.knowledge.state")
MODEL = importlib.import_module(f"{BASE}.model")
BENCHMARK = importlib.import_module(f"{BASE}.benchmark_incremental_features")


def _pokemon(card_id: int, *, player: int, serial: int, hp: int,
             energy_id: int | None = None, energy_units: list[int] | None = None) -> dict:
    energy = [] if energy_id is None else [{"id": energy_id, "playerIndex": player, "serial": serial + 100}]
    return {
        "id": card_id, "playerIndex": player, "serial": serial, "hp": hp, "maxHp": hp,
        "appearThisTurn": False, "energyCards": energy, "energies": energy_units or [],
        "tools": [], "preEvolution": [],
    }


def _snapshot(*, known_hand: tuple = ()) -> object:
    return STATE.CausalSnapshot(
        decision_index=0, perspective_actor=0, self_ledger=MappingProxyType({}),
        known_opponent_hand=known_hand, unknown_opponent_hand=0, recent_events=(),
        deck_membership_known=False, deck_order_known=False,
    )


def _row(energy_id: int = 15, *, energy_units: list[int] | None = None) -> tuple[dict, object]:
    own_active = _pokemon(96, player=0, serial=10, hp=210, energy_id=energy_id,
                          energy_units=energy_units if energy_units is not None else [11, 11])
    opponent_active = _pokemon(63, player=1, serial=20, hp=240)
    players = []
    for active in (own_active, opponent_active):
        players.append({
            "active": [active], "bench": [], "hand": [], "discard": [], "prize": [None] * 6,
            "deckCount": 40, "handCount": 0, "benchMax": 5,
            "asleep": False, "burned": False, "confused": False,
            "paralyzed": False, "poisoned": False,
        })
    row = {
        "actor_observation": {
            "current": {
                "yourIndex": 0, "firstPlayer": 0, "turn": 3, "turnActionCount": 2,
                "players": players, "stadium": [{"id": 1259, "playerIndex": 1, "serial": 70}],
                "looking": [], "supporterPlayed": False, "stadiumPlayed": False,
                "energyAttached": False, "retreated": False,
            },
            "select": {
                "type": 0, "context": 0, "contextCard": None, "effect": None, "deck": [],
                "option": [{"type": 13, "attackId": 120}, {"type": 14}],
                "minCount": 1, "maxCount": 1, "remainDamageCounter": 0, "remainEnergyCost": 0,
            },
        },
        "ordered_action": [0], "action_termination": "forced_max",
        "identity": {"date": "2026-08-02", "episode_id": 1, "player_index": 0},
        "split": "train", "deck_manifest": {"counts": [[1, 60]]},
    }
    return row, _snapshot()


class CanonicalFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prototypes = PROTOTYPES.PrototypeIndex.load(
            Path(f"{BASE.replace('.', '/')}/assets/official_public_prototypes_v1.json")
        )

    def test_layered_compiler_preserves_complete_python_record(self) -> None:
        decisions = 0
        first_cards = None
        for trajectory in BENCHMARK.load_parity_trajectories():
            knowledge = STATE.CausalKnowledge(trajectory.actor, trajectory.deck)
            for decision in trajectory.decisions:
                snapshot = knowledge.consume(decision.observation, decision.event_cursor)
                row = decision.row()
                expected = COMPILER.compile_canonical_row(row, snapshot, self.prototypes)
                cards = LAYERS.compile_card_layer(row, snapshot, self.prototypes)
                actual = LAYERS.assemble_canonical_record(
                    row,
                    cards,
                    LAYERS.compile_resource_layer(row, snapshot),
                    LAYERS.compile_event_layer(row, snapshot, cards),
                    LAYERS.compile_option_layer(row, snapshot, self.prototypes, cards),
                    LAYERS.compile_global_layer(row, snapshot),
                )
                self.assertEqual(actual, expected)
                self.assertEqual(
                    COMPILER.compile_canonical_layers(row, snapshot, self.prototypes),
                    expected,
                )
                first_cards = first_cards or cards
                decisions += 1
        self.assertEqual(decisions, 35)
        self.assertIsNotNone(first_cards)
        cards = first_cards
        self.assertIsInstance(cards.cat, tuple)
        self.assertIsInstance(cards.cat[0], tuple)
        with self.assertRaises(TypeError):
            cards.serial_locations[10] = 99

    def test_team_rocket_energy_preserves_physical_and_engine_semantics(self) -> None:
        row, snapshot = _row()
        record = COMPILER.compile_canonical_row(row, snapshot, self.prototypes)
        actor = record["actor"]
        energy_index = next(i for i, cat in enumerate(actor["card_cat"]) if cat[0] == 15)
        self.assertEqual(actor["card_parent"][energy_index], 1)
        active_index = actor["card_parent"][energy_index] - 1
        self.assertEqual(actor["card_num"][active_index][3], 2.0)
        units = [
            (cat, actor["card_parent"][index])
            for index, cat in enumerate(actor["card_cat"])
            if cat[5] == 9
        ]
        self.assertEqual([cat[7] for cat, _ in units], [12, 12])
        self.assertTrue(all(parent == active_index + 1 for _, parent in units))
        self.assertTrue(all(
            state == int(PROTOTYPES.FieldState.NOT_APPLICABLE)
            for state in actor["card_state"][energy_index][3:]
        ))
        engine = self.prototypes.engine_cards[15]
        self.assertEqual(engine["energy_type_mask"], 80)
        self.assertEqual(engine["energy_count"], 2)
        self.assertEqual(engine["only_team_rocket"], 1)

    def test_attached_energy_option_binds_physical_child(self) -> None:
        row, snapshot = _row()
        row["actor_observation"]["select"]["option"] = [
            {"type": 6, "playerIndex": 0, "area": 4, "index": 0, "energyIndex": 0}
        ]
        record = COMPILER.compile_canonical_row(row, snapshot, self.prototypes)
        actor = record["actor"]
        source = actor["option_source"][0] - 1
        self.assertEqual(actor["card_cat"][source][0], 15)
        self.assertEqual(actor["card_cat"][source][5], 3)

    def test_no_forbidden_derived_energy_or_damage_answers(self) -> None:
        forbidden = {
            "typed_energy_deficit", "total_energy_deficit", "surplus_energy",
            "newly_enabled", "base_damage_is_ko", "target_hp_after_base_damage",
            "selected_energy_type", "preferred_energy_removal",
        }
        self.assertTrue(forbidden.isdisjoint(FIELDS.OPTION_NUM_FIELDS))
        self.assertTrue(forbidden.isdisjoint(FIELDS.OPTION_CAT_FIELDS))

    def test_resolved_energy_multiset_preserves_multiplicity(self) -> None:
        row, snapshot = _row(energy_units=[1, 1, 1, 6])
        actor = COMPILER.compile_canonical_row(row, snapshot, self.prototypes)["actor"]
        unit_types = [cat[7] for cat in actor["card_cat"] if cat[5] == 9]
        self.assertEqual(unit_types, [2, 2, 2, 7])

    def test_skill_options_bind_exact_serial_and_keep_ordinal(self) -> None:
        row, snapshot = _row()
        active = row["actor_observation"]["current"]["players"][0]["active"][0]
        bench = _pokemon(96, player=0, serial=11, hp=210)
        row["actor_observation"]["current"]["players"][0]["bench"] = [bench]
        row["actor_observation"]["select"]["option"] = [
            {"type": 15, "cardId": 96, "serial": active["serial"]},
            {"type": 15, "cardId": 96, "serial": bench["serial"]},
        ]
        actor = COMPILER.compile_canonical_row(row, snapshot, self.prototypes)["actor"]
        self.assertNotEqual(actor["option_source"][0], actor["option_source"][1])
        ordinal = FIELDS.OPTION_CAT_FIELDS.index("option_ordinal")
        serial = FIELDS.OPTION_CAT_FIELDS.index("source_serial")
        self.assertEqual([cat[ordinal] for cat in actor["option_cat"]], [1, 2])
        self.assertEqual([cat[serial] for cat in actor["option_cat"]], [11, 12])

    def test_attack_option_does_not_fabricate_instance_relations(self) -> None:
        row, snapshot = _row()
        actor = COMPILER.compile_canonical_row(row, snapshot, self.prototypes)["actor"]
        self.assertEqual(actor["option_source"][0], 0)
        self.assertEqual(actor["option_target"][0], 0)

    def test_switch_event_uses_official_identity_keys_and_relations(self) -> None:
        row, _ = _row()
        payload = MappingProxyType({
            "type": 8, "playerIndex": 0,
            "cardIdActive": 96, "serialActive": 10,
            "cardIdBench": 96, "serialBench": 11,
        })
        row["actor_observation"]["current"]["players"][0]["bench"] = [
            _pokemon(96, player=0, serial=11, hp=210)
        ]
        event = STATE.TypedEvent(0, 0, 0, 8, "switch", 0, None, None, None, False, payload)
        snapshot = _snapshot()
        snapshot = STATE.CausalSnapshot(
            decision_index=0, perspective_actor=0, self_ledger=snapshot.self_ledger,
            known_opponent_hand=(), unknown_opponent_hand=0, recent_events=(event,),
            deck_membership_known=False, deck_order_known=False,
        )
        actor = COMPILER.compile_canonical_row(row, snapshot, self.prototypes)["actor"]
        active_field = FIELDS.EVENT_CAT_FIELDS.index("active_card_id")
        bench_field = FIELDS.EVENT_CAT_FIELDS.index("bench_card_id")
        self.assertEqual(actor["event_cat"][0][active_field], 96)
        self.assertEqual(actor["event_cat"][0][bench_field], 96)
        self.assertNotEqual(actor["event_source"][0], actor["event_target"][0])

    def test_full_deck_order_persists_through_draw_and_invalidates_on_shuffle(self) -> None:
        row, _ = _row()
        observation = row["actor_observation"]
        observation["current"]["players"][0]["deckCount"] = 2
        observation["select"]["deck"] = [
            {"id": 1, "serial": 101, "playerIndex": 0},
            {"id": 2, "serial": 102, "playerIndex": 0},
        ]
        observation["logs"] = []
        knowledge = STATE.CausalKnowledge(0, [1, 2] + [96] * 58)
        first = knowledge.consume(observation)
        self.assertTrue(first.deck_order_known)
        self.assertEqual([item.card_id for item in first.known_self_deck_order], [1, 2])

        after_draw = copy.deepcopy(observation)
        after_draw["current"]["players"][0]["deckCount"] = 1
        after_draw["select"]["deck"] = []
        after_draw["logs"] = [{"type": 4, "playerIndex": 0, "cardId": 1, "serial": 101}]
        second = knowledge.consume(after_draw)
        self.assertEqual([item.card_id for item in second.known_self_deck_order], [2])

        shuffled = copy.deepcopy(after_draw)
        shuffled["logs"] = [{"type": 0, "playerIndex": 0}]
        third = knowledge.consume(shuffled)
        self.assertFalse(third.deck_order_known)

    def test_ambiguous_opponent_hand_departure_becomes_candidate_set(self) -> None:
        row, _ = _row()
        observation = row["actor_observation"]
        observation["current"]["players"][1]["handCount"] = 2
        observation["logs"] = [
            {"type": 6, "playerIndex": 1, "cardId": 321, "serial": 88,
             "fromArea": 1, "toArea": 2},
            {"type": 5, "playerIndex": 1},
        ]
        knowledge = STATE.CausalKnowledge(0, [1] * 60)
        first = knowledge.consume(observation)
        self.assertEqual([item.card_id for item in first.known_opponent_hand], [321])

        departed = copy.deepcopy(observation)
        departed["current"]["players"][1]["handCount"] = 1
        departed["logs"] = [
            {"type": 7, "playerIndex": 1, "fromArea": 2, "toArea": 1}
        ]
        second = knowledge.consume(departed)
        self.assertFalse(second.known_opponent_hand)
        self.assertEqual([item.card_id for item in second.possible_opponent_hand], [321])
        self.assertEqual(
            (second.possible_opponent_hand_known_lower,
             second.possible_opponent_hand_known_upper),
            (0, 1),
        )

    def test_bench_max_stadium_owner_and_known_opponent_hand_are_visible(self) -> None:
        known = STATE.KnownOpponentCard(card_id=321, serial=88, source_event=4)
        row, _ = _row()
        actor = COMPILER.compile_canonical_row(row, _snapshot(known_hand=(known,)), self.prototypes)["actor"]
        self.assertEqual(actor["global_num"][17:19], [5.0, 5.0])
        stadium = next(cat for cat in actor["card_cat"] if cat[0] == 1259)
        remembered = next(cat for cat in actor["card_cat"] if cat[0] == 321)
        self.assertEqual(stadium[2], 2)
        self.assertEqual(remembered[3], COMPILER.ZONE["known_opponent_hand"])

    def test_target_area_identity_and_conditions_enter_prototype_tables(self) -> None:
        config = MODEL.ModelConfig(d_model=32, heads=4, state_layers=1, option_layers=1)
        encoder = MODEL.OfficialPrototypeEncoder(config, self.prototypes)
        effect_id, effect = next(
            (identity, value) for identity, value in self.prototypes.effects.items()
            if value["target"]["areas"] and value["target"]["conditions"]
        )
        # Effect prefix + boolean fields; target begins with player/not_me/skip, then exact areas.
        area_start = 8 + len(importlib.import_module(f"{BASE}.model.prototype_encoder").EFFECT_BOOLEANS) + 3
        encoded_areas = encoder.effect_cat_table[effect_id, area_start:area_start + 4].tolist()
        expected = [area + 1 for area in effect["target"]["areas"]] + [0] * (4 - len(effect["target"]["areas"]))
        self.assertEqual(encoded_areas, expected)
        condition_start = area_start + 4
        self.assertEqual(int(encoder.effect_cat_table[effect_id, condition_start]),
                         int(effect["target"]["conditions"][0]["target_type"]) + 1)

    def test_fail_skip_keeps_exact_control_flow_magnitude(self) -> None:
        config = MODEL.ModelConfig(d_model=32, heads=4, state_layers=1, option_layers=1)
        encoder = MODEL.OfficialPrototypeEncoder(config, self.prototypes)
        by_skip = {
            int(effect["fail_skip"]): identity
            for identity, effect in self.prototypes.effects.items()
            if int(effect["fail_skip"]) in {1, 2, 3}
        }
        self.assertTrue({1, 2, 3}.issubset(by_skip))
        self.assertEqual(
            [float(encoder.effect_num_table[by_skip[value], 7]) for value in (1, 2, 3)],
            [1.0, 2.0, 3.0],
        )

    def test_prototype_collection_overflow_fails_closed(self) -> None:
        attacks = dict(self.prototypes.engine_attacks)
        attack_id, attack = next(iter(attacks.items()))
        attacks[attack_id] = {**attack, "energies": [0] * 8}
        invalid = replace(self.prototypes, engine_attacks=MappingProxyType(attacks))
        config = MODEL.ModelConfig(d_model=32, heads=4, state_layers=1, option_layers=1)
        with self.assertRaisesRegex(ValueError, "prototype collection capacity exceeded"):
            MODEL.OfficialPrototypeEncoder(config, invalid)


if __name__ == "__main__":
    unittest.main()
