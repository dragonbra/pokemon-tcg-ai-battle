from __future__ import annotations

import importlib
from pathlib import Path
from types import MappingProxyType
import unittest

import torch


BASE = "train.0033_effect_summary_semantic_foundation_pretraining"
PROTOTYPES = importlib.import_module(f"{BASE}.domain.prototypes")
COMPILER = importlib.import_module(f"{BASE}.features.compiler")
FEATURE_AUDIT = importlib.import_module(f"{BASE}.features.audit")
FIELDS = importlib.import_module(f"{BASE}.contracts.fields")
STATE = importlib.import_module(f"{BASE}.knowledge.state")
MODEL = importlib.import_module(f"{BASE}.model")
PROTOTYPE_ENCODER = importlib.import_module(f"{BASE}.model.prototype_encoder")


def _pokemon(
    card_id: int,
    *,
    player: int,
    serial: int,
    hp: int,
    energy_id: int | None = None,
    energy_units: list[int] | None = None,
) -> dict:
    energy = (
        []
        if energy_id is None
        else [{"id": energy_id, "playerIndex": player, "serial": serial + 100}]
    )
    return {
        "id": card_id,
        "playerIndex": player,
        "serial": serial,
        "hp": hp,
        "maxHp": hp,
        "appearThisTurn": False,
        "energyCards": energy,
        "energies": energy_units or [],
        "tools": [],
        "preEvolution": [],
    }


def _snapshot(*, known_hand: tuple = (), events: tuple = ()) -> object:
    return STATE.CausalSnapshot(
        decision_index=0,
        perspective_actor=0,
        self_ledger=MappingProxyType({}),
        known_opponent_hand=known_hand,
        unknown_opponent_hand=0,
        recent_events=events,
    )


def _row(
    energy_id: int = 15,
    *,
    energy_units: list[int] | None = None,
) -> tuple[dict, object]:
    own_active = _pokemon(
        96,
        player=0,
        serial=10,
        hp=210,
        energy_id=energy_id,
        energy_units=energy_units if energy_units is not None else [11, 11],
    )
    opponent_active = _pokemon(63, player=1, serial=20, hp=240)
    players = []
    for active in (own_active, opponent_active):
        players.append(
            {
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
            }
        )
    row = {
        "actor_observation": {
            "current": {
                "yourIndex": 0,
                "firstPlayer": 0,
                "turn": 3,
                "turnActionCount": 2,
                "players": players,
                "stadium": [{"id": 1259, "playerIndex": 1, "serial": 70}],
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
        "identity": {"date": "2026-08-02", "episode_id": 1, "player_index": 0},
        "split": "train",
        "deck_manifest": {"counts": [[1, 60]]},
    }
    return row, _snapshot()


class CanonicalFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prototypes = PROTOTYPES.PrototypeIndex.load(
            Path(f"{BASE.replace('.', '/')}/assets/official_public_prototypes_v1.json")
        )

    def test_compiled_feature_audit_matches_observation_energy_exactly(self) -> None:
        row, snapshot = _row(energy_units=[0, 1, 1, 11])
        compiled = COMPILER.compile_canonical_row(row, snapshot, self.prototypes)

        counts = FEATURE_AUDIT.audit_compiled_feature_input(row, compiled, snapshot)

        self.assertEqual(counts["audited_decisions"], 1)
        self.assertEqual(counts["in_play_pokemon"], 2)
        self.assertEqual(counts["attached_children"], 1)

        compiled["actor"]["card_num"][0][2] += 1.0
        with self.assertRaisesRegex(ValueError, "histogram differs"):
            FEATURE_AUDIT.audit_compiled_feature_input(row, compiled, snapshot)

    def test_team_rocket_energy_keeps_physical_relation_and_engine_semantics(self) -> None:
        row, snapshot = _row()
        actor = COMPILER.compile_canonical_row(row, snapshot, self.prototypes)["actor"]
        energy_index = next(i for i, cat in enumerate(actor["card_cat"]) if cat[0] == 15)
        active_index = actor["card_parent"][energy_index] - 1

        self.assertEqual(active_index, 0)
        histogram = actor["card_num"][active_index][2:]
        self.assertEqual(histogram, [0.0] * 11 + [2.0])
        self.assertEqual(sum(histogram), 2.0)
        self.assertEqual(
            actor["card_state"][energy_index][2:],
            [int(PROTOTYPES.FieldState.NOT_APPLICABLE)] * FIELDS.ENERGY_TYPE_COUNT,
        )
        self.assertEqual(actor["card_cat"][active_index][4], 1)
        self.assertEqual(actor["card_cat"][energy_index][4], 0)

        engine = self.prototypes.engine_cards[15]
        self.assertEqual(engine["energy_type_mask"], 80)
        self.assertEqual(engine["energy_count"], 2)
        self.assertEqual(engine["only_team_rocket"], 1)

        encoder = MODEL.OfficialPrototypeEncoder(
            MODEL.ModelConfig(d_model=32, heads=4, state_layers=1, option_layers=1),
            self.prototypes,
        )
        mask_start = 5 + len(PROTOTYPE_ENCODER.CARD_BOOLEANS)
        self.assertEqual(
            encoder.card_cat_table[15, mask_start:mask_start + 9].tolist(),
            [int(bool(80 & (1 << bit))) + 1 for bit in range(9)],
        )
        self.assertEqual(float(encoder.card_num_table[15, 2]), 2.0)

    def test_attach_option_binds_source_energy_and_target_pokemon(self) -> None:
        row, snapshot = _row(energy_id=None, energy_units=[])
        own = row["actor_observation"]["current"]["players"][0]
        own["hand"] = [{"id": 15, "playerIndex": 0, "serial": 44}]
        own["handCount"] = 1
        row["actor_observation"]["select"]["option"] = [
            {
                "type": 8,
                "area": 2,
                "index": 0,
                "inPlayArea": 4,
                "inPlayIndex": 0,
            }
        ]
        actor = COMPILER.compile_canonical_row(row, snapshot, self.prototypes)["actor"]
        source = actor["option_source"][0] - 1
        target = actor["option_target"][0] - 1
        self.assertEqual(actor["card_cat"][source][0], 15)
        self.assertEqual(actor["card_cat"][source][2], COMPILER.ZONE["hand"])
        self.assertEqual(actor["card_cat"][target][0], 96)
        self.assertEqual(actor["card_cat"][target][2], COMPILER.ZONE["active"])

    def test_duplicate_attached_cards_keep_their_exact_parent(self) -> None:
        row, snapshot = _row(energy_id=None, energy_units=[])
        own = row["actor_observation"]["current"]["players"][0]
        own["active"] = [
            _pokemon(96, player=0, serial=10, hp=210, energy_id=15, energy_units=[11])
        ]
        own["bench"] = [
            _pokemon(96, player=0, serial=11, hp=210, energy_id=15, energy_units=[11])
        ]
        compiled = COMPILER.compile_canonical_row(row, snapshot, self.prototypes)
        actor = compiled["actor"]
        energy_indices = [
            index
            for index, values in enumerate(actor["card_cat"])
            if values[2] == COMPILER.ZONE["energy"]
        ]
        self.assertEqual(len(energy_indices), 2)
        self.assertNotEqual(
            actor["card_parent"][energy_indices[0]],
            actor["card_parent"][energy_indices[1]],
        )
        FEATURE_AUDIT.audit_compiled_feature_input(row, compiled, snapshot)

        actor["card_parent"][energy_indices[1]] = actor["card_parent"][energy_indices[0]]
        with self.assertRaisesRegex(ValueError, "exact observation parent"):
            FEATURE_AUDIT.audit_compiled_feature_input(row, compiled, snapshot)

    def test_duplicate_attach_candidates_are_audited_by_exact_slot(self) -> None:
        row, snapshot = _row(energy_id=None, energy_units=[])
        own = row["actor_observation"]["current"]["players"][0]
        own["hand"] = [
            {"id": 15, "playerIndex": 0, "serial": 44},
            {"id": 15, "playerIndex": 0, "serial": 45},
        ]
        own["handCount"] = 2
        own["bench"] = [
            _pokemon(63, player=0, serial=30, hp=240),
            _pokemon(63, player=0, serial=31, hp=240),
        ]
        row["actor_observation"]["select"]["option"] = [
            {
                "type": 8,
                "area": 2,
                "index": 1,
                "inPlayArea": 5,
                "inPlayIndex": 1,
            }
        ]
        compiled = COMPILER.compile_canonical_row(row, snapshot, self.prototypes)
        actor = compiled["actor"]
        source = actor["option_source"][0] - 1
        target = actor["option_target"][0] - 1
        self.assertEqual(actor["card_cat"][source][3], 2)
        self.assertEqual(actor["card_cat"][target][3], 2)
        FEATURE_AUDIT.audit_compiled_feature_input(row, compiled, snapshot)

        first_hand = next(
            index + 1
            for index, values in enumerate(actor["card_cat"])
            if values[0] == 15
            and values[2] == COMPILER.ZONE["hand"]
            and values[3] == 1
        )
        actor["option_source"][0] = first_hand
        with self.assertRaisesRegex(ValueError, "option_source differs"):
            FEATURE_AUDIT.audit_compiled_feature_input(row, compiled, snapshot)

        compiled = COMPILER.compile_canonical_row(row, snapshot, self.prototypes)
        actor = compiled["actor"]
        first_bench = next(
            index + 1
            for index, values in enumerate(actor["card_cat"])
            if values[0] == 63
            and values[1] == 1
            and values[2] == COMPILER.ZONE["bench"]
            and values[3] == 1
        )
        actor["option_target"][0] = first_bench
        with self.assertRaisesRegex(ValueError, "option_target differs"):
            FEATURE_AUDIT.audit_compiled_feature_input(row, compiled, snapshot)

    def test_options_encode_only_real_owners_and_bind_active_actions(self) -> None:
        row, snapshot = _row()
        actor = COMPILER.compile_canonical_row(row, snapshot, self.prototypes)["actor"]
        attack, end = actor["option_cat"]

        self.assertEqual(attack[1:5], [1, 5, 2, 5])
        self.assertEqual(actor["card_cat"][actor["option_source"][0] - 1][0], 96)
        self.assertEqual(actor["card_cat"][actor["option_target"][0] - 1][0], 63)
        self.assertEqual(end[1:5], [0, 0, 0, 0])
        self.assertEqual(actor["option_source"][1], 0)
        self.assertEqual(actor["option_target"][1], 0)

    def test_card_attack_skill_area_and_trigger_subject_enter_tables(self) -> None:
        config = MODEL.ModelConfig(d_model=32, heads=4, state_layers=1, option_layers=1)
        encoder = MODEL.OfficialPrototypeEncoder(config, self.prototypes)
        card_id, card = next(
            (identity, value)
            for identity, value in self.prototypes.engine_cards.items()
            if value["attack_ids"]
        )
        expected_attacks = list(card["attack_ids"])
        self.assertEqual(
            encoder.card_attack_table[card_id, : len(expected_attacks)].tolist(),
            expected_attacks,
        )

        skill_id, skill = next(
            (identity, value)
            for identity, value in self.prototypes.skills.items()
            if value["triggers"]
            and value["triggers"][0]["subject"]["areas"]
            and value["triggers"][0]["subject"]["conditions"]
        )
        row = encoder.skill_cat_table[skill_id]
        area_set = set(skill["areas"])
        self.assertEqual(
            row[1:1 + PROTOTYPE_ENCODER.AREA_TYPE_COUNT].tolist(),
            [int(area in area_set) + 1 for area in range(PROTOTYPE_ENCODER.AREA_TYPE_COUNT)],
        )
        trigger = skill["triggers"][0]
        subject = trigger["subject"]
        trigger_start = 1 + PROTOTYPE_ENCODER.AREA_TYPE_COUNT + len(
            PROTOTYPE_ENCODER.SKILL_BOOLEANS
        )
        self.assertEqual(int(row[trigger_start]), int(trigger["trigger_type"]) + 1)
        self.assertEqual(int(row[trigger_start + 1]), int(subject["target_player"]) + 1)
        self.assertEqual(
            row[trigger_start + 4:trigger_start + 6].tolist(),
            [area + 1 for area in subject["areas"]] + [0] * (2 - len(subject["areas"])),
        )
        self.assertEqual(
            encoder.skill_num_state_table[skill_id, :2].tolist(),
            [int(PROTOTYPES.FieldState.PRESENT)] * 2,
        )

    def test_every_approved_prototype_column_is_live_in_its_encoder(self) -> None:
        torch.manual_seed(32)
        encoder = MODEL.OfficialPrototypeEncoder(
            MODEL.ModelConfig(d_model=32, heads=4, state_layers=1, option_layers=1),
            self.prototypes,
        ).eval()

        def assert_categorical_columns(
            table_name: str,
            vocabularies: tuple[int, ...],
            identity: int,
            encode,
        ) -> None:
            table = getattr(encoder, table_name)
            ids = torch.tensor([identity])
            for column, vocabulary in enumerate(vocabularies):
                original = int(table[identity, column])
                replacement = 1 if original != 1 else min(2, vocabulary - 1)
                with torch.inference_mode():
                    baseline = encode(ids).clone()
                    table[identity, column] = replacement
                    changed = encode(ids).clone()
                    table[identity, column] = original
                self.assertFalse(
                    torch.allclose(baseline, changed),
                    msg=f"prototype field is inert: {table_name}[{column}]",
                )

        assert_categorical_columns("card_cat_table", encoder.card_cat_vocabs, 15, encoder.card)
        assert_categorical_columns(
            "attack_cat_table", encoder.attack_cat_vocabs, 1, encoder.attack
        )
        assert_categorical_columns("skill_cat_table", encoder.skill_cat_vocabs, 128, encoder.skill)

        for table_name, identity, encode in (
            ("card_num_table", 15, encoder.card),
            ("attack_num_table", 1, encoder.attack),
        ):
            table = getattr(encoder, table_name)
            ids = torch.tensor([identity])
            for column in range(table.shape[1]):
                with torch.inference_mode():
                    baseline = encode(ids).clone()
                    table[identity, column] += 1.0
                    changed = encode(ids).clone()
                    table[identity, column] -= 1.0
                self.assertFalse(
                    torch.allclose(baseline, changed),
                    msg=f"prototype field is inert: {table_name}[{column}]",
                )

        skill_id = 128
        skill_ids = torch.tensor([skill_id])
        for column in range(encoder.skill_num_table.shape[1]):
            original_value = float(encoder.skill_num_table[skill_id, column])
            original_state = int(encoder.skill_num_state_table[skill_id, column])
            with torch.inference_mode():
                encoder.skill_num_state_table[skill_id, column] = int(
                    PROTOTYPES.FieldState.PRESENT
                )
                baseline = encoder.skill(skill_ids).clone()
                encoder.skill_num_table[skill_id, column] = original_value + 1.0
                changed = encoder.skill(skill_ids).clone()
                encoder.skill_num_table[skill_id, column] = original_value
                encoder.skill_num_state_table[skill_id, column] = original_state
            self.assertFalse(
                torch.allclose(baseline, changed),
                msg=f"prototype field is inert: skill_num_table[{column}]",
            )

        card_id = 15
        card_ids = torch.tensor([card_id])
        for table_name in ("card_skill_table", "card_attack_table"):
            table = getattr(encoder, table_name)
            for column in range(table.shape[1]):
                original = int(table[card_id, column])
                replacement = 1 if original != 1 else 2
                with torch.inference_mode():
                    baseline = encoder.card(card_ids).clone()
                    table[card_id, column] = replacement
                    changed = encoder.card(card_ids).clone()
                    table[card_id, column] = original
                self.assertFalse(
                    torch.allclose(baseline, changed),
                    msg=f"prototype relation is inert: {table_name}[{column}]",
                )

    def test_fixed_prototype_slots_cover_the_complete_engine_asset(self) -> None:
        self.assertEqual(len(self.prototypes.engine_cards), 1267)
        self.assertEqual(len(self.prototypes.engine_attacks), 1556)
        self.assertEqual(len(self.prototypes.skills), 433)
        self.assertEqual(
            max(len(card["attack_ids"]) for card in self.prototypes.engine_cards.values()),
            PROTOTYPE_ENCODER.CARD_ATTACK_SLOTS,
        )
        self.assertEqual(
            max(len(attack["energies"]) for attack in self.prototypes.engine_attacks.values()),
            PROTOTYPE_ENCODER.ATTACK_ENERGY_SLOTS,
        )
        self.assertEqual(
            max(len(skill["triggers"]) for skill in self.prototypes.skills.values()),
            PROTOTYPE_ENCODER.TRIGGER_SLOTS,
        )
        self.assertEqual(
            max(
                len(trigger["subject"]["areas"])
                for skill in self.prototypes.skills.values()
                for trigger in skill["triggers"]
            ),
            PROTOTYPE_ENCODER.TRIGGER_AREA_SLOTS,
        )
        self.assertEqual(
            max(
                len(trigger["subject"]["conditions"])
                for skill in self.prototypes.skills.values()
                for trigger in skill["triggers"]
            ),
            PROTOTYPE_ENCODER.TRIGGER_CONDITION_SLOTS,
        )

    def test_event_columns_and_instance_relations_are_not_shifted(self) -> None:
        row, _ = _row()
        event = STATE.TypedEvent(
            source_event=5,
            decision_index=0,
            local_log_ordinal=0,
            log_type=15,
            name="attack",
            actor=0,
            card_id=96,
            from_area=None,
            to_area=None,
            payload=MappingProxyType(
                {
                    "cardId": 96,
                    "cardIdTarget": 63,
                    "attackId": 120,
                    "serial": 10,
                    "serialTarget": 20,
                    "cardIdActive": 96,
                    "cardIdBench": 63,
                    "cardIdBefore": 96,
                    "cardIdAfter": 63,
                    "result": 2,
                    "reason": 3,
                    "value": 40,
                    "putDamageCounter": False,
                    "head": True,
                }
            ),
        )
        actor = COMPILER.compile_canonical_row(
            row,
            _snapshot(events=(event,)),
            self.prototypes,
        )["actor"]
        event_cat = actor["event_cat"][0]
        self.assertEqual(event_cat[:5], [16, 1, 96, 63, 120])
        self.assertEqual(event_cat[7:11], [96, 63, 96, 63])
        self.assertEqual(event_cat[11:], [0, 1, 2, 0, 3, 4])
        self.assertEqual(actor["event_num"][0], [0.0, 40.0, 0.0, 0.0])
        source = actor["event_source"][0] - 1
        target = actor["event_target"][0] - 1
        self.assertEqual(actor["card_cat"][source][0], 96)
        self.assertEqual(actor["card_cat"][target][0], 63)

    def test_switch_event_and_skill_option_bind_official_serial_fields(self) -> None:
        row, _ = _row()
        row["actor_observation"]["select"]["option"] = [
            {"type": 15, "cardId": 96, "serial": 10}
        ]
        event = STATE.TypedEvent(
            source_event=5,
            decision_index=0,
            local_log_ordinal=0,
            log_type=8,
            name="switch",
            actor=0,
            card_id=None,
            from_area=None,
            to_area=None,
            payload=MappingProxyType(
                {
                    "cardIdActive": 96,
                    "cardIdBench": 63,
                    "serialActive": 10,
                    "serialBench": 20,
                    "playerIndex": 0,
                    "type": 8,
                }
            ),
        )
        actor = COMPILER.compile_canonical_row(
            row,
            _snapshot(events=(event,)),
            self.prototypes,
        )["actor"]
        self.assertEqual(actor["event_cat"][0][7:9], [96, 63])
        self.assertEqual(actor["card_cat"][actor["event_source"][0] - 1][0], 96)
        self.assertEqual(actor["card_cat"][actor["event_target"][0] - 1][0], 63)
        self.assertEqual(actor["card_cat"][actor["option_source"][0] - 1][0], 96)

    def test_no_redundant_or_counterfactual_actor_fields(self) -> None:
        names: set[str] = set()
        for family in (
            FIELDS.GLOBAL_CAT_FIELDS,
            FIELDS.GLOBAL_NUM_FIELDS,
            FIELDS.CARD_CAT_FIELDS,
            FIELDS.CARD_NUM_FIELDS,
            FIELDS.RESOURCE_CAT_FIELDS,
            FIELDS.RESOURCE_NUM_FIELDS,
            FIELDS.EVENT_CAT_FIELDS,
            FIELDS.EVENT_NUM_FIELDS,
            FIELDS.OPTION_CAT_FIELDS,
            FIELDS.OPTION_NUM_FIELDS,
        ):
            names.update(family)
        forbidden = {
            "physical_energy_card_count",
            "resolved_energy_units",
            "resolved_energy_type_mask",
            "typed_deficit_before",
            "typed_deficit_after",
            "total_deficit_before",
            "total_deficit_after",
            "newly_enabled_attack_count",
            "attach_provides_type_mask",
            "attach_provides_units",
            "base_damage_is_ko",
            "target_hp_after_base_damage",
            "option_effect_id",
            "is_condition",
            "deck_membership_known",
            "deck_order_known",
            "own_hand_count",
            "known_opponent_hand_count",
            "unknown_opponent_hand_count",
            "identity_visible",
            "option_skill_id",
            "option_skill_role",
            "option_skill_parent",
        }
        self.assertTrue(forbidden.isdisjoint(names))
        self.assertNotIn("option_effect_id", FIELDS.ACTOR_KEYS)
        self.assertEqual(FIELDS.OPTION_NUM_FIELDS, ("number", "count"))

    def test_bench_max_stadium_owner_and_known_opponent_hand_are_visible(self) -> None:
        known = STATE.KnownOpponentCard(card_id=321, serial=88, source_event=4)
        row, _ = _row()
        actor = COMPILER.compile_canonical_row(
            row,
            _snapshot(known_hand=(known,)),
            self.prototypes,
        )["actor"]
        self.assertEqual(actor["global_num"][7:9], [5.0, 5.0])
        stadium = next(cat for cat in actor["card_cat"] if cat[0] == 1259)
        remembered = next(cat for cat in actor["card_cat"] if cat[0] == 321)
        self.assertEqual(stadium[1], 2)
        self.assertEqual(remembered[2], COMPILER.ZONE["known_opponent_hand"])

    def test_public_play_removes_revealed_card_from_known_opponent_hand(self) -> None:
        row, _ = _row()
        observation = row["actor_observation"]
        opponent = observation["current"]["players"][1]
        opponent["handCount"] = 1
        observation["logs"] = [
            {
                "type": STATE.MOVE_CARD,
                "playerIndex": 1,
                "cardId": 321,
                "serial": 88,
                "fromArea": 3,
                "toArea": STATE.HAND_AREA,
            }
        ]
        knowledge = STATE.CausalKnowledge(0, [1] * 60)
        first = knowledge.consume(observation)
        self.assertEqual(
            [(card.card_id, card.serial) for card in first.known_opponent_hand],
            [(321, 88)],
        )

        opponent["handCount"] = 0
        opponent["bench"] = [_pokemon(321, player=1, serial=88, hp=100)]
        observation["logs"] = [
            {"type": 10, "playerIndex": 1, "cardId": 321, "serial": 88}
        ]
        second = knowledge.consume(observation)
        self.assertEqual(second.known_opponent_hand, ())
        self.assertEqual(second.unknown_opponent_hand, 0)


if __name__ == "__main__":
    unittest.main()
