from __future__ import annotations

import importlib
from pathlib import Path
from types import MappingProxyType
import unittest


BASE = "train.0031_rule_faithful_semantic_foundation_pretraining"
PROTOTYPES = importlib.import_module(f"{BASE}.domain.prototypes")
COMPILER = importlib.import_module(f"{BASE}.features.compiler")
FIELDS = importlib.import_module(f"{BASE}.contracts.fields")
STATE = importlib.import_module(f"{BASE}.knowledge.state")
MODEL = importlib.import_module(f"{BASE}.model")


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

    def test_team_rocket_energy_preserves_physical_and_engine_semantics(self) -> None:
        row, snapshot = _row()
        record = COMPILER.compile_canonical_row(row, snapshot, self.prototypes)
        actor = record["actor"]
        energy_index = next(i for i, cat in enumerate(actor["card_cat"]) if cat[0] == 15)
        self.assertEqual(actor["card_parent"][energy_index], 1)
        active_index = actor["card_parent"][energy_index] - 1
        self.assertEqual(actor["card_num"][active_index][6], float(1 << 11))
        self.assertEqual(actor["card_cat"][active_index][6 + 11], 2)
        self.assertEqual(actor["card_num"][active_index][5], 2.0)
        self.assertEqual(actor["card_state"][energy_index][5], int(PROTOTYPES.FieldState.UNKNOWN))
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
        self.assertEqual(actor["card_cat"][source][4], 3)

    def test_no_forbidden_derived_energy_or_damage_answers(self) -> None:
        forbidden = {
            "typed_energy_deficit", "total_energy_deficit", "surplus_energy",
            "newly_enabled", "base_damage_is_ko", "target_hp_after_base_damage",
            "selected_energy_type", "preferred_energy_removal",
        }
        self.assertTrue(forbidden.isdisjoint(FIELDS.OPTION_NUM_FIELDS))
        self.assertTrue(forbidden.isdisjoint(FIELDS.OPTION_CAT_FIELDS))

    def test_bench_max_stadium_owner_and_known_opponent_hand_are_visible(self) -> None:
        known = STATE.KnownOpponentCard(card_id=321, serial=88, source_event=4)
        row, _ = _row()
        actor = COMPILER.compile_canonical_row(row, _snapshot(known_hand=(known,)), self.prototypes)["actor"]
        self.assertEqual(actor["global_num"][17:19], [5.0, 5.0])
        stadium = next(cat for cat in actor["card_cat"] if cat[0] == 1259)
        remembered = next(cat for cat in actor["card_cat"] if cat[0] == 321)
        self.assertEqual(stadium[1], 2)
        self.assertEqual(remembered[2], COMPILER.ZONE["known_opponent_hand"])

    def test_target_area_identity_and_conditions_enter_prototype_tables(self) -> None:
        config = MODEL.ModelConfig(d_model=32, heads=4, state_layers=1, option_layers=1)
        encoder = MODEL.OfficialPrototypeEncoder(config, self.prototypes)
        effect_id, effect = next(
            (identity, value) for identity, value in self.prototypes.effects.items()
            if value["target"]["areas"] and value["target"]["conditions"]
        )
        # Effect prefix + boolean fields; target begins with player/not_me/skip, then exact areas.
        area_start = 9 + len(importlib.import_module(f"{BASE}.model.prototype_encoder").EFFECT_BOOLEANS) + 3
        encoded_areas = encoder.effect_cat_table[effect_id, area_start:area_start + 3].tolist()
        expected = [area + 1 for area in effect["target"]["areas"]] + [0] * (3 - len(effect["target"]["areas"]))
        self.assertEqual(encoded_areas, expected)
        condition_start = area_start + 3
        self.assertEqual(int(encoder.effect_cat_table[effect_id, condition_start]),
                         int(effect["target"]["conditions"][0]["target_type"]) + 1)


if __name__ == "__main__":
    unittest.main()
