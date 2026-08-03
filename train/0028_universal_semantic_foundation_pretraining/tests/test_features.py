from __future__ import annotations

import importlib
from pathlib import Path
from types import MappingProxyType
import unittest


BASE = "train.0028_universal_semantic_foundation_pretraining"
PROTOTYPES = importlib.import_module(f"{BASE}.domain.prototypes")
COMPILER = importlib.import_module(f"{BASE}.features.compiler")
STATE = importlib.import_module(f"{BASE}.knowledge.state")


def _pokemon(card_id: int, *, hp: int, energy_id: int | None = None) -> dict:
    energy = [] if energy_id is None else [{"id": energy_id, "playerIndex": 0, "serial": 90}]
    return {
        "id": card_id,
        "playerIndex": 0,
        "serial": card_id,
        "hp": hp,
        "maxHp": hp,
        "appearThisTurn": False,
        "energyCards": energy,
        "energies": [],
        "tools": [],
        "preEvolution": [],
    }


def _row(energy_id: int, *, include_deck_energy_option: bool = False) -> tuple[dict, object]:
    own_active = _pokemon(96, hp=210, energy_id=energy_id)
    opponent_active = _pokemon(63, hp=240)
    opponent_active["playerIndex"] = 1
    players = []
    for active, hand in ((own_active, []), (opponent_active, [])):
        players.append(
            {
                "active": [active],
                "bench": [],
                "hand": hand,
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
    deck_view = [{"id": 1, "playerIndex": 0, "serial": 101}] if include_deck_energy_option else []
    options = (
        [{"type": 3, "playerIndex": 0, "area": 1, "index": 0}]
        if include_deck_energy_option
        else [{"type": 13, "attackId": 120}, {"type": 14}]
    )
    row = {
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
                "type": 1 if include_deck_energy_option else 0,
                "context": 0,
                "contextCard": None,
                "effect": None,
                "deck": deck_view,
                "option": options,
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
    snapshot = STATE.CausalSnapshot(
        decision_index=0,
        perspective_actor=0,
        self_ledger=MappingProxyType({}),
        known_opponent_hand=(),
        unknown_opponent_hand=0,
        recent_events=(),
        deck_membership_known=False,
        deck_order_known=False,
    )
    return row, snapshot


class CanonicalFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prototypes = PROTOTYPES.PrototypeIndex.load(
            Path(f"{BASE.replace('.', '/')}/assets/official_public_prototypes_v1.json")
        )

    def test_actor_has_no_legacy_and_binds_attack_skill_effect_relations(self) -> None:
        row, snapshot = _row(4)
        record = COMPILER.compile_canonical_row(row, snapshot, self.prototypes)
        actor = record["actor"]
        self.assertNotIn("legacy", actor)
        self.assertNotIn("action", actor)
        self.assertEqual(actor["option_cat"][0][7], 120)
        self.assertIn(
            self.prototypes.engine_cards[96]["ability_skill_id"],
            actor["option_skill_id"],
        )
        self.assertTrue(actor["option_effect_id"])
        self.assertTrue(all(parent == 1 for parent in actor["option_effect_parent"]))

    def test_typed_energy_gap_distinguishes_grass_and_lightning(self) -> None:
        lightning_row, snapshot = _row(4)
        grass_row, _ = _row(1)
        lightning = COMPILER.compile_canonical_row(lightning_row, snapshot, self.prototypes)
        grass = COMPILER.compile_canonical_row(grass_row, snapshot, self.prototypes)
        self.assertEqual(lightning["actor"]["option_num"][0][8], 0.3)
        self.assertEqual(grass["actor"]["option_num"][0][8], 0.2)

    def test_selected_energy_option_has_explicit_energy_type(self) -> None:
        row, snapshot = _row(4, include_deck_energy_option=True)
        record = COMPILER.compile_canonical_row(row, snapshot, self.prototypes)
        self.assertEqual(record["actor"]["option_cat"][0][8], 2)


if __name__ == "__main__":
    unittest.main()
