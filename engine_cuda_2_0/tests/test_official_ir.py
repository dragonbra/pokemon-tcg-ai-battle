from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))

from ptcg_cuda_engine.official_ir import validate_official_ir  # noqa: E402


def minimal_payload() -> dict:
    target = {
        "player": 1,
        "not_me": 0,
        "skip_enemy_target": 0,
        "areas": [1],
        "conditions": [
            {"type": 1, "comparator": 0, "value": 10, "value2": 0, "name_id": 1}
        ],
    }
    effect = {
        "type": 2,
        "select_type": 0,
        "select_count": 1,
        "select_context": 3,
        "flags": 0,
        "loop_count": 1,
        "priority": 0,
        "values": [10, 0],
        "condition_type": 0,
        "comparator": 0,
        "fail_skip": 0,
        "skill_id": 0,
        "linked_skill_id": 0,
        "linked_attack_id": 0,
        "target": target,
    }
    return {
        "schema_version": 1,
        "counts": {"cards": 1, "skills": 1, "attacks": 1, "continuations": 1, "name_sets": 1},
        "flag_schema": {"card": ["x"], "skill": ["x"], "effect": ["x"]},
        "name_sets": [
            {
                "id": 1,
                "equal_cards": [1],
                "contains_cards": [1],
                "ability_cards": [1],
                "attack_cards": [1],
            }
        ],
        "cards": [
            {
                "id": 1,
                "card_type": 0,
                "pokemon_type": 1,
                "evolution_type": 1,
                "retreat_cost": 1,
                "hp": 100,
                "weakness": 1,
                "resistance": 0,
                "energy_type": 1,
                "energy_count": 0,
                "flags": 0,
                "number": 1,
                "name_id": 1,
                "evolves_from_name_id": 0,
                "evolves_from2_name_id": 0,
                "ability_id": 1,
                "play_id": 0,
                "delay_id": 0,
                "attack_ids": [1],
            }
        ],
        "skills": [
            {
                "id": 1,
                "card_id": 1,
                "skill_type": 0,
                "flags": 0,
                "priority": 0,
                "first_condition_count": 0,
                "second_effect_start": 0,
                "second_effect_start_enemy": 0,
                "trigger_start": 0,
                "name_id": 1,
                "areas": [1],
                "triggers": [{"type": 1, "subject": copy.deepcopy(target)}],
                "effects": [copy.deepcopy(effect)],
            }
        ],
        "attacks": [
            {
                "id": 1,
                "card_id": 1,
                "damage": 10,
                "flags": 0,
                "last_cancel_fail_attack": 0,
                "name_id": 1,
                "energies": [0],
                "pre_effects": [],
                "post_effects": [copy.deepcopy(effect)],
            }
        ],
        "continuations": [{"id": 0, "symbol": "Continue(State&)"}],
    }


class OfficialIRTest(unittest.TestCase):
    def test_minimal_ir_is_valid_and_deterministic(self) -> None:
        summary = validate_official_ir(minimal_payload())
        self.assertEqual(summary.cards, 1)
        self.assertEqual(summary.effects, 2)
        self.assertEqual(summary.triggers, 1)
        self.assertEqual(summary.target_conditions, 3)
        self.assertEqual(summary.effect_types_used, (2,))
        self.assertEqual(summary.target_types_used, (1,))
        self.assertEqual(summary.select_contexts_used, (3,))
        self.assertEqual(summary.unique_effect_signatures, 1)
        self.assertEqual(summary.canonical_sha256, validate_official_ir(minimal_payload()).canonical_sha256)

    def test_reference_integrity_fails_closed(self) -> None:
        payload = minimal_payload()
        payload["cards"][0]["ability_id"] = 999
        with self.assertRaisesRegex(ValueError, "references skill 999"):
            validate_official_ir(payload)

    def test_dense_continuation_contract_is_required(self) -> None:
        payload = minimal_payload()
        payload["continuations"] = [{"id": 1, "symbol": "Continue(State&)"}]
        with self.assertRaisesRegex(ValueError, "dense and zero-based"):
            validate_official_ir(payload)

    def test_unknown_fields_cannot_leak_into_private_ir(self) -> None:
        payload = minimal_payload()
        payload["cards"][0]["name_en"] = "private card name"
        with self.assertRaisesRegex(ValueError, "unexpected=\\['name_en'\\]"):
            validate_official_ir(payload)

    def test_name_set_references_must_resolve(self) -> None:
        payload = minimal_payload()
        payload["name_sets"][0]["contains_cards"] = [2]
        with self.assertRaisesRegex(ValueError, "references card 2"):
            validate_official_ir(payload)

    def test_special_target_name_set_cannot_be_missing(self) -> None:
        payload = minimal_payload()
        condition = payload["skills"][0]["effects"][0]["target"]["conditions"][0]
        condition.update({"type": 51, "name_id": 0})
        with self.assertRaisesRegex(ValueError, "TargetType 51 numeric name set"):
            validate_official_ir(payload)

    def test_special_target_name_set_requires_expected_family(self) -> None:
        payload = minimal_payload()
        condition = payload["skills"][0]["effects"][0]["target"]["conditions"][0]
        condition.update({"type": 52, "name_id": 1})
        payload["name_sets"][0]["equal_cards"] = []
        payload["name_sets"][0]["contains_cards"] = [1]
        with self.assertRaisesRegex(ValueError, "requires 2 contains_cards, actual=1"):
            validate_official_ir(payload)

    def test_special_target_name_set_requires_exact_member_count(self) -> None:
        payload = minimal_payload()
        condition = payload["skills"][0]["effects"][0]["target"]["conditions"][0]
        condition.update({"type": 53, "name_id": 1})
        payload["name_sets"][0]["equal_cards"] = [1]
        with self.assertRaisesRegex(ValueError, "requires 3 equal_cards, actual=1"):
            validate_official_ir(payload)


if __name__ == "__main__":
    unittest.main()
