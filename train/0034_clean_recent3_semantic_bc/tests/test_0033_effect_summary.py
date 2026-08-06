from __future__ import annotations

import importlib
from pathlib import Path
import unittest

import torch


BASE = "train.0034_clean_recent3_semantic_bc"
PROTOTYPES = importlib.import_module(f"{BASE}.domain.prototypes")
MODEL = importlib.import_module(f"{BASE}.model")
PROTOTYPE_ENCODER = importlib.import_module(f"{BASE}.model.prototype_encoder")


class EffectSummaryPrototypeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prototypes = PROTOTYPES.PrototypeIndex.load(
            Path("train/0034_clean_recent3_semantic_bc/assets/official_public_prototypes_v1.json")
        )

    def _encoder(self) -> object:
        torch.manual_seed(33)
        return MODEL.OfficialPrototypeEncoder(
            MODEL.ModelConfig(d_model=32, heads=4, state_layers=1, option_layers=1),
            self.prototypes,
        ).eval()

    def test_effect_slots_cover_full_engine_asset(self) -> None:
        max_attack_effects = max(
            len(attack["pre_effects"]) + len(attack["post_effects"])
            for attack in self.prototypes.engine_attacks.values()
        )
        max_skill_effects = max(
            len(skill["effects"])
            for skill in self.prototypes.skills.values()
        )
        self.assertLessEqual(max_attack_effects, PROTOTYPE_ENCODER.EFFECT_SLOTS)
        self.assertLessEqual(max_skill_effects, PROTOTYPE_ENCODER.EFFECT_SLOTS)

    def test_attack_and_skill_effect_summary_fields_are_live(self) -> None:
        encoder = self._encoder()
        attack_id = next(
            identity
            for identity, attack in self.prototypes.engine_attacks.items()
            if attack["pre_effects"] or attack["post_effects"]
        )
        skill_id = next(
            identity
            for identity, skill in self.prototypes.skills.items()
            if skill["effects"]
        )

        def assert_cat_live(kind: str, identity: int, encode) -> None:
            table = getattr(encoder, f"{kind}_effect_cat_table")
            ids = torch.tensor([identity])
            original = int(table[identity, 0, 0])
            replacement = 1 if original != 1 else 2
            with torch.inference_mode():
                baseline = encode(ids).clone()
                table[identity, 0, 0] = replacement
                changed = encode(ids).clone()
                table[identity, 0, 0] = original
            self.assertFalse(torch.allclose(baseline, changed), f"{kind} effect_type is inert")

        assert_cat_live("attack", attack_id, encoder.attack)
        assert_cat_live("skill", skill_id, encoder.skill)

    def test_effect_numeric_values_are_live_when_present(self) -> None:
        encoder = self._encoder()
        skill_id, slot = next(
            (identity, index)
            for identity in self.prototypes.skills
            for index in range(PROTOTYPE_ENCODER.EFFECT_SLOTS)
            if int(encoder.skill_effect_state_table[identity, index, 3]) == int(PROTOTYPES.FieldState.PRESENT)
            and float(encoder.skill_effect_num_table[identity, index, 3]) != 0.0
        )
        ids = torch.tensor([skill_id])
        original = float(encoder.skill_effect_num_table[skill_id, slot, 3])
        with torch.inference_mode():
            baseline = encoder.skill(ids).clone()
            encoder.skill_effect_num_table[skill_id, slot, 3] = original + 1.0
            changed = encoder.skill(ids).clone()
            encoder.skill_effect_num_table[skill_id, slot, 3] = original
        self.assertFalse(torch.allclose(baseline, changed), "effect numeric value is inert")

    def test_shared_type_space_and_card_type_projection_are_live(self) -> None:
        encoder = self._encoder()
        energy_card_id = 1
        energy_ids = torch.tensor([energy_card_id])
        attack_id = next(
            identity
            for identity, attack in self.prototypes.engine_attacks.items()
            if attack["energies"]
        )
        attack_ids = torch.tensor([attack_id])

        with torch.inference_mode():
            card_baseline = encoder.card(energy_ids).clone()
            attack_baseline = encoder.attack(attack_ids).clone()
            original_grass = encoder.type_identity.weight[2].clone()
            original_colorless = encoder.type_identity.weight[1].clone()
            encoder.type_identity.weight[2].add_(0.25)
            card_changed = encoder.card(energy_ids).clone()
            encoder.type_identity.weight[2].copy_(original_grass)
            encoder.type_identity.weight[1].add_(0.25)
            attack_changed = encoder.attack(attack_ids).clone()
            encoder.type_identity.weight[1].copy_(original_colorless)
        self.assertFalse(torch.allclose(card_baseline, card_changed), "shared card type embedding is inert")
        self.assertFalse(torch.allclose(attack_baseline, attack_changed), "shared attack cost type embedding is inert")

        card_type = int(encoder.card_cat_table[energy_card_id, 0])
        with torch.inference_mode():
            baseline = encoder.card(energy_ids).clone()
            original_weight = encoder.card_type_projections[card_type].weight.clone()
            encoder.card_type_projections[card_type].weight[0, 0].add_(0.25)
            changed = encoder.card(energy_ids).clone()
            encoder.card_type_projections[card_type].weight.copy_(original_weight)
        self.assertFalse(torch.allclose(baseline, changed), "card-type projection is inert")


if __name__ == "__main__":
    unittest.main()
