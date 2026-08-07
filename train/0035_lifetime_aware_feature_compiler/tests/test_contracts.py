from __future__ import annotations

import importlib
import unittest

import torch


FIELDS = importlib.import_module(
    "train.0035_lifetime_aware_feature_compiler.contracts.fields"
)
BATCH = importlib.import_module(
    "train.0035_lifetime_aware_feature_compiler.contracts.batch"
)


def _valid_mapping() -> dict[str, torch.Tensor]:
    b, c, r, e, o, s, f, a = 2, 3, 4, 2, 5, 3, 4, 2
    widths = FIELDS.WIDTHS
    return {
        "global_cat": torch.zeros(b, widths.global_cat, dtype=torch.long),
        "global_num": torch.zeros(b, widths.global_num),
        "global_state": torch.ones(b, widths.global_state, dtype=torch.long),
        "card_cat": torch.zeros(b, c, widths.card_cat, dtype=torch.long),
        "card_num": torch.zeros(b, c, widths.card_num),
        "card_state": torch.ones(b, c, widths.card_state, dtype=torch.long),
        "card_parent": torch.zeros(b, c, dtype=torch.long),
        "card_mask": torch.ones(b, c, dtype=torch.bool),
        "resource_cat": torch.zeros(b, r, widths.resource_cat, dtype=torch.long),
        "resource_num": torch.zeros(b, r, widths.resource_num),
        "resource_state": torch.ones(b, r, widths.resource_state, dtype=torch.long),
        "resource_mask": torch.ones(b, r, dtype=torch.bool),
        "event_cat": torch.zeros(b, e, widths.event_cat, dtype=torch.long),
        "event_num": torch.zeros(b, e, widths.event_num),
        "event_state": torch.ones(b, e, widths.event_state, dtype=torch.long),
        "event_source": torch.zeros(b, e, dtype=torch.long),
        "event_target": torch.zeros(b, e, dtype=torch.long),
        "event_before": torch.zeros(b, e, dtype=torch.long),
        "event_after": torch.zeros(b, e, dtype=torch.long),
        "event_mask": torch.ones(b, e, dtype=torch.bool),
        "option_cat": torch.zeros(b, o, widths.option_cat, dtype=torch.long),
        "option_num": torch.zeros(b, o, widths.option_num),
        "option_state": torch.zeros(b, o, widths.option_state, dtype=torch.long),
        "option_source": torch.zeros(b, o, dtype=torch.long),
        "option_target": torch.zeros(b, o, dtype=torch.long),
        "option_context": torch.zeros(b, o, dtype=torch.long),
        "option_effect_card": torch.zeros(b, o, dtype=torch.long),
        "option_mask": torch.ones(b, o, dtype=torch.bool),
        "option_skill_id": torch.zeros(b, s, dtype=torch.long),
        "option_skill_role": torch.zeros(b, s, dtype=torch.long),
        "option_skill_parent": torch.zeros(b, s, dtype=torch.long),
        "option_skill_mask": torch.ones(b, s, dtype=torch.bool),
        "option_effect_id": torch.zeros(b, f, dtype=torch.long),
        "option_effect_role": torch.zeros(b, f, dtype=torch.long),
        "option_effect_parent": torch.zeros(b, f, dtype=torch.long),
        "option_effect_mask": torch.ones(b, f, dtype=torch.bool),
        "min_count": torch.ones(b, dtype=torch.long),
        "max_count": torch.full((b,), a, dtype=torch.long),
        "targets": torch.zeros(b, a, dtype=torch.long),
    }


class ContractTests(unittest.TestCase):
    def test_actor_keys_exclude_provenance_and_legacy(self) -> None:
        forbidden = {"source_id", "team_name", "expert_id", "persona", "legacy", "action"}
        self.assertTrue(forbidden.isdisjoint(FIELDS.ACTOR_KEYS))

    def test_batch_accepts_exact_named_contract(self) -> None:
        batch = BATCH.DecisionBatch.from_mapping(_valid_mapping())
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.option_count, 5)
        self.assertEqual(batch.global_cat.shape[-1], len(FIELDS.GLOBAL_CAT_FIELDS))

    def test_batch_rejects_missing_and_extra_tensors(self) -> None:
        missing = _valid_mapping()
        missing.pop("event_num")
        with self.assertRaisesRegex(ValueError, "missing=.*event_num"):
            BATCH.DecisionBatch.from_mapping(missing)

        extra = _valid_mapping()
        extra["source_id"] = torch.zeros(2, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "extra=.*source_id"):
            BATCH.DecisionBatch.from_mapping(extra)

    def test_batch_rejects_wrong_mask_dtype_and_field_width(self) -> None:
        wrong_mask = _valid_mapping()
        wrong_mask["option_mask"] = wrong_mask["option_mask"].float()
        with self.assertRaisesRegex(ValueError, "option_mask must have dtype torch.bool"):
            BATCH.DecisionBatch.from_mapping(wrong_mask)

        wrong_width = _valid_mapping()
        wrong_width["card_num"] = torch.zeros(2, 3, len(FIELDS.CARD_NUM_FIELDS) + 1)
        with self.assertRaisesRegex(ValueError, "card_num width"):
            BATCH.DecisionBatch.from_mapping(wrong_width)


if __name__ == "__main__":
    unittest.main()
