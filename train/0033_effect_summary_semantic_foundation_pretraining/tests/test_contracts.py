from __future__ import annotations

import importlib
import unittest

import torch


BASE = "train.0033_effect_summary_semantic_foundation_pretraining"
FIELDS = importlib.import_module(f"{BASE}.contracts.fields")
BATCH = importlib.import_module(f"{BASE}.contracts.batch")


def _valid_mapping() -> dict[str, torch.Tensor]:
    b, c, r, e, o = 2, 3, 4, 2, 5
    widths = FIELDS.WIDTHS
    targets = torch.tensor([[0, o], [1, o]], dtype=torch.long)
    return {
        "global_cat": torch.ones(b, widths.global_cat, dtype=torch.long),
        "global_num": torch.zeros(b, widths.global_num),
        "global_state": torch.ones(b, widths.global_state, dtype=torch.long),
        "card_cat": torch.ones(b, c, widths.card_cat, dtype=torch.long),
        "card_num": torch.zeros(b, c, widths.card_num),
        "card_state": torch.ones(b, c, widths.card_state, dtype=torch.long),
        "card_parent": torch.zeros(b, c, dtype=torch.long),
        "card_mask": torch.ones(b, c, dtype=torch.bool),
        "resource_cat": torch.ones(b, r, widths.resource_cat, dtype=torch.long),
        "resource_num": torch.zeros(b, r, widths.resource_num),
        "resource_state": torch.ones(b, r, widths.resource_state, dtype=torch.long),
        "resource_mask": torch.ones(b, r, dtype=torch.bool),
        "event_cat": torch.ones(b, e, widths.event_cat, dtype=torch.long),
        "event_num": torch.zeros(b, e, widths.event_num),
        "event_state": torch.ones(b, e, widths.event_state, dtype=torch.long),
        "event_source": torch.zeros(b, e, dtype=torch.long),
        "event_target": torch.zeros(b, e, dtype=torch.long),
        "event_mask": torch.ones(b, e, dtype=torch.bool),
        "option_cat": torch.ones(b, o, widths.option_cat, dtype=torch.long),
        "option_num": torch.zeros(b, o, widths.option_num),
        "option_state": torch.ones(b, o, widths.option_state, dtype=torch.long),
        "option_source": torch.zeros(b, o, dtype=torch.long),
        "option_target": torch.zeros(b, o, dtype=torch.long),
        "option_mask": torch.ones(b, o, dtype=torch.bool),
        "min_count": torch.ones(b, dtype=torch.long),
        "max_count": torch.full((b,), 2, dtype=torch.long),
        "targets": targets,
    }


class ContractTests(unittest.TestCase):
    def test_actor_keys_exclude_provenance_effect_graph_and_answers(self) -> None:
        forbidden = {
            "source_id", "team_name", "expert_id", "legacy", "action",
            "option_effect_id", "option_effect_role", "option_effect_parent",
            "typed_deficit_before", "typed_deficit_after", "newly_enabled_attack_count",
        }
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

    def test_batch_rejects_wrong_mask_width_and_real_token_padding(self) -> None:
        wrong_mask = _valid_mapping()
        wrong_mask["option_mask"] = wrong_mask["option_mask"].float()
        with self.assertRaisesRegex(ValueError, "option_mask must have dtype torch.bool"):
            BATCH.DecisionBatch.from_mapping(wrong_mask)

        wrong_width = _valid_mapping()
        wrong_width["card_num"] = torch.zeros(2, 3, len(FIELDS.CARD_NUM_FIELDS) + 1)
        with self.assertRaisesRegex(ValueError, "card_num width"):
            BATCH.DecisionBatch.from_mapping(wrong_width)

        padded_identity = _valid_mapping()
        padded_identity["card_cat"][0, 0, 0] = 0
        with self.assertRaisesRegex(ValueError, "card_cat field 0 cannot be padding"):
            BATCH.DecisionBatch.from_mapping(padded_identity)

    def test_batch_rejects_invalid_states_relations_and_targets(self) -> None:
        invalid_state = _valid_mapping()
        invalid_state["option_state"][0, 0, 0] = 0
        with self.assertRaisesRegex(ValueError, "option_state has padding"):
            BATCH.DecisionBatch.from_mapping(invalid_state)

        invalid_relation = _valid_mapping()
        invalid_relation["option_target"][0, 0] = 4
        with self.assertRaisesRegex(ValueError, "option_target contains an invalid"):
            BATCH.DecisionBatch.from_mapping(invalid_relation)

        invalid_target = _valid_mapping()
        invalid_target["option_mask"][0, 3:] = False
        invalid_target["option_state"][0, 3:] = 0
        invalid_target["max_count"][0] = 2
        invalid_target["targets"][0, 0] = 4
        with self.assertRaisesRegex(ValueError, "illegal or padded option"):
            BATCH.DecisionBatch.from_mapping(invalid_target)


if __name__ == "__main__":
    unittest.main()
