from __future__ import annotations

import unittest
from pathlib import Path

import torch
from torch.distributions import Categorical

from .. import FOCAL_DECK_ID
from ..checkpoint import FORBIDDEN_FIELDS, _validate
from ..foundation import verify_foundation
from ..league import load_frozen_catalog
from ..policy import load_actor_critic
from ..policy.action_distribution import cached_features, evaluate_actions_encoded
from ..foundation.contracts.fields import WIDTHS
from ..training.run import build_jobs


class ProjectContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _, cls.identity = load_actor_critic("cpu")

    def test_foundation_and_trainable_contract(self):
        self.assertEqual(verify_foundation().checkpoint_sha256, "5e0a6eea42bf228a9bd977cf56fdd14ad360fbceffcf5c19e2d9fe1b39713d98")
        self.model.assert_trainable_contract()
        self.assertTrue(all(name.startswith(("actor.action_decoder.", "value_head.")) for name in self.model.trainable_parameter_names()))
        self.assertFalse(any(p.requires_grad for p in self.model.opponent_head.parameters()))

    def test_frozen_catalog_and_schedule(self):
        catalog = load_frozen_catalog()
        self.assertEqual(len(catalog), 51)
        self.assertEqual(len(next(row for row in catalog if row.deck_id == FOCAL_DECK_ID).deck), 60)
        jobs = build_jobs(count=102, source_policy_update=0, seed=30, greedy_eval=True)
        self.assertEqual(len({job.opponent_id for job in jobs}), 51)
        self.assertEqual(sum(job.focal_first for job in jobs), 51)

    def test_model_only_checkpoint_rejects_training_state(self):
        for forbidden in FORBIDDEN_FIELDS:
            with self.assertRaises(ValueError):
                _validate({forbidden: torch.tensor(0)})

    def test_cached_decoder_matches_full_representation_path(self):
        b, c, r, e, o, s, f = 2, 3, 4, 2, 5, 3, 4
        batch = {
            "global_cat": torch.zeros(b, WIDTHS.global_cat, dtype=torch.long),
            "global_num": torch.zeros(b, WIDTHS.global_num),
            "card_cat": torch.zeros(b, c, WIDTHS.card_cat, dtype=torch.long),
            "card_num": torch.zeros(b, c, WIDTHS.card_num),
            "card_parent": torch.zeros(b, c, dtype=torch.long),
            "card_mask": torch.ones(b, c, dtype=torch.bool),
            "resource_cat": torch.zeros(b, r, WIDTHS.resource_cat, dtype=torch.long),
            "resource_num": torch.zeros(b, r, WIDTHS.resource_num),
            "resource_mask": torch.ones(b, r, dtype=torch.bool),
            "event_cat": torch.zeros(b, e, WIDTHS.event_cat, dtype=torch.long),
            "event_num": torch.zeros(b, e, WIDTHS.event_num),
            "event_mask": torch.ones(b, e, dtype=torch.bool),
            "option_cat": torch.zeros(b, o, WIDTHS.option_cat, dtype=torch.long),
            "option_num": torch.zeros(b, o, WIDTHS.option_num),
            "option_state": torch.zeros(b, o, WIDTHS.option_state, dtype=torch.long),
            "option_source": torch.zeros(b, o, dtype=torch.long),
            "option_target": torch.zeros(b, o, dtype=torch.long),
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
            "max_count": torch.full((b,), 2, dtype=torch.long),
            "targets": torch.zeros(b, 2, dtype=torch.long),
        }
        self.model.eval()
        with torch.inference_mode():
            validated, state, options = self.model.actor.encode(batch)
            self.model.prepare_inference_cache()
            cached_batch, cached_summary, cached_options = self.model.encode(batch)
            decoder_state = self.model.actor.action_decoder.initialize(validated, state.summary)
            distribution = Categorical(logits=self.model.actor.action_decoder.logits(validated, options, decoder_state).float())
            direct_log_prob = distribution.log_prob(torch.zeros(b, dtype=torch.long))
            direct_entropy = distribution.entropy()
            direct_value = self.model.value_head(state.summary).squeeze(-1)
            result = evaluate_actions_encoded(
                self.model.head,
                cached_features(validated, state.summary, options),
                torch.zeros(b, 1, dtype=torch.long),
                torch.ones(b, dtype=torch.long),
                torch.zeros(b, dtype=torch.bool),
            )
        torch.testing.assert_close(result.log_prob, direct_log_prob)
        torch.testing.assert_close(result.entropy, direct_entropy)
        torch.testing.assert_close(result.value, direct_value)
        torch.testing.assert_close(cached_summary, state.summary)
        torch.testing.assert_close(cached_options, options)
        self.assertTrue(torch.equal(cached_batch.option_mask, validated.option_mask))

    def test_no_cross_numbered_runtime_imports(self):
        root = Path(__file__).resolve().parents[1]
        sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in root.rglob("*.py")
            if "tests" not in path.parts
        )
        self.assertNotIn("from train.0028_", sources)
        self.assertNotIn("from train.0026_", sources)
        self.assertNotIn("import train.0028_", sources)
        self.assertNotIn("import train.0026_", sources)


if __name__ == "__main__":
    unittest.main()
