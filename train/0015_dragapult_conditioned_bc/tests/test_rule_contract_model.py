from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from ..rule_contract_model import (
    SourceConditionedR15RuleConfig,
    SourceConditionedR15RulePolicy,
)
from ..export_candidate import export_candidate, select_checkpoint
from ..initialization import freeze_to_new_model_keys, load_r15_base_checkpoint
from ..run import build_policy
from ..training.data import iter_batches


DATASET = Path("rl_runs/0015_dragapult_conditioned_bc/dataset/V1_core_r15_features")


class RuleContractConfigTest(unittest.TestCase):
    def test_rejects_invalid_rule_hyperparameters(self) -> None:
        base = SourceConditionedR15RuleConfig(source_vocabulary_size=84)
        for invalid in (
            replace(base, rule_layers=0),
            replace(base, rule_ffn_multiplier=0),
            replace(base, rule_initial_scale=0.0),
            replace(base, rule_initial_scale=2.0),
            replace(base, rule_model_width=0),
            replace(base, rule_heads=0),
            replace(base, rule_model_width=80, rule_heads=3),
        ):
            with self.subTest(config=invalid):
                with self.assertRaises(ValueError):
                    invalid.validate()

    def test_default_rule_width_preserves_parameter_count(self) -> None:
        model = SourceConditionedR15RulePolicy(
            SourceConditionedR15RuleConfig(source_vocabulary_size=84),
            ontology_path=DATASET / "card_ontology.json",
        )
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 20_728_002)

    def test_low_rank_rule_width_preserves_output_contract(self) -> None:
        batch = next(
            iter_batches(
                DATASET,
                "validation",
                arm="t1",
                batch_size=2,
                shuffle=False,
                seed=20260723,
            )
        )
        config = SourceConditionedR15RuleConfig(
            source_vocabulary_size=84,
            rule_model_width=80,
            rule_heads=4,
        )
        model = SourceConditionedR15RulePolicy(
            config,
            ontology_path=DATASET / "card_ontology.json",
        ).eval()
        with torch.no_grad():
            logits = model.teacher_logits(batch)
            scale = model.rule_scale_gate(torch.randn(2, 3 * config.rule_model_width))
        self.assertEqual(logits.shape[:2], batch["targets"].shape)
        self.assertEqual(logits.shape[2], batch["option_mask"].shape[1] + 1)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.allclose(scale, torch.full_like(scale, 0.10), atol=1e-6))
        rule_parameters = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith(
                (
                    "rule_",
                    "turn_budget_reader.",
                    "prize_race_reader.",
                    "library_pressure_reader.",
                    "board_relay_reader.",
                    "option_rule_attention.",
                )
            )
        )
        self.assertLess(rule_parameters, 3_311_360)

    def test_rule_gate_starts_at_declared_low_scale(self) -> None:
        config = SourceConditionedR15RuleConfig(
            source_vocabulary_size=84,
            rule_initial_scale=0.10,
        )
        model = SourceConditionedR15RulePolicy(
            config,
            ontology_path=DATASET / "card_ontology.json",
        )
        width = model.config.d_model
        scale = model.rule_scale_gate(torch.randn(2, 3 * width))
        self.assertTrue(torch.allclose(scale, torch.full_like(scale, 0.10), atol=1e-6))

    def test_teacher_logits_are_finite_and_shape_compatible(self) -> None:
        batch = next(
            iter_batches(
                DATASET,
                "validation",
                arm="t1",
                batch_size=2,
                shuffle=False,
                seed=20260723,
            )
        )
        model = SourceConditionedR15RulePolicy(
            SourceConditionedR15RuleConfig(source_vocabulary_size=84),
            ontology_path=DATASET / "card_ontology.json",
        ).eval()
        with torch.no_grad():
            logits = model.teacher_logits(batch)
        self.assertEqual(logits.shape[:2], batch["targets"].shape)
        self.assertEqual(logits.shape[2], batch["option_mask"].shape[1] + 1)
        self.assertFalse(torch.isnan(logits).any())
        self.assertTrue(torch.isfinite(logits).any())

    def test_default_factory_preserves_frozen_r15(self) -> None:
        model, config = build_policy(
            model_family="r15",
            source_vocabulary_size=84,
            rule_initial_scale=0.10,
            rule_model_width=320,
            rule_heads=8,
            ontology_path=DATASET / "card_ontology.json",
        )
        self.assertEqual(type(model).__name__, "SourceConditionedR15Policy")
        self.assertEqual(type(config).__name__, "SourceConditionedR15Config")
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 17_416_642)

    def test_factory_builds_source_conditioned_v12_r2_contract(self) -> None:
        model, config = build_policy(
            model_family="r2",
            source_vocabulary_size=84,
            rule_initial_scale=0.10,
            rule_model_width=320,
            rule_heads=8,
            ontology_path=DATASET / "card_ontology.json",
        )
        self.assertEqual(type(model).__name__, "SourceConditionedR2Policy")
        self.assertEqual(type(config).__name__, "SourceConditionedR2Config")
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 17_416_642)
        width = model.config.d_model
        state_scale = model.state_scale_gate(torch.randn(2, 4 * width))
        option_scale = model.option_scale_gate(torch.randn(2, 3, 2 * width))
        self.assertTrue(torch.allclose(state_scale, torch.ones_like(state_scale)))
        self.assertTrue(torch.allclose(option_scale, torch.ones_like(option_scale)))
        self.assertTrue(
            torch.allclose(
                torch.tanh(model.source_gate),
                torch.full_like(model.source_gate, 0.10),
            )
        )

    def test_factory_threads_source_initial_scale(self) -> None:
        model, config = build_policy(
            model_family="r15",
            source_vocabulary_size=84,
            rule_initial_scale=0.10,
            rule_model_width=320,
            rule_heads=8,
            ontology_path=DATASET / "card_ontology.json",
            source_initial_scale=0.03,
        )
        self.assertEqual(config.source_initial_scale, 0.03)
        self.assertTrue(
            torch.allclose(
                torch.tanh(model.source_gate),
                torch.full_like(model.source_gate, 0.03),
            )
        )

    def test_export_accepts_source_conditioned_r2_checkpoint(self) -> None:
        model, config = build_policy(
            model_family="r2",
            source_vocabulary_size=84,
            rule_initial_scale=0.10,
            rule_model_width=320,
            rule_heads=8,
            ontology_path=DATASET / "card_ontology.json",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            version = root / "V18_r2_test"
            (version / "checkpoint").mkdir(parents=True)
            (version / "artifact").mkdir()
            checkpoint = version / "checkpoint/epoch-0001-test.pt"
            metadata = {
                "schema_version": "0015_r2_source_conditioned_training_v1",
                "model_family": "r2",
                "model": config.to_dict(),
                "no_deck": False,
                "metrics": {
                    "bc/validation/exact_action": 0.5,
                    "bc/validation/loss": 0.7,
                },
            }
            torch.save({"model": model.state_dict(), "metadata": metadata}, checkpoint)
            checkpoint.with_suffix(".json").write_text(
                json.dumps({"epoch": 1, "metadata": metadata}), encoding="utf-8"
            )
            (version / "artifact/model_contract.json").write_text(
                json.dumps({"target_source_id": 1}), encoding="utf-8"
            )
            deck = root / "deck.csv"
            deck.write_text("1\n" * 60, encoding="utf-8")
            cg = root / "cg"
            cg.mkdir()
            (cg / "__init__.py").write_text("", encoding="utf-8")
            output = root / "candidate"
            manifest = export_candidate(
                version_root=version,
                criterion="best_greedy_exact",
                ontology=DATASET / "card_ontology.json",
                deck=deck,
                cg_source=cg,
                output=output,
            )
            self.assertEqual(manifest["model_family"], "r2")
            self.assertEqual(
                manifest["schema_version"], "0015_source_conditioned_r2_candidate_v1"
            )
            self.assertIn("SourcePolicy", (output / "main.py").read_text(encoding="utf-8"))

    def test_v2_warm_start_loads_only_inherited_parameters(self) -> None:
        model = SourceConditionedR15RulePolicy(
            SourceConditionedR15RuleConfig(
                source_vocabulary_size=84,
                rule_initial_scale=0.03,
            ),
            ontology_path=DATASET / "card_ontology.json",
        )
        checkpoint = select_checkpoint(
            Path(
                "rl_runs/0015_dragapult_conditioned_bc/versions/"
                "V2_t1_plus_pure_dragapult"
            ),
            "best_greedy_exact",
        )
        before = {name: value.clone() for name, value in model.state_dict().items()}
        provenance = load_r15_base_checkpoint(model, checkpoint)
        source = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
        after = model.state_dict()
        self.assertEqual(provenance["loaded_key_count"], len(source))
        self.assertEqual(set(source), set(after) - set(provenance["new_model_keys"]))
        for name, value in source.items():
            self.assertTrue(torch.equal(after[name], value), name)
        for name in provenance["new_model_keys"]:
            self.assertTrue(torch.equal(after[name], before[name]), name)

    def test_rule_only_freeze_matches_audited_new_keys(self) -> None:
        model = SourceConditionedR15RulePolicy(
            SourceConditionedR15RuleConfig(source_vocabulary_size=84),
            ontology_path=DATASET / "card_ontology.json",
        )
        checkpoint = select_checkpoint(
            Path(
                "rl_runs/0015_dragapult_conditioned_bc/versions/"
                "V2_t1_plus_pure_dragapult"
            ),
            "best_greedy_exact",
        )
        provenance = load_r15_base_checkpoint(model, checkpoint)
        freeze = freeze_to_new_model_keys(model, provenance["new_model_keys"])
        new_keys = set(provenance["new_model_keys"])
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name in new_keys, name)
        self.assertEqual(
            freeze["trainable_parameter_count"] + freeze["frozen_parameter_count"],
            sum(parameter.numel() for parameter in model.parameters()),
        )



if __name__ == "__main__":
    unittest.main()
