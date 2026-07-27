from __future__ import annotations

from importlib import import_module
import json
from pathlib import Path
import tempfile
import unittest

import torch


R2 = import_module("train.0016_alakazam_multideck_bc.r2_model")
R15 = import_module("train.0016_alakazam_multideck_bc.r15_model")
SOURCE = import_module("train.0016_alakazam_multideck_bc.source_model")
SOURCE_R15 = import_module("train.0016_alakazam_multideck_bc.source_r15_model")
INFERENCE = import_module("train.0016_alakazam_multideck_bc.source_inference")
SEMANTICS = import_module("train.0016_alakazam_multideck_bc.card_semantics")
MATERIALIZED = import_module("train.0016_alakazam_multideck_bc.training.materialized")


class ModelContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.ontology = Path(cls.temporary.name) / "card_ontology.json"
        registry = SEMANTICS.CardSemanticRegistry.from_official_csv(
            "data/official/EN_Card_Data.csv"
        )
        cls.ontology.write_text(
            json.dumps(MATERIALIZED._ontology_payload(registry)),
            encoding="utf-8",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_v12_r2_backbone_contract_is_frozen(self) -> None:
        config = R2.R2ModelConfig()
        model = R2.R2StrongScenarioPolicy(config, ontology_path=self.ontology)
        self.assertEqual(config.ac.base.d_model, 320)
        self.assertEqual(config.ac.base.heads, 8)
        self.assertEqual(config.ac.base.encoder_layers, 4)
        self.assertEqual(config.ac.base.max_action_steps, 64)
        self.assertEqual(config.scenario_layers, 2)
        self.assertEqual(R2.parameter_count(model), 17_387_842)
        scenario = torch.zeros(2, 4 * config.ac.base.d_model)
        scale = model.state_scale_gate(scenario)
        self.assertTrue(torch.equal(scale, torch.ones_like(scale)))

    def test_source_residual_is_small_and_neutral_at_id_zero(self) -> None:
        model = SOURCE.SourceConditionedR2Policy(
            R2.R2ModelConfig(),
            SOURCE.SourceModelConfig(),
            ontology_path=self.ontology,
        )

    def test_r15_changes_only_the_option_gate_initial_scale(self) -> None:
        config = R15.R15ModelConfig()
        model = SOURCE_R15.SourceConditionedR15Policy(
            config,
            SOURCE.SourceModelConfig(),
            ontology_path=self.ontology,
        )
        width = config.ac.base.d_model
        scenario = torch.zeros(2, 4 * width)
        option_scenario = torch.zeros(2, 5, 2 * width)
        self.assertEqual(config.ac.base.max_action_steps, 64)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 17_601_282)
        self.assertTrue(
            torch.allclose(model.state_scale_gate(scenario), torch.ones(2, width))
        )
        self.assertTrue(
            torch.allclose(
                model.option_scale_gate(option_scenario),
                torch.full((2, 5, width), 0.35),
            )
        )
        self.assertTrue(
            torch.allclose(torch.tanh(model.source_gate), torch.full((width,), 0.05))
        )
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 17_601_282)
        self.assertEqual(
            sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if name.startswith("source_")
            ),
            213_440,
        )
        self.assertTrue(torch.equal(model.source_persona.weight[0], torch.zeros(320)))
        self.assertTrue(
            torch.allclose(
                torch.tanh(model.source_gate),
                torch.full((320,), 0.05),
            )
        )

    def test_action_batch_mask_keeps_short_rows_sparse_and_supports_17_steps(self) -> None:
        tensors = {
            "legacy_global_cat": torch.zeros((2, 4), dtype=torch.int16),
            "legacy_global_num": torch.zeros((2, 12), dtype=torch.float32),
            "legacy_entity_cat": torch.zeros((2, 1, 7), dtype=torch.int16),
            "legacy_entity_num": torch.zeros((2, 1, 5), dtype=torch.float32),
            "legacy_entity_mask": torch.ones((2, 1), dtype=torch.bool),
            "legacy_option_cat": torch.zeros((2, 20, 12), dtype=torch.int16),
            "legacy_option_mask": torch.ones((2, 20), dtype=torch.bool),
            "ordered_action": torch.tensor(
                [[0, 1, *([0] * 15)], list(range(17))], dtype=torch.int16
            ),
            "action_mask": torch.tensor(
                [[True, True, *([False] * 15)], [True] * 17], dtype=torch.bool
            ),
            "min_count": torch.tensor([1, 17], dtype=torch.int16),
            "max_count": torch.tensor([2, 17], dtype=torch.int16),
            "a0_eligible": torch.ones(2, dtype=torch.bool),
        }
        batch = MATERIALIZED.select_a0_batch(tensors, torch.tensor([0, 1]))
        self.assertEqual(tuple(batch["targets"].shape), (2, 18))
        self.assertEqual(batch["targets"][0, :3].tolist(), [0, 1, 20])
        self.assertTrue(torch.equal(batch["targets"][0, 3:], torch.full((15,), -100)))
        self.assertEqual(batch["targets"][1, :17].tolist(), list(range(17)))
        self.assertEqual(int(batch["targets"][1, 17]), 20)

    def test_capacity_independent_fallback_obeys_engine_count_contract(self) -> None:
        observation = {
            "select": {
                "option": [{"type": 0} for _ in range(80)],
                "minCount": 65,
                "maxCount": 80,
            }
        }
        action = INFERENCE.legal_fallback(observation)
        self.assertEqual(action, list(range(65)))
        self.assertEqual(len(action), len(set(action)))


if __name__ == "__main__":
    unittest.main()
