from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from torch import nn


PROJECT = "train.0042_full_model_design"
ROOT = Path(__file__).resolve().parents[3]


class CandidateDeploymentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.deployment = importlib.import_module(f"{PROJECT}.candidate_deployment")

    @staticmethod
    def payload(dtype: torch.dtype = torch.float16) -> dict[str, object]:
        state = {"weight": torch.ones((2, 2), dtype=dtype)}
        return {
            "schema_version": "0042_strategy_conditioned_kaggle_candidate_v1",
            "actor_state_dict": state,
            "value_head_state_dict": state,
            "allocation_head_state_dict": state,
            "value_adapter_state_dict": {**state, "gate": torch.zeros((), dtype=dtype)},
            "policy_strategy_adapter_state_dict": {
                **state, "gate": torch.zeros((), dtype=dtype)
            },
            "metadata": {"checkpoint_update": 5},
        }

    @staticmethod
    def manifest() -> dict[str, object]:
        return {
            "storage_dtype": "fp16",
            "runtime_dtype": "fp32",
            "rl_checkpoint_sha256": "a" * 64,
            "portable_checkpoint_sha256": "b" * 64,
            "checkpoint_update": 5,
            "action_boundary_deployment": {"decision_gate": True},
        }

    def test_fp32_storage_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            self.deployment.CandidateDeploymentIdentityViolation,
            "FATAL.*FP16 storage",
        ):
            self.deployment.audit_candidate_deployment(
                manifest=self.manifest(),
                payload=self.payload(torch.float32),
                runtime_model=nn.Linear(2, 2, dtype=torch.float32),
                source_checkpoint_sha256="a" * 64,
                portable_checkpoint_sha256="b" * 64,
            )

    def test_non_fp32_runtime_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            self.deployment.CandidateDeploymentIdentityViolation,
            "FATAL.*FP32 runtime",
        ):
            self.deployment.audit_candidate_deployment(
                manifest=self.manifest(),
                payload=self.payload(),
                runtime_model=nn.Linear(2, 2, dtype=torch.float16),
                source_checkpoint_sha256="a" * 64,
                portable_checkpoint_sha256="b" * 64,
            )

    def test_source_checkpoint_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            self.deployment.CandidateDeploymentIdentityViolation,
            "FATAL.*source checkpoint",
        ):
            self.deployment.audit_candidate_deployment(
                manifest=self.manifest(),
                payload=self.payload(),
                runtime_model=nn.Linear(2, 2, dtype=torch.float32),
                source_checkpoint_sha256="c" * 64,
                portable_checkpoint_sha256="b" * 64,
            )

    def test_passing_audit_records_deployment_identity(self) -> None:
        audit = self.deployment.audit_candidate_deployment(
            manifest=self.manifest(),
            payload=self.payload(),
            runtime_model=nn.Linear(2, 2, dtype=torch.float32),
            source_checkpoint_sha256="a" * 64,
            portable_checkpoint_sha256="b" * 64,
        )
        self.assertEqual(audit.status, "PASS")
        self.assertEqual(audit.contract_id, "kaggle_fp16_storage_fp32_runtime_v1")
        self.assertEqual(audit.storage_dtype, "fp16")
        self.assertEqual(audit.runtime_dtype, "fp32")
        self.assertEqual(len(audit.effective_candidate_sha256), 64)

    def test_custom_focal_deck_manifest_mismatch_is_rejected(self) -> None:
        payload = self.payload()
        payload["metadata"] = {
            **payload["metadata"], "focal_exact_deck_sha256": "c" * 64,
        }
        manifest = {**self.manifest(), "deck_sha256": "d" * 64}
        with self.assertRaisesRegex(
            self.deployment.CandidateDeploymentIdentityViolation,
            "FATAL.*custom focal exact-deck",
        ):
            self.deployment.audit_candidate_deployment(
                manifest=manifest,
                payload=payload,
                runtime_model=nn.Linear(2, 2, dtype=torch.float32),
                source_checkpoint_sha256="a" * 64,
                portable_checkpoint_sha256="b" * 64,
            )

    def test_formal_gate_rejects_missing_audit(self) -> None:
        with self.assertRaisesRegex(
            self.deployment.CandidateDeploymentIdentityViolation,
            "FATAL.*candidate deployment identity",
        ):
            self.deployment.require_kaggle_candidate_deployment(nn.Linear(2, 2))

    def test_formal_gate_rejects_failed_or_wrong_contract(self) -> None:
        model = nn.Linear(2, 2)
        model._candidate_deployment_audit = SimpleNamespace(
            status="FAIL", contract_id="other"
        )
        with self.assertRaises(self.deployment.CandidateDeploymentIdentityViolation):
            self.deployment.require_kaggle_candidate_deployment(model)

    def test_real_update0_strict_exports_and_materializes(self) -> None:
        initialization = importlib.import_module(f"{PROJECT}.initialization")
        presets = importlib.import_module(f"{PROJECT}.integrated.presets")
        runner = importlib.import_module(f"{PROJECT}.training.run_full_semantic")
        storage = importlib.import_module(f"{PROJECT}.training.storage_full_semantic")
        temporary_parent = ROOT / ".tmp/evaluation/0042_candidate_test"
        temporary_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_parent) as directory:
            root = Path(directory)
            checkpoint = root / "V1_strategy_conditioned_base/checkpoint/update-000000.pt"
            model, identity = initialization.build_preset_from_common_update0(
                runner.focal_deck(), presets.preset("FULL_MODEL")
            )
            storage.save_model_only(model, checkpoint, update=0, metadata={
                "project": "0042_full_model_design",
                "version": "V1_strategy_conditioned_base",
                "source_checkpoint_sha256": identity.checkpoint_sha256,
            })
            materialized, audit = self.deployment.materialize_kaggle_evaluation_candidate(
                source=runner.CANDIDATE_ROOT,
                checkpoint=checkpoint,
                deck=runner.focal_deck(),
                device="cpu",
                temporary_root=root / "materialized",
            )
            self.assertEqual(audit.status, "PASS")
            self.assertTrue(hasattr(materialized, "value_adapter"))
            self.assertTrue(hasattr(materialized, "policy_strategy_adapter"))
            self.assertEqual(float(materialized.value_adapter.gate), 0.0)
            self.assertEqual(float(materialized.policy_strategy_adapter.gate), 0.0)
            frozen = importlib.import_module(f"{PROJECT}.evaluation.frozen_jobs")
            self.assertEqual(
                audit.effective_candidate_sha256,
                frozen.EXPECTED_007_U0_DEPLOYMENT_SHA256,
            )
            custom_deck = list(runner.focal_deck())
            custom_deck[-1] = 2
            custom_model, custom_audit = (
                self.deployment.materialize_kaggle_evaluation_candidate(
                    source=runner.CANDIDATE_ROOT,
                    checkpoint=checkpoint,
                    deck=custom_deck,
                    deck_id="test_custom_exact_60",
                    deck_display_name="Test custom exact 60",
                    deck_source="unit_test",
                    device="cpu",
                    temporary_root=root / "custom_materialized",
                )
            )
            self.assertEqual(custom_audit.status, "PASS")
            self.assertNotEqual(
                custom_audit.effective_candidate_sha256,
                audit.effective_candidate_sha256,
            )
            self.assertEqual(int(custom_model.default_own_archetype_id), 0)

    def test_canonical_protocol_and_root_agent_rule_bind_the_contract(self) -> None:
        protocol = (
            ROOT / "docs/rl/RL_PROMOTE_CHAMPION_FROZEN_POLICY_PROTOCOL_V1.md"
        ).read_text(encoding="utf-8")
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        for text in (protocol, agents):
            self.assertIn("kaggle_fp16_storage_fp32_runtime_v1", text)
            self.assertIn("FP16", text)
            self.assertIn("FP32", text)
        self.assertIn("every RL Frozen evaluation", protocol)
        self.assertIn("PPO behavior-policy rollout", protocol)


if __name__ == "__main__":
    unittest.main()
