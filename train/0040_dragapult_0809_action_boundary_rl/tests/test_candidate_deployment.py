from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn


PROJECT = "train.0040_dragapult_0809_action_boundary_rl"
ROOT = Path(__file__).resolve().parents[3]


class CandidateDeploymentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.deployment = importlib.import_module(f"{PROJECT}.candidate_deployment")

    @staticmethod
    def payload(dtype: torch.dtype = torch.float16) -> dict[str, object]:
        state = {"weight": torch.ones((2, 2), dtype=dtype)}
        return {
            "schema_version": "0038_compound_kaggle_candidate_v4",
            "actor_state_dict": state,
            "value_head_state_dict": state,
            "allocation_head_state_dict": state,
            "opponent_meta_head_state_dict": state,
            "opponent_meta_conditioner_state_dict": state,
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
