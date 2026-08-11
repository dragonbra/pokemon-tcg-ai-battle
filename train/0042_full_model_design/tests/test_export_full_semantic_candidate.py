from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile
import unittest

import torch


exporter = importlib.import_module(
    "train.0042_full_model_design.export_full_semantic_candidate"
)


class ExportFullSemanticCandidateTest(unittest.TestCase):
    def test_kaggle_entrypoint_accepts_the_0042_export_schema(self) -> None:
        entrypoint = (
            exporter.ROOT
            / "train/0042_full_model_design/kaggle_runtime/main.py"
        ).read_text(encoding="utf-8")
        self.assertIn(exporter.OUTPUT_SCHEMA, entrypoint)
        self.assertNotIn("0038 package manifest schema mismatch", entrypoint)

    def test_portable_value_network_matches_training_state_contract(self) -> None:
        training_value = importlib.import_module(
            "train.0042_full_model_design.policy.value_network"
        ).LatentQueryValueHead(320, queries=8, layers=2, heads=8, dropout=0.0)
        portable_value = importlib.import_module(
            "train.0042_full_model_design.kaggle_runtime.value_network"
        ).LatentQueryValueHead(320, 8)
        training_state = training_value.state_dict()
        portable_state = portable_value.state_dict()
        self.assertEqual(set(training_state), set(portable_state))
        self.assertEqual(
            {name: tuple(value.shape) for name, value in training_state.items()},
            {name: tuple(value.shape) for name, value in portable_state.items()},
        )

    def test_export_uses_0038_cached_semantic_runtime(self) -> None:
        runtime = exporter.SEMANTIC_RUNTIME_ROOT / "model/policy.py"
        source = runtime.read_text(encoding="utf-8")
        self.assertIn("def prepare_prototype_cache", source)
        self.assertIn("def prototype_cache_stats", source)
        self.assertNotEqual(
            runtime.resolve(),
            Path(
                "evaluation/arena/candidates/"
                "0034_dragapult_third_large_model_zero_shot/strategy/model/policy.py"
            ).resolve(),
        )
        export_source = Path(exporter.__file__).read_text(encoding="utf-8")
        self.assertIn(
            '"prototype_embedding_cache_version": "frozen_model_owned_v1"',
            export_source,
        )
        self.assertIn('"prototype_embedding_cache_required": True', export_source)

    def test_export_uses_exact_frozen_007_deck(self) -> None:
        deck = [
            int(value)
            for value in exporter.FOCAL_DECK_PATH.read_text(encoding="utf-8").splitlines()
            if value.strip()
        ]
        self.assertEqual(len(deck), 60)
        self.assertEqual(exporter._deck_hash(deck), exporter.FOCAL_DECK_SHA256)
        stale = [
            int(value)
            for value in Path(
                "evaluation/arena/candidates/"
                "0034_dragapult_third_large_model_zero_shot/deck.csv"
            ).read_text(encoding="utf-8").splitlines()
            if value.strip()
        ]
        self.assertNotEqual(exporter._deck_hash(stale), exporter.FOCAL_DECK_SHA256)

    def test_effective_hash_supports_scalar_adapter_gates(self) -> None:
        state = {"weight": torch.ones(2, dtype=torch.float16)}
        payload = {
            "schema_version": exporter.OUTPUT_SCHEMA,
            "metadata": {"checkpoint_update": 0},
            **{field: dict(state) for field in exporter.PORTABLE_STATE_FIELDS},
        }
        payload["value_adapter_state_dict"]["gate"] = torch.zeros((), dtype=torch.float16)
        payload["policy_strategy_adapter_state_dict"]["gate"] = torch.zeros(
            (), dtype=torch.float16
        )
        digest = exporter.deployment_effective_sha256(payload, {
            "action_boundary_deployment": {},
            "storage_dtype": "fp16",
            "runtime_dtype": "fp32",
        })
        self.assertEqual(len(digest), 64)

    def test_effective_hash_excludes_checkpoint_provenance(self) -> None:
        state = {"weight": torch.ones(2, dtype=torch.float16)}
        payload = {
            "schema_version": exporter.OUTPUT_SCHEMA,
            "metadata": {
                "project_id": "0042_full_model_design",
                "version": "V1_a",
                "checkpoint_update": 0,
                "checkpoint_sha256": "a" * 64,
                "source_policy_update": 0,
                "actor_metadata": {
                    "epoch": 11,
                    "model_config": {"model": "SemanticPolicy", "width": 320},
                },
                "own_archetype_id": 3,
                "no_option_lora": True,
            },
            **{field: dict(state) for field in exporter.PORTABLE_STATE_FIELDS},
        }
        manifest = {
            "action_boundary_deployment": {"decision_gate": True},
            "storage_dtype": "fp16",
            "runtime_dtype": "fp32",
        }
        first = exporter.deployment_effective_sha256(payload, manifest)
        payload["metadata"] = {
            **payload["metadata"],
            "version": "V99_same_effective_policy",
            "checkpoint_update": 99,
            "checkpoint_sha256": "b" * 64,
            "source_policy_update": 98,
            "actor_metadata": {
                "epoch": 999,
                "model_config": {"model": "SemanticPolicy", "width": 320},
            },
        }
        second = exporter.deployment_effective_sha256(payload, manifest)
        self.assertEqual(first, second)

    def test_effective_hash_includes_runtime_semantics(self) -> None:
        state = {"weight": torch.ones(2, dtype=torch.float16)}
        payload = {
            "schema_version": exporter.OUTPUT_SCHEMA,
            "metadata": {
                "actor_metadata": {"model_config": {"width": 320}},
                "own_archetype_id": 3,
                "no_option_lora": True,
            },
            **{field: dict(state) for field in exporter.PORTABLE_STATE_FIELDS},
        }
        manifest = {
            "action_boundary_deployment": {"decision_gate": True},
            "storage_dtype": "fp16",
            "runtime_dtype": "fp32",
        }
        first = exporter.deployment_effective_sha256(payload, manifest)
        payload["metadata"] = {**payload["metadata"], "own_archetype_id": 4}
        second = exporter.deployment_effective_sha256(payload, manifest)
        self.assertNotEqual(first, second)

    def test_frozen_selection_requires_v4_candidate_deployment_evidence(self) -> None:
        (exporter.ROOT / ".tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=exporter.ROOT / ".tmp") as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint/update-000005.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")
            result = root / "artifact/frozen_results/core-update-000005.json"
            result.parent.mkdir(parents=True)
            entries = []
            for index in range(2048):
                chance = index == 17
                entries.append({
                    "seed": index,
                    "valid": True,
                    "error": None,
                    "outcome": 1 if index % 2 else -1,
                    "focal_first": index < 1024,
                    "fallback": int(chance),
                    "fallback_reason": (
                        "chance_boundary_before_allocation" if chance else None
                    ),
                    "chance_boundary": chance,
                    "semantic_fallback": False,
                    "opponent_exact_deck_sha256": "d" * 64,
                    "opponent_effective_policy_sha256": "e" * 64,
                })
            result.write_text(json.dumps({
                "schema_version": "0042_frozen_per_game_results_policy0809_v1",
                "checkpoint_update": 5,
                "frozen_panel_version": "0042_frozen_0809_seeded_agent_first_player_v1",
                "schedule_sha256": "a" * 64,
                "opponent_policy_id": "Policy-0809",
                "policy_identity_audit": {
                    "status": "PASS",
                    "requested_policy_id": "Policy-0809",
                    "effective_policy_sha256": "e" * 64,
                },
                "candidate_deployment_identity_audit": {
                    "status": "PASS",
                    "contract_id": "kaggle_fp16_storage_fp32_runtime_v1",
                    "storage_dtype": "fp16",
                    "runtime_dtype": "fp32",
                    "checkpoint_update": 5,
                    "portable_checkpoint_sha256": "b" * 64,
                    "effective_candidate_sha256": "c" * 64,
                    "source_checkpoint_sha256": exporter._sha256(checkpoint),
                },
                "entries": entries,
            }))
            selection = exporter._frozen_selection(checkpoint, 5)
        self.assertEqual(selection["chance_boundaries"], 1)
        self.assertEqual(selection["semantic_fallbacks"], 0)


if __name__ == "__main__":
    unittest.main()
