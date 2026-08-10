from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile
import unittest

import torch


exporter = importlib.import_module(
    "train.0040_dragapult_0809_action_boundary_rl.export_full_semantic_candidate"
)


class ExportFullSemanticCandidateTest(unittest.TestCase):
    def test_portable_value_network_matches_training_state_contract(self) -> None:
        training_value = importlib.import_module(
            "train.0040_dragapult_0809_action_boundary_rl.policy.value_network"
        ).LatentQueryValueHead(320, queries=8, layers=2, heads=8, dropout=0.0)
        portable_value = importlib.import_module(
            "train.0040_dragapult_0809_action_boundary_rl.kaggle_runtime.value_network"
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

    def test_merge_qv_weight_changes_only_q_and_v_rows(self) -> None:
        width = 3
        base = torch.arange(27, dtype=torch.float32).reshape(9, 3)
        q_a = torch.tensor([[1.0, 2.0, 3.0]])
        q_b = torch.tensor([[1.0], [2.0], [3.0]])
        v_a = torch.tensor([[2.0, 1.0, 0.5]])
        v_b = torch.tensor([[3.0], [2.0], [1.0]])

        merged = exporter._merge_qv_weight(
            base,
            q_a=q_a,
            q_b=q_b,
            v_a=v_a,
            v_b=v_b,
            alpha=2.0,
            rank=1,
        )

        torch.testing.assert_close(
            merged[:width], base[:width] + (q_b @ q_a) * 2.0
        )
        self.assertTrue(torch.equal(merged[width : 2 * width], base[width : 2 * width]))
        torch.testing.assert_close(
            merged[2 * width :], base[2 * width :] + (v_b @ v_a) * 2.0
        )
        self.assertTrue(torch.equal(base, torch.arange(27, dtype=torch.float32).reshape(9, 3)))

    def test_merge_qv_weight_rejects_malformed_factors(self) -> None:
        base = torch.zeros(9, 3)
        good_a = torch.zeros(1, 3)
        good_b = torch.zeros(3, 1)
        with self.assertRaises(ValueError):
            exporter._merge_qv_weight(
                base,
                q_a=torch.zeros(2, 3),
                q_b=good_b,
                v_a=good_a,
                v_b=good_b,
                alpha=2.0,
                rank=1,
            )
        with self.assertRaises(ValueError):
            exporter._merge_qv_weight(
                torch.zeros(8, 3),
                q_a=good_a,
                q_b=good_b,
                v_a=good_a,
                v_b=good_b,
                alpha=2.0,
                rank=1,
            )

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
                })
            result.write_text(json.dumps({
                "schema_version": "0040_frozen_per_game_results_policy_identity_v4",
                "checkpoint_update": 5,
                "frozen_panel_version": "frozen_0806_seeded_agent_first_player_v3",
                "opponent_policy_id": "Policy-0806",
                "policy_identity_audit": {
                    "status": "PASS",
                    "requested_policy_id": "Policy-0806",
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
