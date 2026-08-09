from __future__ import annotations

import importlib
import unittest


PROJECT = "train.0038_action_boundary_rl"


class SemanticParitySubmissionManifestTest(unittest.TestCase):
    def test_failed_gate_and_fallback_block_release(self) -> None:
        module = importlib.import_module(
            f"{PROJECT}.semantic_parity.submission_manifest"
        )
        inventory = {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "checkpoint": {
                "update": 230, "sha256": "b" * 64,
                "schema_version": "checkpoint-v1",
                "source_actor_sha256": "0" * 64,
                "critical_missing_keys": [], "critical_unexpected_keys": [],
                "component_sha256": {
                    "base_encoder": "1" * 64, "option_encoder_lora": "2" * 64,
                    "action_decoder": "3" * 64, "value": "4" * 64,
                    "allocation_head": "5" * 64,
                },
            },
            "package": {
                "root": "package", "model_sha256": "6" * 64,
                "manifest_sha256": "7" * 64, "component_sha256": {},
                "portable_schema": "portable-v1", "checkpoint_update": 230,
                "strict_load": True, "critic_deployed": False,
                "silent_legacy_fallback": True, "model_eval": True,
                "storage_dtype": "fp16", "runtime_dtype": "fp32",
                "action_boundary": "ab", "decision_gate": "gate",
                "canonicalizer": "canon", "official_protocol_adapter": "adapter",
            },
            "runtime": {
                "card_database_sha256": "8" * 64, "cuda_rules_sha256": "9" * 64,
                "cuda_extension_sha256": "a" * 64,
                "official_cpu_library_sha256": "b" * 64,
            },
            "contracts": {
                "action_boundary": "ab", "decision_gate": "gate",
                "canonicalizer": "canon", "trajectory": "trajectory",
                "official_protocol_adapter": "adapter",
            },
        }
        result = module.build_manifest(
            inventory=inventory,
            gates={name: {"status": "FAIL" if name == "C" else "PASS"}
                   for name in "ABCD"},
        )
        self.assertFalse(result["release_ready"])
        self.assertIn("package contains silent macro-to-policy fallback", result["release_blockers"])
        self.assertIn("Gate C is FAIL", result["release_blockers"])

    def test_validator_rejects_removed_field_and_false_ready_claim(self) -> None:
        module = importlib.import_module(
            f"{PROJECT}.semantic_parity.submission_manifest"
        )
        manifest = {
            "schema_version": "0038_submission_release_manifest_v1",
            "release_ready": False,
            "release_blockers": ["blocked"],
            "checkpoint": {
                "sha256": "0" * 64, "base_model_sha256": "1" * 64,
                "base_encoder_component_sha256": "2" * 64,
                "lora_sha256": "3" * 64, "action_decoder_sha256": "4" * 64,
                "value_sha256": "5" * 64, "allocation_head_sha256": "6" * 64,
            },
            "package": {
                "model_sha256": "7" * 64, "manifest_sha256": "8" * 64,
                "silent_legacy_fallback": False, "strict_load": True,
            },
            "source": {"git_dirty": False},
            "contracts": {}, "decision_gate_config": {},
            "precision_backend": {}, "inference": {},
            "gates": {name: "PASS" for name in "ABCD"},
        }
        module.validate_release_manifest(manifest)
        for key in ("checkpoint", "package", "contracts", "gates"):
            mutated = dict(manifest)
            mutated.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.validate_release_manifest(mutated)
        invalid = dict(manifest)
        invalid["release_ready"] = True
        with self.assertRaises(ValueError):
            module.validate_release_manifest(invalid)


if __name__ == "__main__":
    unittest.main()
