from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import torch


PROJECT = "train.0042_full_model_design"
ROOT = Path(__file__).resolve().parents[3]
DECK = tuple(
    int(value)
    for value in (
        ROOT
        / "train/0042_full_model_design/league/decks/007_dragapult_ex/deck.csv"
    ).read_text(encoding="ascii").splitlines()
)


class PolicyIdentityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.identity = importlib.import_module(f"{PROJECT}.policy_identity")
        cls.registry = cls.identity.load_policy_registry()

    def test_policy_0806_registry_is_full_0806(self) -> None:
        entry = self.registry["Policy-0806"]
        self.assertEqual(entry["policy_kind"], "immutable_pretrained")
        self.assertEqual(
            set(entry["components"]), set(self.identity.EFFECTIVE_COMPONENTS)
        )
        self.assertTrue(all(
            component["source_policy_id"] == "Policy-0806"
            for component in entry["components"].values()
        ))
        audit = self.identity.audit_checkpoint("Policy-0806")
        self.assertEqual(audit.status, "PASS")
        self.assertEqual(audit.effective_policy_sha256, entry["effective_policy_sha256"])

    def test_policy_0809_registry_is_full_0809(self) -> None:
        entry = self.registry["Policy-0809"]
        self.assertTrue(all(
            component["source_policy_id"] == "Policy-0809"
            for component in entry["components"].values()
        ))
        audit = self.identity.audit_checkpoint("Policy-0809")
        self.assertEqual(audit.status, "PASS")
        self.assertEqual(audit.effective_policy_sha256, entry["effective_policy_sha256"])

    def test_focal_switch_does_not_change_frozen_0806_identity(self) -> None:
        before = self.identity.audit_checkpoint(
            "Policy-0806", focal_policy_id="Policy-0809"
        )
        after = self.identity.audit_checkpoint(
            "Policy-0806", focal_policy_id="Policy-future-0812"
        )
        self.assertEqual(before.effective_policy_sha256, after.effective_policy_sha256)
        self.assertEqual(before.components, after.components)

    def test_focal_trunk_plus_0806_decoder_hard_fails(self) -> None:
        frozen = self.identity.checkpoint_state_dict("Policy-0806")
        focal = self.identity.checkpoint_state_dict("Policy-0809")
        hybrid = dict(frozen)
        for name in tuple(hybrid):
            if not name.startswith("action_decoder."):
                hybrid[name] = focal[name]
        with self.assertRaisesRegex(
            self.identity.PolicyIdentityViolation,
            "FATAL: Opponent policy identity violation",
        ):
            self.identity.audit_materialized_state_dict("Policy-0806", hybrid)

    def test_promoted_snapshot_manifest_round_trip_is_immutable(self) -> None:
        base = self.registry["Policy-0809"]
        promoted_components = copy.deepcopy(base["components"])
        promoted_components["action_decoder"] = {
            "source_policy_id": "Policy-update00250",
            "effective_sha256": "a" * 64,
        }
        promoted_components["lora"] = {
            "source_policy_id": "Policy-update00250",
            "effective_sha256": "b" * 64,
        }
        manifest = self.identity.build_promoted_snapshot_manifest(
            policy_id="Policy-update00250",
            parent_policy_id="Policy-0809",
            base_checkpoint_sha256=base["base_checkpoint"]["sha256"],
            trained_checkpoint_sha256="c" * 64,
            components=promoted_components,
            model_schema_version=base["model_schema_version"],
            observation_schema_version=base["observation_schema_version"],
            action_schema_version=base["action_schema_version"],
            created_from_update=250,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy_registry.json"
            path.write_text(
                json.dumps({"schema_version": self.identity.REGISTRY_SCHEMA,
                            "policies": [manifest]}),
                encoding="utf-8",
            )
            loaded = self.identity.load_policy_registry(path)["Policy-update00250"]
        self.assertEqual(
            loaded["effective_policy_sha256"], manifest["effective_policy_sha256"]
        )
        changed = copy.deepcopy(manifest)
        changed["components"]["action_decoder"]["effective_sha256"] = "d" * 64
        with self.assertRaises(self.identity.PolicyIdentityViolation):
            self.identity.validate_policy_manifest(changed)

    def test_promoted_snapshot_reload_matches_promote_effective_identity(self) -> None:
        checkpoint = (
            ROOT
            / "rl_runs/0042_full_model_design/versions/"
              "V2_snapshot_loader_fix_long_run/checkpoint/update-000000.pt"
        )
        if not checkpoint.is_file():
            self.skipTest("0042 update-0 checkpoint is unavailable")
        promoted_model = self.identity.materialize_promoted_actor(
            "Policy-0809", checkpoint, DECK, "cpu"
        )
        hashes = self.identity.component_hashes(promoted_model.state_dict())
        del promoted_model
        base = self.registry["Policy-0809"]
        components = {
            name: {
                "source_policy_id": (
                    "Policy-0809"
                    if hashes[name] == base["components"][name]["effective_sha256"]
                    else "Policy-update000000-test"
                ),
                "effective_sha256": hashes[name],
            }
            for name in self.identity.EFFECTIVE_COMPONENTS
        }
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        manifest = self.identity.build_promoted_snapshot_manifest(
            policy_id="Policy-update000000-test",
            parent_policy_id="Policy-0809",
            base_checkpoint_sha256=base["base_checkpoint"]["sha256"],
            trained_checkpoint_sha256=digest,
            trained_checkpoint_path=str(checkpoint.relative_to(ROOT)),
            components=components,
            model_schema_version=base["model_schema_version"],
            observation_schema_version=base["observation_schema_version"],
            action_schema_version="0038_compound_action_v2_semantic_parity",
            created_from_update=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "policy_registry.json"
            registry_path.write_text(json.dumps({
                "schema_version": self.identity.REGISTRY_SCHEMA,
                "policies": [base, manifest],
            }), encoding="utf-8")
            reloaded = self.identity.materialize_policy(
                manifest["policy_id"], DECK, "cpu", purpose="snapshot_reload_test",
                registry_path=registry_path,
            )
        self.assertEqual(reloaded.audit.status, "PASS")
        self.assertEqual(
            reloaded.audit.effective_policy_sha256,
            manifest["effective_policy_sha256"],
        )

    def test_training_and_evaluation_share_one_resolver(self) -> None:
        training = self.identity.resolve_policy_identity(
            "Policy-0806", purpose="rl_rollout"
        )
        evaluation = self.identity.resolve_policy_identity(
            "Policy-0806", purpose="frozen_evaluation"
        )
        self.assertEqual(
            training.effective_policy_sha256, evaluation.effective_policy_sha256
        )
        self.assertEqual(training.components, evaluation.components)

    def test_legacy_routed_head_runtime_is_fail_closed(self) -> None:
        runtime = importlib.import_module(f"{PROJECT}.training.runtime")
        with self.assertRaisesRegex(RuntimeError, "FATAL.*Policy Identity"):
            runtime.build_runtime(
                rules=Path("rules"), extension_dir=Path("extension"),
                source_checkpoint=Path("source"), support_report=Path("support"),
                frozen_root=Path("frozen"), checkpoint_root=Path("checkpoints"),
            )


if __name__ == "__main__":
    unittest.main()
