from __future__ import annotations

from dataclasses import asdict
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


BASE = "train.0035_lifetime_aware_feature_compiler"
MODEL = importlib.import_module(f"{BASE}.model")
PROTOTYPES = importlib.import_module(f"{BASE}.domain.prototypes")
INFERENCE = importlib.import_module(f"{BASE}.deployment.inference")
ONLINE_RUNTIME = importlib.import_module(f"{BASE}.deployment.online_runtime")
EXPORT = importlib.import_module(f"{BASE}.export_candidate")
BENCHMARK = importlib.import_module(f"{BASE}.benchmark_incremental_features")

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "train/0035_lifetime_aware_feature_compiler/assets"
def _first_raw_row() -> tuple[dict, list[int]]:
    trajectory = BENCHMARK.load_parity_trajectories()[0]
    return trajectory.decisions[0].row(), list(trajectory.deck)


class DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.prototypes = PROTOTYPES.PrototypeIndex.load(
            ASSETS / "official_public_prototypes_v1.json",
            ASSETS / "official_full_engine_prototypes_v2.json",
        )

    def _checkpoint(self, root: Path) -> tuple[Path, list[int]]:
        _, deck = _first_raw_row()
        config = MODEL.ModelConfig(
            d_model=64,
            heads=4,
            state_layers=1,
            event_layers=1,
            option_layers=1,
            dropout=0.0,
        )
        model = MODEL.SemanticPolicy(config, self.prototypes)
        checkpoint = root / "latest.pt"
        torch.save(
            {
                "schema_version": "0031_model_only_checkpoint_v1",
                "state_dict": model.state_dict(),
                "metadata": {
                    "project_id": "0031_rule_faithful_semantic_foundation_pretraining",
                    "version": "V4_lr5e4_no_early_stop_b512",
                    "arm": "rule_faithful_semantic",
                    "epoch": 2,
                    "global_step": 33002,
                    "model_config": {
                        "model": "SemanticPolicy",
                        "config": asdict(config),
                        "parameter_count": sum(p.numel() for p in model.parameters()),
                        "actor_view": "canonical_typed_state_and_option_relations_only",
                    },
                },
            },
            checkpoint,
        )
        return checkpoint, deck

    def test_checkpoint_online_encode_and_greedy(self) -> None:
        row, _ = _first_raw_row()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, deck = self._checkpoint(Path(directory))
            policy = INFERENCE.PortableSemanticPolicy.from_checkpoint(checkpoint, deck)
            action = policy.select(row["actor_observation"])
            select = row["actor_observation"]["select"]
            self.assertGreaterEqual(len(action), int(select["minCount"]))
            self.assertLessEqual(len(action), int(select["maxCount"]))
            self.assertEqual(len(action), len(set(action)))
            self.assertTrue(all(0 <= value < len(select["option"]) for value in action))
            self.assertFalse(policy.requires_source_id)
            self.assertTrue(policy.fail_closed_inference_errors)
            self.assertEqual(
                policy.model.prototype_cache_stats()["builds"], 1
            )
            self.assertFalse(
                any("prototype_cache" in name for name in policy.model.state_dict())
            )

    def test_incremental_online_record_matches_full_rebuild(self) -> None:
        row, deck = _first_raw_row()
        observation = row["actor_observation"]
        actor = int(observation["current"]["yourIndex"])
        config = MODEL.ModelConfig(
            d_model=64,
            heads=4,
            state_layers=1,
            event_layers=1,
            option_layers=1,
            dropout=0.0,
        )
        full = ONLINE_RUNTIME.OnlineCausalEncoder(
            actor, deck, config, incremental=False
        )
        incremental = ONLINE_RUNTIME.OnlineCausalEncoder(
            actor, deck, config, incremental=True
        )

        full_record = full.encode_record(observation)
        incremental_record = incremental.encode_record(observation)

        self.assertEqual(incremental_record, full_record)
        full_batch = importlib.import_module(f"{BASE}.features.collate").collate_canonical_records(
            [full_record]
        )
        incremental_batch = importlib.import_module(
            f"{BASE}.features.collate"
        ).collate_canonical_records([incremental_record])
        self.assertEqual(set(incremental_batch), set(full_batch))
        for name in sorted(full_batch):
            with self.subTest(name=name):
                self.assertTrue(torch.equal(incremental_batch[name], full_batch[name]))

    def test_process_and_battle_constants_are_reused(self) -> None:
        row, deck = _first_raw_row()
        observation = row["actor_observation"]
        actor = int(observation["current"]["yourIndex"])
        config = MODEL.ModelConfig(d_model=64, heads=4, dropout=0.0)
        first = ONLINE_RUNTIME.OnlineCausalEncoder(actor, deck, config)
        second = ONLINE_RUNTIME.OnlineCausalEncoder(actor, deck, config)
        self.assertIs(first.prototypes, second.prototypes)
        manifest = first.deck_manifest
        first.encode_record(observation)
        self.assertIs(first.deck_manifest, manifest)

    def test_persistent_tensor_runtime_matches_reference_chronologically(self) -> None:
        trajectory = BENCHMARK.load_parity_trajectories()[0]
        config = MODEL.ModelConfig(d_model=64, heads=4, dropout=0.0)
        reference = ONLINE_RUNTIME.OnlineCausalEncoder(
            trajectory.actor, trajectory.deck, config, incremental=True
        )
        persistent = ONLINE_RUNTIME.OnlineCausalEncoder(
            trajectory.actor,
            trajectory.deck,
            config,
            incremental=True,
            persistent_tensors=True,
        )
        for decision_index, decision in enumerate(trajectory.decisions):
            expected = reference.encode(decision.observation)
            actual = persistent.encode(decision.observation)
            self.assertEqual(actual.keys(), expected.keys())
            for name in expected:
                with self.subTest(decision=decision_index, name=name):
                    self.assertEqual(actual[name].dtype, expected[name].dtype)
                    self.assertEqual(actual[name].shape, expected[name].shape)
                    self.assertTrue(torch.equal(actual[name], expected[name]))
        stats = persistent.tensor_bank.stats.snapshot()
        self.assertEqual(stats["collates"], len(trajectory.decisions))
        self.assertGreater(stats["slot_hits"], 0)

    def test_export_is_self_contained_and_preserves_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, deck = self._checkpoint(root)
            deck_path = root / "deck.csv"
            deck_path.write_text(
                "".join(f"{card_id}\n" for card_id in deck), encoding="ascii"
            )
            cg = root / "cg_source"
            cg.mkdir()
            (cg / "__init__.py").write_text("", encoding="ascii")
            (cg / "api.py").write_text("", encoding="ascii")
            output = root / "candidate"
            manifest = EXPORT.export_candidate(
                checkpoint=checkpoint,
                deck_path=deck_path,
                cg_source=cg,
                output=output,
                deck_id="test_rule_faithful_deck",
            )
            self.assertEqual(manifest["project_id"], BASE.split(".", 1)[1])
            self.assertEqual(
                manifest["source_checkpoint_project_id"],
                "0031_rule_faithful_semantic_foundation_pretraining",
            )
            self.assertEqual(
                manifest["compiler_implementation_version"],
                "0035_lifetime_aware_feature_compiler_v1",
            )
            self.assertEqual(manifest["checkpoint_selection"], "latest")
            self.assertEqual(manifest["checkpoint_epoch"], 2)
            self.assertEqual(len((output / "deck.csv").read_text().splitlines()), 60)
            self.assertTrue((output / "strategy/inference.py").is_file())
            self.assertTrue((output / "strategy/online_runtime.py").is_file())
            self.assertTrue((output / "strategy/features/layers.py").is_file())
            self.assertTrue((output / "strategy/features/incremental.py").is_file())
            self.assertTrue((output / "strategy/features/fragments.py").is_file())
            self.assertTrue((output / "strategy/features/tensor_bank.py").is_file())
            self.assertTrue((output / "strategy/model/policy.py").is_file())
            self.assertFalse(any(path.is_symlink() for path in output.rglob("*")))
            portable = torch.load(
                output / "strategy/model.bin", map_location="cpu", weights_only=True
            )
            self.assertEqual(
                portable["schema_version"],
                "0031_shared_prototype_candidate_checkpoint_v1",
            )
            self.assertTrue(
                any(key.startswith("prototype_encoder.") for key in portable["state_dict"])
            )
            self.assertFalse(
                any(
                    key.startswith(("state_encoder.prototypes.", "option_encoder.prototypes."))
                    for key in portable["state_dict"]
                )
            )

            fp16_output = root / "candidate_fp16"
            fp16_manifest = EXPORT.export_candidate(
                checkpoint=checkpoint,
                deck_path=deck_path,
                cg_source=cg,
                output=fp16_output,
                deck_id="test_rule_faithful_deck",
                storage_dtype="fp16",
                runtime_dtype="fp16",
            )
            self.assertEqual(fp16_manifest["storage_dtype"], "fp16")
            fp16_payload = torch.load(
                fp16_output / "strategy/model.bin",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(
                fp16_payload["schema_version"],
                "0031_shared_prototype_fp16_storage_candidate_checkpoint_v1",
            )
            self.assertTrue(
                all(
                    value.dtype == torch.float16
                    for value in fp16_payload["state_dict"].values()
                    if torch.is_floating_point(value)
                )
            )
            fp16_policy = INFERENCE.PortableSemanticPolicy.from_checkpoint(
                fp16_output / "strategy/model.bin", deck
            )
            self.assertTrue(
                all(
                    parameter.dtype == torch.float16
                    for parameter in fp16_policy.model.parameters()
                )
            )
            row, _ = _first_raw_row()
            action = fp16_policy.select(row["actor_observation"])
            select = row["actor_observation"]["select"]
            self.assertGreaterEqual(len(action), int(select["minCount"]))
            self.assertLessEqual(len(action), int(select["maxCount"]))

            fp16_fp32_output = root / "candidate_fp16_storage_fp32_runtime"
            fp16_fp32_manifest = EXPORT.export_candidate(
                checkpoint=checkpoint,
                deck_path=deck_path,
                cg_source=cg,
                output=fp16_fp32_output,
                deck_id="test_rule_faithful_deck",
                storage_dtype="fp16",
                runtime_dtype="fp32",
            )
            self.assertEqual(fp16_fp32_manifest["storage_dtype"], "fp16")
            self.assertEqual(fp16_fp32_manifest["runtime_dtype"], "fp32")
            fp16_fp32_policy = INFERENCE.PortableSemanticPolicy.from_checkpoint(
                fp16_fp32_output / "strategy/model.bin", deck
            )
            self.assertTrue(
                all(
                    parameter.dtype == torch.float32
                    for parameter in fp16_fp32_policy.model.parameters()
                )
            )

            environment = dict(os.environ)
            environment.pop("PYTHONPATH", None)
            completed = subprocess.run(
                [sys.executable, "-c", "import main; assert len(main.read_deck_csv()) == 60"],
                cwd=output,
                env=environment,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
