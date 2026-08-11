from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import unittest

import torch


PROJECT = "train.0042_full_model_design"
ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = ROOT / "train/0042_full_model_design"
INITIALIZATION = importlib.import_module(f"{PROJECT}.initialization")
PRESETS = importlib.import_module(f"{PROJECT}.integrated.presets")
RUNNER = importlib.import_module(f"{PROJECT}.training.run_full_semantic")
SOURCE = importlib.import_module(f"{PROJECT}.source")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Initialization0042Test(unittest.TestCase):
    def test_snapshot_gate_selects_the_0042_strict_training_loader(self) -> None:
        runner_source = (PROJECT_ROOT / "training/run_full_semantic.py").read_text()
        scaffold_source = (
            ROOT / "engine_cuda/tools/run_official_semantic0031_v2_parity_scaffold.py"
        ).read_text()
        self.assertIn('"--training-project", f"train.{PROJECT}"', runner_source)
        self.assertIn('"--training-project"', scaffold_source)
        self.assertIn("training_project: str", scaffold_source)
        self.assertIn("def decode_strategy_conditioned", scaffold_source)
        self.assertIn('hasattr(policy, "policy_strategy_adapter")', scaffold_source)
        self.assertIn('"no_option_lora": True', scaffold_source)

    def test_sources_are_the_strict_paired_0809_checkpoints(self) -> None:
        self.assertEqual(
            SOURCE.ACTOR_CHECKPOINT,
            ROOT / "archive/pretrained/0031_friend_0809_gsb_v5_value_v9/model.pt",
        )
        self.assertEqual(
            SOURCE.VALUE_CHECKPOINT,
            ROOT / "archive/pretrained/0031_friend_0809_gsb_v5_value_v9/value_head.pt",
        )
        self.assertEqual(sha256(SOURCE.ACTOR_CHECKPOINT), SOURCE.ACTOR_SHA256)
        self.assertEqual(sha256(SOURCE.VALUE_CHECKPOINT), SOURCE.VALUE_SHA256)
        value = torch.load(SOURCE.VALUE_CHECKPOINT, map_location="cpu", weights_only=True)
        self.assertEqual(
            value["metadata"]["source_checkpoint_sha256"], SOURCE.ACTOR_SHA256
        )

    def test_u0_keeps_0809_decoder_and_value_and_only_reuses_allocation_bc(self) -> None:
        model, identity = INITIALIZATION.build_preset_from_common_update0(
            RUNNER.focal_deck(), PRESETS.preset("FULL_MODEL"), device="cpu"
        )
        self.assertEqual(identity.checkpoint_sha256, SOURCE.ACTOR_SHA256)

        actor_payload = torch.load(
            SOURCE.ACTOR_CHECKPOINT, map_location="cpu", weights_only=True
        )
        decoder = model.actor.action_decoder.state_dict()
        for name, value in decoder.items():
            torch.testing.assert_close(
                value,
                actor_payload["state_dict"][f"action_decoder.{name}"],
                rtol=0,
                atol=0,
            )

        value_payload = torch.load(
            SOURCE.VALUE_CHECKPOINT, map_location="cpu", weights_only=True
        )
        for name, value in model.value_head.state_dict().items():
            torch.testing.assert_close(
                value, value_payload["value_head_state_dict"][name], rtol=0, atol=0
            )

        allocation_payload = torch.load(
            INITIALIZATION.ALLOCATION_HEAD_CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )
        expected = allocation_payload["allocation_head_state_dict"]
        self.assertEqual(set(model.allocation_head.state_dict()), set(expected))
        for name, value in model.allocation_head.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        INITIALIZATION.assert_zero_gate_base(model)

    def test_allocation_sidecar_is_project_local_and_allocation_only(self) -> None:
        self.assertEqual(
            INITIALIZATION.ALLOCATION_HEAD_CHECKPOINT.parent,
            PROJECT_ROOT / "assets",
        )
        self.assertEqual(
            sha256(INITIALIZATION.ALLOCATION_HEAD_CHECKPOINT),
            INITIALIZATION.ALLOCATION_HEAD_SHA256,
        )
        payload = torch.load(
            INITIALIZATION.ALLOCATION_HEAD_CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )
        self.assertEqual(
            set(payload),
            {"schema_version", "allocation_head_state_dict", "metadata"},
        )
        self.assertEqual(payload["metadata"]["ppo_updates"], 0)
        self.assertEqual(
            payload["metadata"]["source_checkpoint_sha256"],
            INITIALIZATION.ALLOCATION_HEAD_SOURCE_SHA256,
        )
        self.assertTrue(payload["allocation_head_state_dict"])
        self.assertFalse(
            any("action_decoder" in name for name in payload["allocation_head_state_dict"])
        )

    def test_initialization_has_no_cross_project_run_asset_dependency(self) -> None:
        source = (PROJECT_ROOT / "initialization.py").read_text(encoding="utf-8")
        self.assertNotIn("rl_runs/0038_action_boundary_rl", source)

    def test_runtime_has_no_numbered_project_import(self) -> None:
        project_root = ROOT / "train/0042_full_model_design"
        offenders = []
        for path in project_root.rglob("*.py"):
            if "tests" in path.parts:
                continue
            source = path.read_text(encoding="utf-8")
            if "train.0038_action_boundary_rl" in source or "train.0040_dragapult" in source:
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
