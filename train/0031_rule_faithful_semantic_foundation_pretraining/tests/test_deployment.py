from __future__ import annotations

from dataclasses import asdict
import gzip
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


BASE = "train.0031_rule_faithful_semantic_foundation_pretraining"
MODEL = importlib.import_module(f"{BASE}.model")
PROTOTYPES = importlib.import_module(f"{BASE}.domain.prototypes")
INFERENCE = importlib.import_module(f"{BASE}.deployment.inference")
EXPORT = importlib.import_module(f"{BASE}.export_candidate")

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "train/0031_rule_faithful_semantic_foundation_pretraining/assets"
RAW = (
    ROOT
    / "rl_runs/0031_rule_faithful_semantic_foundation_pretraining/dataset/"
    "source_delta_20260802_raw/validation-00000.jsonl.gz"
)


def _first_raw_row() -> tuple[dict, list[int]]:
    with gzip.open(RAW, "rt", encoding="utf-8") as handle:
        row = json.loads(next(handle))
    deck = [
        int(card_id)
        for card_id, count in row["deck_manifest"]["counts"]
        for _ in range(int(count))
    ]
    return row, deck


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
            self.assertEqual(manifest["checkpoint_selection"], "latest")
            self.assertEqual(manifest["checkpoint_epoch"], 2)
            self.assertEqual(len((output / "deck.csv").read_text().splitlines()), 60)
            self.assertTrue((output / "strategy/inference.py").is_file())
            self.assertTrue((output / "strategy/model/policy.py").is_file())
            self.assertFalse(any(path.is_symlink() for path in output.rglob("*")))

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
