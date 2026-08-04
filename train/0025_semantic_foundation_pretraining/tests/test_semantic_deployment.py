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


BASE = "train.0025_semantic_foundation_pretraining"
MODEL = importlib.import_module(f"{BASE}.model.multi_memory")
PROTOTYPES = importlib.import_module(f"{BASE}.features.prototypes")
SEMANTIC_INFERENCE = importlib.import_module(f"{BASE}.deployment.semantic_inference")
EXPORT = importlib.import_module(f"{BASE}.export_candidate")

ROOT = Path(__file__).resolve().parents[3]
PROTOTYPE_PATH = (
    ROOT
    / "train/0025_semantic_foundation_pretraining/assets/official_public_prototypes_v1.json"
)
RAW = (
    ROOT
    / "rl_runs/0019_universal_winner_bc/dataset/"
    "V1_universal_winner_raw/train-00000.jsonl.gz"
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


class SemanticDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.prototypes = PROTOTYPES.PrototypeIndex.load(PROTOTYPE_PATH)

    def _checkpoint(self, root: Path) -> tuple[Path, list[int]]:
        row, deck = _first_raw_row()
        config = MODEL.SemanticModelConfig(d_model=64, heads=4, dropout=0.0)
        model = MODEL.SemanticFoundationPolicy(config, self.prototypes)
        checkpoint = root / "semantic.pt"
        torch.save(
            {
                "schema_version": "0025_model_only_checkpoint_v1",
                "state_dict": model.state_dict(),
                "metadata": {
                    "arm": "semantic",
                    "epoch": 3,
                    "project_id": "0029_multi_deck_semantic_foundation_bc",
                    "version": "V1_archetype_balanced_paired_bc",
                    "model_config": {
                        "model": "SemanticFoundationPolicy",
                        "config": asdict(config),
                        "parameter_count": sum(p.numel() for p in model.parameters()),
                        "actor_view": "legacy_plus_typed_multi_memory",
                    },
                    "validation": {"exact_action": 0.5},
                },
            },
            checkpoint,
        )
        self.assertEqual(len(deck), 60)
        self.assertIsNotNone(row["actor_observation"].get("select"))
        return checkpoint, deck

    def test_checkpoint_online_encode_and_batched_greedy(self) -> None:
        row, _ = _first_raw_row()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, deck = self._checkpoint(Path(directory))
            policy = SEMANTIC_INFERENCE.PortableSemanticPolicy.from_checkpoint(
                checkpoint, deck
            )
            encoder = policy.online_encoder(
                int(row["actor_observation"]["current"]["yourIndex"]), deck
            )
            batch = encoder.encode(row["actor_observation"])
            with torch.inference_mode():
                decoded = policy.model.deterministic_action_tensors(batch)
            self.assertTrue(bool(decoded.legal[0]))
            self.assertGreaterEqual(int(decoded.lengths[0]), int(batch["min_count"][0]))
            self.assertLessEqual(int(decoded.lengths[0]), int(batch["max_count"][0]))
            self.assertFalse(policy.requires_source_id)
            self.assertTrue(policy.fail_closed_inference_errors)

    def test_semantic_export_is_self_contained_and_preserves_provenance(self) -> None:
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
                deck_id="test_semantic_deck",
            )
            self.assertEqual(manifest["arm"], "semantic")
            self.assertEqual(
                manifest["project_id"], "0029_multi_deck_semantic_foundation_bc"
            )
            self.assertEqual(len((output / "deck.csv").read_text().splitlines()), 60)
            self.assertTrue((output / "strategy/deployment/semantic_inference.py").is_file())
            self.assertTrue((output / "strategy/model/multi_memory.py").is_file())
            self.assertFalse(any(path.is_symlink() for path in output.rglob("*")))

            environment = dict(os.environ)
            environment.pop("PYTHONPATH", None)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import main; assert len(main.read_deck_csv()) == 60",
                ],
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
