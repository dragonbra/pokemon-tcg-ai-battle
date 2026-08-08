from __future__ import annotations

import gzip
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


builder = importlib.import_module(
    "train.0037_dragapult_value_initialized_rl.data.build_adaptation_corpus"
)
CUDA_PYTHON = Path(__file__).resolve().parents[3] / "engine_cuda" / "python"
if str(CUDA_PYTHON) not in sys.path:
    sys.path.insert(0, str(CUDA_PYTHON))
from ptcg_cuda_engine.corpus import iter_policy_codec_v1_shards, pad_policy_codec_v1_rows


class AdaptationDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = Path(
            "rl_runs/0031_rule_faithful_semantic_foundation_pretraining/dataset/"
            "source_delta_20260802_raw/train-00000.jsonl.gz"
        )
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            cls.raw = json.loads(next(handle))

    def test_preserves_ordered_action_and_provenance_outside_actor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "train"
            manifest = builder.build_split(
                [self.raw], root, split="train", limit=1, shard_decisions=1
            )
            shard = next(iter_policy_codec_v1_shards(root))
            padded = pad_policy_codec_v1_rows(
                shard.arrays, 0, 1, entity_capacity=128, option_capacity=128
            )
            length = int(padded["action_len"][0])
            self.assertEqual(
                padded["actions"][0, :length].tolist(), self.raw["ordered_action"]
            )
            self.assertEqual(manifest["metadata"]["split"], "train")
            self.assertEqual(manifest["summary"]["selected_decisions"], 1)
            self.assertIn(str(self.raw["source_id"]), manifest["summary"]["source_counts"])
            actor_fields = set(padded) - {
                "actions", "action_len", "action_family", "state_digest", "game_index",
                "step", "select_type", "select_context", "select_player", "acting_agent_id",
            }
            self.assertNotIn("source_id", actor_fields)

    def test_rejects_cross_split_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "no valid rows"):
                builder.build_split(
                    [self.raw],
                    Path(temporary) / "validation",
                    split="validation",
                    limit=1,
                    shard_decisions=1,
                )


if __name__ == "__main__":
    unittest.main()
