from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))

from ptcg_cuda_engine.corpus import (  # noqa: E402
    PolicyCodecV1CorpusWriter,
    iter_policy_codec_v1_shards,
    pad_policy_codec_v1_rows,
)


class FakeBatch:
    def __init__(self) -> None:
        self.global_cat = np.arange(16, dtype=np.int64).reshape(2, 8)
        self.global_num = np.arange(32, dtype=np.float32).reshape(2, 16) / 10.0
        self.entity_cat = np.zeros((2, 3, 6), dtype=np.int64)
        self.entity_num = np.zeros((2, 3, 10), dtype=np.float32)
        self.entity_parent = np.full((2, 3), -1, dtype=np.int64)
        self.option_cat = np.zeros((2, 3, 12), dtype=np.int64)
        self.option_num = np.zeros((2, 3, 4), dtype=np.float32)
        self.option_equiv = np.full((2, 3), -1, dtype=np.int64)
        self.scalars = np.asarray(
            [
                [2, 3, 1, 2, 4, 0, 1, 6, 6],
                [1, 2, 0, 1, 2, 1, 2, 5, 6],
            ],
            dtype=np.int64,
        )
        self.entity_cat[0, :2, 0] = [100, 101]
        self.entity_cat[1, 0, 0] = 200
        self.entity_num[0, :2, 0] = [0.5, 0.75]
        self.entity_num[1, 0, 0] = 1.0
        self.entity_parent[0, 1] = 0
        self.option_cat[0, :3, 0] = [1, 2, 3]
        self.option_cat[1, :2, 0] = [4, 5]
        self.option_num[0, :3, 2] = [0.0, 0.1, 0.2]
        self.option_num[1, :2, 2] = [0.0, 0.1]
        self.option_equiv[0, :3] = [0, 1, 1]
        self.option_equiv[1, :2] = [0, 1]


def write_fixture(path: Path) -> dict:
    writer = PolicyCodecV1CorpusWriter(
        path,
        shard_decisions=2,
        entity_capacity=4,
        option_capacity=4,
        metadata={"seed": 7, "source": "unit"},
    )
    writer.append_batch(
        FakeBatch(),
        state_digests=[11, 12],
        game_indices=[3, 4],
        steps=[5, 6],
        select_types=[1, 2],
        select_contexts=[7, 8],
        select_players=[0, 1],
        acting_agent_ids=[9, 10],
        actions=[[1], []],
    )
    return writer.close(summary={"status": "pass"})


class PolicyCodecCorpusTest(unittest.TestCase):
    def test_round_trip_ragged_rows_and_padding(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "corpus"
            manifest = write_fixture(root)
            self.assertEqual(manifest["totals"]["decisions"], 2)
            shard = next(iter_policy_codec_v1_shards(root))
            padded = pad_policy_codec_v1_rows(
                shard.arrays,
                0,
                2,
                entity_capacity=4,
                option_capacity=4,
            )
            self.assertEqual(tuple(padded["entity_cat"].shape), (2, 4, 6))
            self.assertEqual(tuple(padded["option_cat"].shape), (2, 4, 12))
            self.assertEqual(padded["entity_mask"].tolist(), [[True, True, False, False], [True, False, False, False]])
            self.assertEqual(padded["option_mask"].tolist(), [[True, True, True, False], [True, True, False, False]])
            self.assertEqual(padded["actions"].tolist(), [[1], [-1]])
            self.assertEqual(padded["action_len"].tolist(), [1, 0])
            self.assertEqual(padded["entity_parent"][0].tolist(), [-1, 0, -1, -1])
            self.assertEqual(padded["state_digest"].tolist(), [11, 12])

    def test_archive_bytes_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "first"
            second = Path(temp) / "second"
            first_manifest = write_fixture(first)
            second_manifest = write_fixture(second)
            first_shard = first / first_manifest["shards"][0]["path"]
            second_shard = second / second_manifest["shards"][0]["path"]
            self.assertEqual(first_shard.read_bytes(), second_shard.read_bytes())
            self.assertEqual(
                hashlib.sha256(first_shard.read_bytes()).hexdigest(),
                first_manifest["shards"][0]["sha256"],
            )
            self.assertEqual(
                (first / "manifest.json").read_text(encoding="utf-8"),
                (second / "manifest.json").read_text(encoding="utf-8"),
            )

    def test_writer_rejects_action_outside_selection_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            writer = PolicyCodecV1CorpusWriter(
                Path(temp) / "corpus",
                entity_capacity=4,
                option_capacity=4,
            )
            writer.append_batch(
                FakeBatch(),
                state_digests=[11, 12],
                game_indices=[3, 4],
                steps=[5, 6],
                select_types=[1, 2],
                select_contexts=[7, 8],
                select_players=[0, 1],
                acting_agent_ids=[9, 10],
                actions=[[1], [3]],
            )
            with self.assertRaisesRegex(ValueError, "out-of-range"):
                writer.close()


if __name__ == "__main__":
    unittest.main()
