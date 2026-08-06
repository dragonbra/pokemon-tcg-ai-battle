from __future__ import annotations

import importlib
import json
from pathlib import Path
import unittest


runtime = importlib.import_module(
    "train.0033_dragapult_third_ptcg_club_rl.training.runtime"
)
transfer = importlib.import_module("train.0033_dragapult_third_ptcg_club_rl.transfer")


class ProjectIdentityTest(unittest.TestCase):
    def test_exact_focal_deck_identity(self) -> None:
        deck = runtime.read_deck(runtime.FOCAL_DECK_PATH)
        self.assertEqual(len(deck), 60)
        self.assertEqual(
            runtime.deck_multiset_sha256(deck), runtime.FOCAL_DECK_MULTISET_SHA256
        )
        manifest = json.loads(runtime.FOCAL_DECK_PATH.with_name("manifest.json").read_text())
        self.assertEqual(manifest["deck_id"], runtime.FOCAL_DECK_ID)
        self.assertEqual(
            manifest["deck_file_sha256"], runtime.FOCAL_DECK_FILE_SHA256
        )
        self.assertEqual(
            manifest["deck_multiset_sha256"], runtime.FOCAL_DECK_MULTISET_SHA256
        )

    def test_pt0805_epoch13_source_identity(self) -> None:
        self.assertEqual(transfer.SOURCE_EPOCH, 13)
        self.assertEqual(transfer.SOURCE_GLOBAL_STEP, 109135)
        self.assertEqual(
            transfer.SOURCE_CHECKPOINT_SHA256,
            "285b88f5e30c40ad07ad025b429c5bd1594f0359cc0d44849c368056222d63ae",
        )

    def test_cuda_opponent_snapshot_is_complete(self) -> None:
        path = Path("experiments/0033_dragapult_third_ptcg_club_rl/") / (
            "cuda_supported_frozen0019_snapshot.json"
        )
        payload = json.loads(path.read_text())
        self.assertEqual(payload["decks_per_update"], 38)
        self.assertEqual(len(payload["opponents"]), 38)
        self.assertEqual(len({row["deck_id"] for row in payload["opponents"]}), 38)
        self.assertEqual(
            payload["foundation_checkpoint_sha256"],
            "da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb",
        )


if __name__ == "__main__":
    unittest.main()
