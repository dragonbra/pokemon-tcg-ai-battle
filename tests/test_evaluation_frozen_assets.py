from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = ROOT / "evaluation" / "arena" / "frozen"
CATALOG_PATH = ROOT / "evaluation" / "configs" / "frozen.json"
EXPECTED_FOUNDATION_SHA256 = (
    "da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb"
)


def _sorted_deck_sha(deck: list[int]) -> str:
    canonical = ",".join(str(card_id) for card_id in sorted(deck)).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


class FrozenArenaAssetTest(unittest.TestCase):
    def test_frozen_catalog_has_one_policy_and_49_exact_decks(self) -> None:
        manifest = json.loads((FROZEN_ROOT / "manifest.json").read_text())
        catalog = json.loads(CATALOG_PATH.read_text())
        entries = catalog["opponents"]

        self.assertEqual(manifest["schema_version"], "evaluation_frozen_arena_v1")
        self.assertEqual(manifest["foundation"]["weights_sha256"], EXPECTED_FOUNDATION_SHA256)
        self.assertEqual(manifest["foundation"]["deployment_source_id"], 0)
        self.assertEqual(manifest["deck_count"], 49)
        self.assertEqual(len(entries), 49)
        self.assertEqual(len({entry["name"] for entry in entries}), 49)
        self.assertTrue((FROZEN_ROOT / "_policy" / "strategy" / "model.bin").is_file())

        deck_hashes: set[str] = set()
        for entry in entries:
            deck_id = entry["name"]
            deck_root = FROZEN_ROOT / deck_id
            with self.subTest(deck_id=deck_id):
                self.assertEqual(entry["package"], f"arena/frozen/{deck_id}")
                self.assertTrue(entry["enabled"])
                self.assertIn("foundation_frozen", entry["tags"])
                self.assertTrue(entry["display_name"].startswith("[Frozen 0019] "))
                deck = [int(value) for value in (deck_root / "deck.csv").read_text().splitlines()]
                self.assertEqual(len(deck), 60)
                deck_manifest = json.loads((deck_root / "manifest.json").read_text())
                self.assertEqual(deck_manifest["deck_id"], deck_id)
                self.assertEqual(deck_manifest["decoder_ref"], "foundation")
                self.assertEqual(deck_manifest["foundation_sha256"], EXPECTED_FOUNDATION_SHA256)
                self.assertEqual(deck_manifest["deck_sha256"], _sorted_deck_sha(deck))
                self.assertNotIn(deck_manifest["deck_sha256"], deck_hashes)
                deck_hashes.add(deck_manifest["deck_sha256"])
                self.assertTrue(set(entry["representative_card_ids"]).issubset(deck))

    def test_frozen_deck_identities_are_lightweight_and_cache_free(self) -> None:
        for path in FROZEN_ROOT.iterdir():
            if not path.is_dir() or path.name == "_policy":
                continue
            with self.subTest(deck_id=path.name):
                self.assertEqual({item.name for item in path.iterdir()}, {"deck.csv", "manifest.json"})
                self.assertFalse(path.is_symlink())
        self.assertFalse(any("__pycache__" in path.parts for path in FROZEN_ROOT.rglob("*")))
        self.assertFalse(any(path.suffix == ".pyc" for path in FROZEN_ROOT.rglob("*")))


if __name__ == "__main__":
    unittest.main()
