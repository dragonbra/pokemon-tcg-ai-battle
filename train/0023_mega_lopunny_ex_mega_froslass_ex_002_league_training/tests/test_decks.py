from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ..decks import DeckRole, load_deck_plugins


def _write_plugin(
    root: Path,
    deck_id: str,
    *,
    role: str = "live",
    focal: bool = True,
    offset: int = 0,
) -> Path:
    plugin = root / deck_id
    plugin.mkdir()
    manifest = {
        "schema_version": "0023_league_deck_plugin_v1",
        "deck_id": deck_id,
        "display_name": deck_id,
        "role": role,
        "focal": focal,
        "decoder_ref": "foundation",
        "decoder_sha256": None,
        "provenance": {
            "source": "test source",
            "evidence": "test evidence",
            "captured_at": "2026-07-31",
        },
    }
    (plugin / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (plugin / "deck.csv").write_text(
        "".join(f"{offset + index + 1}\n" for index in range(60)), encoding="ascii"
    )
    return plugin


class DeckPluginTest(unittest.TestCase):
    def test_empty_staging_is_valid_and_deterministic_plugins_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(load_deck_plugins(root), ())
            _write_plugin(root, "zeta_deck", offset=100)
            _write_plugin(root, "alpha_deck")
            plugins = load_deck_plugins(root)
            self.assertEqual([plugin.deck_id for plugin in plugins], ["alpha_deck", "zeta_deck"])
            self.assertEqual(plugins[0].role, DeckRole.LIVE)
            self.assertEqual(len(plugins[0].deck), 60)

    def test_exact_60_positive_ids_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plugin = _write_plugin(Path(temporary), "bad_deck")
            (plugin / "deck.csv").write_text("1\n" * 59, encoding="ascii")
            with self.assertRaisesRegex(ValueError, "exactly 60"):
                load_deck_plugins(Path(temporary))

    def test_manifest_and_files_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin = _write_plugin(root, "valid_deck")
            (plugin / "notes.txt").write_text("unexpected", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "unknown files"):
                load_deck_plugins(root)

    def test_duplicate_exact_deck_and_frozen_focal_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_plugin(root, "first_deck")
            _write_plugin(root, "second_deck")
            with self.assertRaisesRegex(ValueError, "duplicate exact deck"):
                load_deck_plugins(root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_plugin(root, "frozen_deck", role="frozen", focal=True)
            with self.assertRaisesRegex(ValueError, "only a Live deck"):
                load_deck_plugins(root)


if __name__ == "__main__":
    unittest.main()
