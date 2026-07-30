from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ..decoder import (
    DECODER_COMPONENTS,
    create_value_head,
    extract_decoder_state,
    load_decoder_checkpoint,
    save_decoder_checkpoint,
)
from ..foundation import load_foundation


class DecoderCheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.identity = load_foundation("cpu")

    def test_extraction_is_exact_and_independent(self) -> None:
        first = extract_decoder_state(self.model, create_value_head(self.model.config.d_model))
        second = extract_decoder_state(self.model, create_value_head(self.model.config.d_model))
        expected_prefixes = {f"decoder.{name}." for name in DECODER_COMPONENTS}
        actual_prefixes = {
            prefix
            for prefix in expected_prefixes
            if any(key.startswith(prefix) for key in first)
        }
        self.assertEqual(
            actual_prefixes,
            expected_prefixes,
        )
        self.assertTrue(any(key.startswith("value_head.") for key in first))
        for key in first:
            self.assertNotEqual(first[key].data_ptr(), second[key].data_ptr())

    def test_atomic_model_only_round_trip_and_forbidden_field_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "deck.pt"
            value_head = create_value_head(self.model.config.d_model)
            digest = save_decoder_checkpoint(
                path,
                self.model,
                value_head,
                foundation_sha256=self.identity.weights_sha256,
                deck_id="test_deck",
                deck_sha256="1" * 64,
                policy_role="live",
                policy_version="V1_test",
                update=0,
            )
            audit = load_decoder_checkpoint(
                path,
                expected_foundation_sha256=self.identity.weights_sha256,
                expected_deck_id="test_deck",
                expected_deck_sha256="1" * 64,
                model=self.model,
                value_head=value_head,
            )
            self.assertEqual(audit.checkpoint_sha256, digest)
            self.assertTrue(audit.includes_value_head)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["optimizer"] = {"state": {}}
            bad = root / "bad.pt"
            torch.save(payload, bad)
            with self.assertRaisesRegex(ValueError, "forbidden training state"):
                load_decoder_checkpoint(
                    bad,
                    expected_foundation_sha256=self.identity.weights_sha256,
                    expected_deck_id="test_deck",
                    expected_deck_sha256="1" * 64,
                )


if __name__ == "__main__":
    unittest.main()
