from __future__ import annotations

import importlib
from pathlib import Path
import unittest

import torch


module = importlib.import_module("train.0042_full_model_design.model.frozen_heads")
ROOT = Path(__file__).resolve().parents[3]
FrozenDeckHeads = module.FrozenDeckHeads
DECODER_NAMES = module.DECODER_NAMES


class FrozenHeadsTest(unittest.TestCase):
    def test_0019_foundation_matches_materialized_0022_head(self) -> None:
        foundation = ROOT / "archive/pretrained/0019_universal_winner_bc_0730_epoch13/model.pt"
        materialized = (
            ROOT
            / "rl_runs/0022_league_training/versions/V11_multidecoder_league_20h/checkpoint/decks/alakazam_dudunsparce_001.pt"
        )
        if not foundation.is_file() or not materialized.is_file():
            self.skipTest("real 0019/0022 assets are unavailable")
        shared = FrozenDeckHeads.from_0019_foundation(foundation)
        deck_payload = torch.load(materialized, map_location="cpu", weights_only=True)
        for name in DECODER_NAMES:
            actual = getattr(shared, name.replace(".", "__"))[0]
            expected = deck_payload["state"][f"decoder.{name}"]
            self.assertTrue(torch.equal(actual, expected), name)

    def test_vectorized_gru_matches_grucell(self) -> None:
        width = 5
        cell = torch.nn.GRUCell(width, width)
        state = {f"decoder.{name}": torch.zeros(1) for name in module.DECODER_NAMES}
        state.update(
            {
                "decoder.decoder.weight_ih": cell.weight_ih.detach(),
                "decoder.decoder.weight_hh": cell.weight_hh.detach(),
                "decoder.decoder.bias_ih": cell.bias_ih.detach(),
                "decoder.decoder.bias_hh": cell.bias_hh.detach(),
            }
        )
        identity = module.FrozenHeadIdentity("x", "h", "p", "c", "f", 0)
        heads = module.FrozenDeckHeads([state], [identity])
        x = torch.randn(3, width)
        hidden = torch.randn(3, width)
        actual = heads._gru(x, hidden, torch.zeros(3, dtype=torch.long))
        torch.testing.assert_close(actual, cell(x, hidden))


if __name__ == "__main__":
    unittest.main()
