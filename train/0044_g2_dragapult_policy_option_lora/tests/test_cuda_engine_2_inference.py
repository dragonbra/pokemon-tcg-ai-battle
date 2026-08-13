from __future__ import annotations

import importlib
from pathlib import Path

import torch


arena_module = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.cuda_engine_2.arena")
inference = importlib.import_module("train.0044_g2_dragapult_policy_option_lora.cuda_engine_2.inference")
ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "train/0044_g2_dragapult_policy_option_lora"
RULES = ROOT / ".tmp/cuda_0032_rules/official_rules.bin"


def _deck(deck_id: str) -> list[int]:
    path = PROJECT / f"assets/decks/definitions/{deck_id}/deck.csv"
    return [int(row) for row in path.read_text().splitlines()]


def test_real_001_and_048_cuda_codec_reach_both_complete_policies() -> None:
    arena = arena_module.create_official_cuda_arena(ROOT, rule_pack=RULES, batch_size=2)
    decks = torch.tensor(
        [[_deck("001"), _deck("048")], [_deck("048"), _deck("001")]],
        dtype=torch.int32, device="cuda:0",
    )
    seeds = torch.tensor([430044001, 430044002], dtype=torch.int64, device="cuda:0")
    arena.engine.reset_seeded_first_min_semantic(decks, seeds)
    encoded = arena.engine.encode_semantic0031_v2_lanes(
        torch.arange(2, dtype=torch.int32, device="cuda:0")
    )
    anchor = inference.CudaPolicyCohort.load(
        PROJECT, policy_id="Policy-0809", deck_id="001"
    )
    champion = inference.CudaPolicyCohort.load(
        PROJECT, policy_id="Champion-G1", deck_id="048"
    )
    anchor_actions = anchor.greedy({name: value[:1] for name, value in encoded.items()})
    champion_actions = champion.greedy({name: value[1:] for name, value in encoded.items()})
    sequences = torch.cat((anchor_actions[0], champion_actions[0]), dim=0)
    lengths = torch.cat((anchor_actions[1], champion_actions[1]), dim=0)
    arena.engine.pack_actions(sequences, lengths)
    arena.engine.apply_packed_actions()
    torch.cuda.synchronize()
    assert anchor_actions[0].device.type == "cuda"
    assert champion_actions[0].device.type == "cuda"
    assert anchor_actions[1].shape == champion_actions[1].shape == (1,)
    assert arena.engine.statuses().cpu().tolist() == [1, 1]
    assert arena.engine.turns().cpu().tolist() == [1, 1]
