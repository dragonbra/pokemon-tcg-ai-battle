from __future__ import annotations

import importlib
from pathlib import Path

import pytest


policy = importlib.import_module(
    "train.0044_g2_dragapult_policy_option_lora.policy.actor_critic"
)
assets = importlib.import_module(
    "train.0044_g2_dragapult_policy_option_lora.assets"
)


ROOT = Path(__file__).resolve().parents[3]
SP08 = ROOT / "docs/reports/sp-series/decks/SP08_MAGA/deck.csv"


def test_external_focal_requires_and_honors_explicit_own_archetype() -> None:
    cards = tuple(map(int, SP08.read_text(encoding="utf-8").splitlines()))
    with pytest.raises(assets.AssetIntegrityError):
        policy.load_actor_critic(deck=cards, deck_id="SP08_MAGA")
    model, _ = policy.load_actor_critic(
        deck=cards, deck_id="SP08_MAGA", own_archetype_id_override=3,
    )
    assert model.default_own_archetype_id.item() == 3
