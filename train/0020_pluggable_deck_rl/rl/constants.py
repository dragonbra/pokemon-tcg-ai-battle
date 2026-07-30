from __future__ import annotations

import hashlib
from pathlib import Path


PROJECT_ID = "0020_pluggable_deck_rl"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_CHECKPOINT = (
    REPOSITORY_ROOT
    / "rl_runs/0020_pluggable_deck_rl/versions/"
    "V1_frozen_0019_epoch13/checkpoint/epoch-0013-da9b13d6f82d19d4.pt"
)
SOURCE_CHECKPOINT_SHA256 = (
    "da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb"
)
ONTOLOGY_PATH = Path(__file__).resolve().parent / "assets/card_ontology.json"
TARGET_SOURCE_ID = 0
TARGET_DECK_PATH = Path(__file__).resolve().parent / "deck.csv"
TARGET_DECK_SHA256 = "00468e64b7c5eefb1cc4586ba9351a7f2a5bce223ef209d2b9e4b0bb0ca42e17"
TARGET_DECK = tuple(
    int(line)
    for line in TARGET_DECK_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
)

if len(TARGET_DECK) != 60:
    raise RuntimeError("the frozen 0020 Dragapult target deck must contain 60 cards")
if hashlib.sha256(TARGET_DECK_PATH.read_bytes()).hexdigest() != TARGET_DECK_SHA256:
    raise RuntimeError("the frozen 0020 Dragapult target deck hash does not match")
