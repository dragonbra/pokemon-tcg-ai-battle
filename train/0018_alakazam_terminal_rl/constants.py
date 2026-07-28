from __future__ import annotations

import hashlib
from pathlib import Path


PROJECT_ID = "0018_alakazam_terminal_rl"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CHECKPOINT = (
    REPOSITORY_ROOT
    / "rl_runs/0016_alakazam_multideck_bc/versions/"
    "V2_win_multideck_r15_gradual_option/checkpoint/epoch-0010-9db3b4d37a4d4f5c.pt"
)
SOURCE_CHECKPOINT_SHA256 = (
    "9db3b4d37a4d4f5cc483df1ac78ad4c5d39fed320c5325ebaa5fa61fd1114ebf"
)
ONTOLOGY_PATH = Path(__file__).resolve().parent / "assets/card_ontology.json"
TARGET_SOURCE_ID = 0
TARGET_DECK_PATH = Path(__file__).resolve().parent / "deck.csv"
TARGET_DECK_SHA256 = "267ce842b45f960afef843e68f3aad86573bf7d19329ea23e7469ec8265c017b"
TARGET_DECK = tuple(
    int(line)
    for line in TARGET_DECK_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
)

if len(TARGET_DECK) != 60:
    raise RuntimeError("the frozen 0016 Alakazam target deck must contain 60 cards")
if hashlib.sha256(TARGET_DECK_PATH.read_bytes()).hexdigest() != TARGET_DECK_SHA256:
    raise RuntimeError("the frozen 0016 Alakazam target deck hash does not match")
