"""Template entrypoint copied into each self-contained 0020 candidate."""
from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.inference import FrozenNeutralPolicy


DECK = [
    int(line)
    for line in (ROOT / "deck.csv").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
MANIFEST = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
POLICY = FrozenNeutralPolicy.from_checkpoint(
    ROOT / "strategy/model.bin",
    ROOT / "strategy/card_ontology.json",
    DECK,
    source_id=int(MANIFEST["source_id"]),
)


def read_deck_csv():
    return list(DECK)


def agent(observation):
    if observation.get("select") is None:
        POLICY.reset()
        return list(DECK)
    return POLICY.select(observation)
