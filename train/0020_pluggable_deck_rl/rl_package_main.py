"""Template entrypoint copied into each self-contained 0020 PPO candidate."""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.portable_inference import PortablePolicy


DECK = [
    int(line)
    for line in (ROOT / "deck.csv").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
POLICY = PortablePolicy.from_checkpoint(
    ROOT / "strategy/model.bin",
    ROOT / "strategy/card_ontology.json",
    DECK,
)


def read_deck_csv():
    return list(DECK)


def agent(observation):
    if observation.get("select") is None:
        POLICY.reset()
        return list(DECK)
    return POLICY.select(observation)
