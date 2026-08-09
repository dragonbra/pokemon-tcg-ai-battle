"""Kaggle-compatible entrypoint for the self-contained 0038 compound policy."""

import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")

ROOT = Path(globals().get("__file__", Path.cwd())).resolve()
if ROOT.is_file():
    ROOT = ROOT.parent
if not (ROOT / "deck.csv").is_file() and Path("/kaggle_simulations/agent/deck.csv").is_file():
    ROOT = Path("/kaggle_simulations/agent")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.deployment.compound_inference import PortableCompoundSemanticPolicy

DECK = [int(line) for line in (ROOT / "deck.csv").read_text().splitlines() if line.strip()]
POLICY = PortableCompoundSemanticPolicy.from_checkpoint(
    ROOT / "strategy/model.bin", DECK
)


def read_deck_csv():
    return list(DECK)


def agent(observation):
    if observation.get("select") is None:
        POLICY.reset()
        return list(DECK)
    return POLICY.select(observation)
