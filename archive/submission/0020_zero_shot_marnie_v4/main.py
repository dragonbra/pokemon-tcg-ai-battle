"""Kaggle-compatible entrypoint copied into each self-contained 0020 candidate."""
from pathlib import Path
import json
import os
import sys


# Kaggle's agent runner may exec this file without defining ``__file__``.
def _agent_directory() -> Path:
    try:
        source_path = Path(__file__).resolve()
        if (source_path.parent / "deck.csv").is_file():
            return source_path.parent
    except (NameError, OSError, TypeError, ValueError):
        pass
    cwd = Path.cwd().resolve()
    kaggle_dir = Path("/kaggle_simulations/agent")
    if (cwd / "deck.csv").is_file():
        return cwd
    if (kaggle_dir / "deck.csv").is_file():
        return kaggle_dir
    return cwd


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = _agent_directory()
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
