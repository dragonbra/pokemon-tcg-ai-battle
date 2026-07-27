from pathlib import Path
import sys


def _submission_root():
    source_file = globals().get("__file__")
    if isinstance(source_file, str):
        root = Path(source_file).resolve().parent
        if (root / "deck.csv").is_file():
            return root
    kaggle_root = Path("/kaggle_simulations/agent")
    if (kaggle_root / "deck.csv").is_file():
        return kaggle_root
    return Path.cwd()


ROOT = _submission_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.r2_inference import R2Policy

DECK = [int(line) for line in (ROOT / "deck.csv").read_text(encoding="utf-8").splitlines() if line.strip()]
_POLICY = R2Policy.from_checkpoint(
    ROOT / "strategy" / "model.bin",
    ROOT / "strategy" / "card_ontology.json",
    DECK,
)


def read_deck_csv():
    return list(DECK)


def agent(observation):
    if observation.get("select") is None:
        _POLICY.reset()
        return list(DECK)
    return _POLICY.select(observation)
