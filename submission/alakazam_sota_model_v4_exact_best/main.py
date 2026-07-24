from pathlib import Path
import sys


def _submission_root():
    source_file = globals().get("__file__")
    if isinstance(source_file, str):
        source_root = Path(source_file).resolve().parent
        if (source_root / "deck.csv").is_file():
            return source_root
    kaggle_root = Path("/kaggle_simulations/agent")
    if (kaggle_root / "deck.csv").is_file():
        return kaggle_root
    return Path.cwd()


ROOT = _submission_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.inference import IDOnlyPolicy


DECK = [
    int(line)
    for line in (ROOT / "deck.csv").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
_POLICY = IDOnlyPolicy.from_checkpoint(ROOT / "strategy" / "model.bin")


def read_deck_csv():
    return list(DECK)


def agent(observation):
    if observation.get("select") is None:
        _POLICY.reset()
        return list(DECK)
    return _POLICY.select(observation)
