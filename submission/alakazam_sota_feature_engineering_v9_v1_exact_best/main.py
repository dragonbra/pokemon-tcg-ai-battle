from pathlib import Path
import sys

def _root():
    source = globals().get("__file__")
    if isinstance(source, str) and (Path(source).resolve().parent / "deck.csv").is_file():
        return Path(source).resolve().parent
    kaggle = Path("/kaggle_simulations/agent")
    return kaggle if (kaggle / "deck.csv").is_file() else Path.cwd()

ROOT = _root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from strategy.inference import FeatureEngineeringInference

DECK = [int(line) for line in (ROOT / "deck.csv").read_text().splitlines() if line.strip()]
_POLICY = FeatureEngineeringInference.from_checkpoint(ROOT / "strategy" / "model.bin")

def read_deck_csv():
    return list(DECK)

def agent(observation):
    if observation.get("select") is None:
        _POLICY.reset()
        return list(DECK)
    return _POLICY.select(observation)
