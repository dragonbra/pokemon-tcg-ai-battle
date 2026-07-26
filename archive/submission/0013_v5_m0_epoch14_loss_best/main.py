from pathlib import Path
import sys


def _package_root() -> Path:
    source = globals().get("__file__")
    if isinstance(source, str) and (Path(source).resolve().parent / "deck.csv").is_file():
        return Path(source).resolve().parent
    kaggle = Path("/kaggle_simulations/agent")
    return kaggle if (kaggle / "deck.csv").is_file() else Path.cwd()


ROOT = _package_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.inference import SemanticGoalInference


DECK = [int(line) for line in (ROOT / "deck.csv").read_text().splitlines() if line.strip()]
_POLICY = SemanticGoalInference.from_checkpoint(
    ROOT / "strategy" / "model.bin",
    deck=DECK,
)


def read_deck_csv() -> list[int]:
    return list(DECK)


def agent(observation: dict) -> list[int]:
    if observation.get("select") is None:
        _POLICY.reset()
        return list(DECK)
    return _POLICY.select(observation)
