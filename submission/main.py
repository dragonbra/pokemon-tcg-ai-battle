from __future__ import annotations

from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def read_deck_csv() -> list[int]:
    """Read the 60-card deck next to this entry point.

    Kaggle runs ``main.py`` from the submission directory, while local tools
    may use another working directory. Resolving the file relative to this
    module keeps both execution modes identical.
    """
    candidates = [
        ROOT / "deck.csv",
        Path("/kaggle_simulations/agent/deck.csv"),
        Path("deck.csv"),
    ]
    for file_path in candidates:
        if file_path.exists():
            values = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines()]
            deck = [int(value) for value in values if value]
            if len(deck) != 60:
                raise ValueError(f"deck.csv must contain 60 cards, got {len(deck)}")
            return deck
    raise FileNotFoundError("deck.csv was not found next to main.py or in /kaggle_simulations/agent")


def _option_priority(option: dict[str, Any], index: int) -> tuple[int, int]:
    """Prefer useful deterministic actions while retaining simulator legality.

    The baseline intentionally does not infer hidden information. It only
    uses fields supplied in the current selection and falls back to the first
    legal option when a new simulator option is encountered.
    """
    option_type = option.get("type")
    priorities = {
        13: 0,  # ATTACK
        8: 1,   # ATTACH
        9: 2,   # EVOLVE
        7: 3,   # PLAY
        10: 4,  # ABILITY
        12: 5,  # RETREAT
        14: 99, # END
    }
    return priorities.get(option_type, 10), index


def _select_indices(select: dict[str, Any]) -> list[int]:
    options = select.get("option") or []
    min_count = int(select.get("minCount", 0))
    max_count = int(select.get("maxCount", len(options)))
    if not options:
        if min_count:
            raise ValueError("simulator returned a required selection with no options")
        return []

    # Optional effect selections can safely be declined. This avoids spending
    # resources on an effect before the baseline has a card-specific policy.
    if min_count == 0 and select.get("type") != 0:
        return []

    count = min_count if min_count > 0 else 1
    count = min(count, max_count, len(options))
    ranked = sorted(enumerate(options), key=lambda item: _option_priority(item[1], item[0]))
    return [index for index, _ in ranked[:count]]

def agent(obs_dict: dict) -> list[int]:
    """Implement Your Pokémon Trading Card Game Agent.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.
    
    Returns:
        list[int]: A list of option index.
    """
    # At the initial evaluator callback there is no selection object; the
    # returned list is interpreted as the submitted deck.
    if obs_dict.get("select") is None:
        return read_deck_csv()

    return _select_indices(obs_dict["select"])
