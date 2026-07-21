from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(globals().get("__file__", "/kaggle_simulations/agent/main.py")).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.cards import *  # noqa: E402,F401,F403
from strategy.cards import load_deck  # noqa: E402
from strategy.orchestrator import StrategyOrchestrator  # noqa: E402
from strategy.profiles import BASELINE_PROFILE  # noqa: E402


DECK = load_deck(ROOT)
_ORCHESTRATOR = StrategyOrchestrator(DECK, BASELINE_PROFILE)


def read_deck_csv() -> list[int]:
    """供仓库本地 battle runner 使用；Kaggle 直接读取 ``DECK``。"""
    return list(DECK)


def agent(obs_dict: dict[str, Any]) -> list[int]:
    if obs_dict.get("select") is None:
        _ORCHESTRATOR.reset()
        return list(DECK)
    return list(_ORCHESTRATOR.choose(obs_dict).option_indexes)
