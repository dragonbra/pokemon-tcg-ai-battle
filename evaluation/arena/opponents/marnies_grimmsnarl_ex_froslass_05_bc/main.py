from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _agent_directory() -> Path:
    try:
        source_dir = Path(__file__).resolve().parent
        if (source_dir / "deck.csv").is_file():
            return source_dir
    except (NameError, OSError, TypeError, ValueError):
        pass
    try:
        source_dir = Path(sys._getframe().f_code.co_filename).resolve().parent
        if (source_dir / "deck.csv").is_file():
            return source_dir
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    cwd = Path.cwd().resolve()
    kaggle_dir = Path("/kaggle_simulations/agent")
    if (cwd / "deck.csv").is_file():
        return cwd
    if (kaggle_dir / "deck.csv").is_file():
        return kaggle_dir
    return cwd


ROOT = _agent_directory()
my_deck = [int(line) for line in (ROOT / "deck.csv").read_text(encoding="utf-8").splitlines() if line.strip()]
_RUNTIME: Any | None = None
_FALLBACK_COUNT = 0


def _runtime() -> Any:
    global _RUNTIME
    if _RUNTIME is None:
        source = ROOT / "idonly_policy.py"
        module_name = "_marnie_probability_ensemble_" + ROOT.name
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load model source: {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            config = __import__("json").loads((ROOT / "ensemble.json").read_text(encoding="utf-8"))
            _RUNTIME = module.MarnieIDOnlyProbabilityEnsembleRuntime(
                ROOT / "policy_left.pt",
                ROOT / "policy_right.pt",
                my_deck,
                float(config["right_weight"]),
            )
        except Exception:
            sys.modules.pop(module_name, None)
            raise
    return _RUNTIME


def agent(obs: dict[str, Any]) -> list[int]:
    global _FALLBACK_COUNT
    if not isinstance(obs, dict) or not isinstance(obs.get("current"), dict) or not isinstance(obs.get("select"), dict):
        return list(my_deck)
    try:
        return _runtime().decode(obs)
    except Exception:
        _FALLBACK_COUNT += 1
        return []


def fallback_count() -> int:
    return _FALLBACK_COUNT
