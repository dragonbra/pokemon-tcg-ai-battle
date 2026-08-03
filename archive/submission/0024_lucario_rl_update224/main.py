"""Template entrypoint copied into each self-contained 0020 candidate."""
import importlib
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DECK = [
    int(line)
    for line in (ROOT / "deck.csv").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
_RUNTIME_PACKAGE = "_ptcg_zero_shot_marnie_runtime"
_POLICY: Any | None = None


def _runtime_module():
    runtime = sys.modules.get(_RUNTIME_PACKAGE)
    if runtime is not None:
        return runtime

    runtime_root = ROOT / "strategy"
    spec = importlib.util.spec_from_file_location(
        _RUNTIME_PACKAGE,
        runtime_root / "__init__.py",
        submodule_search_locations=[str(runtime_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create isolated runtime package from {runtime_root}")
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[_RUNTIME_PACKAGE] = runtime
    try:
        spec.loader.exec_module(runtime)
    except BaseException:
        sys.modules.pop(_RUNTIME_PACKAGE, None)
        raise
    return runtime


def _policy() -> Any:
    global _POLICY
    if _POLICY is None:
        _runtime_module()
        inference = importlib.import_module(f"{_RUNTIME_PACKAGE}.inference")

        _POLICY = inference.FrozenNeutralPolicy.from_checkpoint(
            ROOT / "strategy/model.bin",
            ROOT / "strategy/card_ontology.json",
            DECK,
        )
    return _POLICY


POLICY = _policy()


def read_deck_csv():
    return list(DECK)


def agent(observation):
    if not isinstance(observation, dict) or observation.get("select") is None:
        if _POLICY is not None:
            _POLICY.reset()
        return list(DECK)
    return _policy().select(observation)
