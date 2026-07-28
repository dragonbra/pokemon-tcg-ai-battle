"""Greedy CPU inference for the Marnie multi-deck mean-pool BC checkpoint."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


# NumPy and PyTorch can load distinct OpenMP runtimes on this Windows setup.
# This must be set before the training module imports either library.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

def _agent_directory() -> Path:
    """Resolve the bundle path when Kaggle executes main.py without __file__."""
    try:
        return Path(__file__).resolve().parent
    except (NameError, OSError, TypeError, ValueError):
        try:
            source_path = Path(sys._getframe().f_code.co_filename).resolve()
            if (source_path.parent / "deck.csv").is_file():
                return source_path.parent
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
MODEL_PATH = ROOT / "policy.pt"
TRAIN_MODULE_RELATIVE = Path("notebooks/rl_pipeline/ptcg_yushin_idonly_bc_train_v1/ptcg_yushin_idonly_bc_train_v1.py")

_MODEL: Any | None = None
_CODEC: Any | None = None
_TORCH: Any | None = None


def _read_deck_csv() -> list[int]:
    return [int(line) for line in (ROOT / "deck.csv").read_text(encoding="utf-8").splitlines() if line.strip()]


my_deck = _read_deck_csv()


def _find_train_module() -> Path:
    bundled = ROOT / "idonly_policy.py"
    if bundled.is_file():
        return bundled
    for parent in (ROOT, *ROOT.parents):
        candidate = parent / TRAIN_MODULE_RELATIVE
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot locate {TRAIN_MODULE_RELATIVE}")


def _load_train_module() -> Any:
    source = _find_train_module()
    module_name = "_idonly_bc_train_" + hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12]
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load model source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _ensure_model() -> tuple[Any, Any, Any]:
    global _MODEL, _CODEC, _TORCH
    if _MODEL is not None and _CODEC is not None and _TORCH is not None:
        return _MODEL, _CODEC, _TORCH

    train = _load_train_module()
    torch = train.torch
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # The interpreter may already have set its inter-op thread pool.
        pass

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    config = train.ModelConfig(**checkpoint["model_config"])
    model = train.IDOnlyPointerPolicy(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    _MODEL, _CODEC, _TORCH = model, train.IDOnlyCodec(config), torch
    return _MODEL, _CODEC, _TORCH


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _decode(obs: dict[str, Any]) -> list[int]:
    model, codec, torch = _ensure_model()
    select = obs.get("select") if isinstance(obs, dict) else None
    options = select.get("option") if isinstance(select, dict) else None
    if not isinstance(options, list) or not options or len(options) > model.config.max_options:
        return []

    option_count = len(options)
    min_count = max(0, _as_int(select.get("minCount")))
    max_count = max(min_count, _as_int(select.get("maxCount"), option_count))
    if min_count > option_count or min_count > model.config.max_action_steps:
        return []
    max_count = min(max_count, option_count, model.config.max_action_steps)
    row = codec.encode(obs, list(range(min_count)))
    if row is None:
        return []

    counts = Counter(int(card) for card in my_deck)
    row.update(
        deck_ids=[int(card) for card in my_deck],
        deck_counts=[counts[int(card)] for card in my_deck],
        sample_weight=1.0,
        deck_hash_code=0,
    )

    batch = _load_train_module().collate_examples([row])
    with torch.inference_mode():
        state, option_values = model.encode(batch)
        keys = model.pointer_key(option_values)
        hidden = torch.tanh(model.decoder_init(state))
        available = batch["option_mask"].clone()
        chosen = torch.zeros((1, option_count), dtype=torch.bool)
        action: list[int] = []

        for step in range(max_count):
            pointer = (model.pointer_query(hidden).unsqueeze(1) * keys).sum(-1) / (model.config.d_model ** 0.5)
            pointer = pointer + model.option_bias(option_values).squeeze(-1)
            pointer = pointer.masked_fill(~available | chosen, torch.finfo(pointer.dtype).min)
            stop = model.stop(hidden)
            if step < min_count:
                stop = stop.masked_fill(torch.ones_like(stop, dtype=torch.bool), torch.finfo(stop.dtype).min)
            choice = int(torch.argmax(torch.cat([pointer, stop], dim=1), dim=1).item())
            if choice == option_count:
                break
            action.append(choice)
            chosen[0, choice] = True
            hidden = model.decoder(option_values[:, choice], hidden)

    return action


def agent(obs: dict[str, Any]) -> list[int]:
    """Return a legal option-index sequence; the arena normalizer handles rare fallbacks."""
    # Kaggle calls the agent once before a game state exists. This call is the
    # deck-registration handshake, so it must not initialize the ML runtime.
    if not isinstance(obs, dict) or not isinstance(obs.get("current"), dict) or not isinstance(obs.get("select"), dict):
        return list(my_deck)
    try:
        return _decode(obs)
    except Exception:
        return []
