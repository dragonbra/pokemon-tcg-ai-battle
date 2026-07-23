from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE_AGENT = ROOT.parent / "public_great_tusk_lo_koushikrudra"
if str(BASE_AGENT) not in sys.path:
    sys.path.insert(0, str(BASE_AGENT))

_MODULE_NAME = "_weighted_meta_base_" + hashlib.sha1((str(BASE_AGENT) + str(ROOT)).encode("utf-8")).hexdigest()[:12]
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, BASE_AGENT / "main.py")
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load base agent: {BASE_AGENT}")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)


def _read_deck_csv() -> list[int]:
    return [int(line) for line in (ROOT / "deck.csv").read_text(encoding="utf-8").splitlines() if line.strip()]


def agent(obs_dict):
    if not isinstance(obs_dict, dict) or obs_dict.get("select") is None:
        return _read_deck_csv()
    return _BASE.agent(obs_dict)
