"""Lucario Gen0 Residual LSTM Policy with narrow tactical overlay."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def _find_agent_dir() -> Path:
    candidates: list[Path] = []
    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    candidates += [Path("/kaggle_simulations/agent"), Path(os.getcwd())]
    for candidate in candidates:
        if candidate and (candidate / "cg").exists():
            return candidate
    return Path("/kaggle_simulations/agent")


def _raw_legal_fallback(obs_dict: dict[str, Any]) -> list[int]:
    select = obs_dict.get("select") if isinstance(obs_dict, dict) else None
    options = (select or {}).get("option") or []
    min_count = int((select or {}).get("minCount") or 0)
    max_count = int((select or {}).get("maxCount") or 0)
    n = len(options)
    if n <= 0 or max_count <= 0:
        return []
    k = min(max(min_count, 1), max_count, n)
    return list(range(k))


def _is_legal(choice: Any, obs_dict: dict[str, Any]) -> bool:
    if not isinstance(choice, list) or not all(isinstance(x, int) for x in choice):
        return False
    select = obs_dict.get("select") if isinstance(obs_dict, dict) else None
    options = (select or {}).get("option") or []
    min_count = int((select or {}).get("minCount") or 0)
    max_count = int((select or {}).get("maxCount") or 0)
    n = len(options)
    return (
        min_count <= len(choice) <= max_count
        and len(set(choice)) == len(choice)
        and all(0 <= x < n for x in choice)
    )


AGENT_DIR = _find_agent_dir()
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from src import engine  # noqa: E402
from src.agents.ver58 import Ver58Agent  # noqa: E402


_DECK = engine.read_deck_csv(str(AGENT_DIR / "deck.csv"))
if len(_DECK) != 60:
    raise ValueError(f"submission deck must contain 60 cards, got {len(_DECK)}")

_MODEL_PATH = AGENT_DIR / "policy.pt"
if not _MODEL_PATH.exists():
    _MODEL_PATH = AGENT_DIR / "model" / "policy.pt"
_RULE58 = Ver58Agent(deck=_DECK)
_FALLBACK_COUNT = 0
_FIRST_LEGAL_COUNT = 0
_STARTUP_ERROR = ""
_POLICY_AGENT = None
_POLICY_LOAD_STARTED = False
_POLICY_LOAD_DONE = False
_POLICY_LOAD_ERROR = ""
_POLICY_LOADING_FALLBACK_COUNT = 0


class _RuleFallback:
    def reset(self) -> None:
        if hasattr(_RULE58, "reset"):
            _RULE58.reset()

    def __call__(self, obs_dict: dict[str, Any]) -> list[int]:
        global _FALLBACK_COUNT, _FIRST_LEGAL_COUNT
        try:
            choice = _RULE58(obs_dict)
            if _is_legal(choice, obs_dict):
                _FALLBACK_COUNT += 1
                return choice
        except Exception:
            pass
        _FIRST_LEGAL_COUNT += 1
        return _raw_legal_fallback(obs_dict)


_RULE_FALLBACK = _RuleFallback()
_MODE = "primary=gen0_lucario_tactical_overlay_lazy fallback=ver58_rule_limited"

sys.stderr.write(f"[gen0 lucario tactical overlay submission] {_MODE}\n")


def _load_policy_once() -> None:
    global _POLICY_AGENT, _POLICY_LOAD_DONE, _POLICY_LOAD_ERROR
    global _POLICY_LOAD_STARTED
    if _POLICY_LOAD_STARTED:
        return
    _POLICY_LOAD_STARTED = True
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        import torch

        torch.set_num_threads(1)
        from src.agents.gen0_policy import Generation0TacticalOverlayPolicyAgent, load_generation0_policy

        net = load_generation0_policy(_MODEL_PATH, device="cpu")
        agent_obj = Generation0TacticalOverlayPolicyAgent(
            net,
            deck=_DECK,
            fallback=_RULE_FALLBACK,
            device="cpu",
            overlay_bonus=1.25,
            min_rule_margin=1000.0,
        )
        agent_obj.reset()
        _POLICY_AGENT = agent_obj
        _POLICY_LOAD_DONE = True
        _POLICY_LOAD_ERROR = ""
    except Exception as exc:  # pragma: no cover
        _POLICY_LOAD_DONE = True
        _POLICY_LOAD_ERROR = f"{type(exc).__name__}: {exc}"


def _get_policy_agent():
    if _POLICY_AGENT is None and not _POLICY_LOAD_STARTED:
        _load_policy_once()
    return _POLICY_AGENT


def submission_stats() -> dict[str, Any]:
    policy = _POLICY_AGENT
    if policy is None:
        total = max(1, _FALLBACK_COUNT + _FIRST_LEGAL_COUNT + _POLICY_LOADING_FALLBACK_COUNT)
        return {
            "total_decisions": _FALLBACK_COUNT + _FIRST_LEGAL_COUNT + _POLICY_LOADING_FALLBACK_COUNT,
            "generation0_policy_decisions": 0,
            "rule_fallback_decisions": _FALLBACK_COUNT,
            "first_legal_fallback_decisions": _FIRST_LEGAL_COUNT,
            "policy_exceptions": 1 if _STARTUP_ERROR else 0,
            "illegal_policy_actions": 0,
            "fallback_reasons": {"policy_loading": _POLICY_LOADING_FALLBACK_COUNT},
            "startup_error": _STARTUP_ERROR or _POLICY_LOAD_ERROR,
            "policy_load_started": _POLICY_LOAD_STARTED,
            "policy_load_done": _POLICY_LOAD_DONE,
            "policy_usage_rate": 0.0,
            "rule_fallback_rate": float(_FALLBACK_COUNT) / total,
        }
    stats = dict(policy.stats)
    stats["rule_fallback_decisions"] = int(stats.get("rule_fallback_decisions", 0))
    stats["first_legal_fallback_decisions"] = int(_FIRST_LEGAL_COUNT)
    stats["fallback_reasons"] = dict(policy.fallback_reasons)
    if _POLICY_LOADING_FALLBACK_COUNT:
        stats["fallback_reasons"]["policy_loading"] = _POLICY_LOADING_FALLBACK_COUNT
    stats["policy_load_started"] = _POLICY_LOAD_STARTED
    stats["policy_load_done"] = _POLICY_LOAD_DONE
    stats["policy_load_error"] = _POLICY_LOAD_ERROR
    total = max(1, int(stats.get("total_decisions", 0)))
    stats["policy_usage_rate"] = float(stats.get("generation0_policy_decisions", 0)) / total
    stats["rule_fallback_rate"] = float(stats.get("rule_fallback_decisions", 0)) / total
    return stats


def agent(obs_dict: dict[str, Any]) -> list[int]:
    global _FIRST_LEGAL_COUNT, _POLICY_LOADING_FALLBACK_COUNT
    if isinstance(obs_dict, dict) and obs_dict.get("select") is None:
        sys.stderr.write(f"[gen0 lucario tactical overlay submission] initial_deck_cards={len(_DECK)}\n")
        policy = _POLICY_AGENT
        if policy is not None:
            return policy(obs_dict)
        _RULE_FALLBACK.reset()
        return [int(card) for card in _DECK]

    policy = _get_policy_agent()
    if policy is None:
        _POLICY_LOADING_FALLBACK_COUNT += 1
        choice = _RULE_FALLBACK(obs_dict)
        if _is_legal(choice, obs_dict):
            return choice
        _FIRST_LEGAL_COUNT += 1
        return _raw_legal_fallback(obs_dict)

    try:
        choice = policy(obs_dict)
        if _is_legal(choice, obs_dict):
            return choice
        policy.stats["illegal_policy_actions"] += 1
    except Exception as exc:
        policy.stats["policy_exceptions"] += 1
        policy._record_fallback_reason(f"agent_exception:{type(exc).__name__}")

    _FIRST_LEGAL_COUNT += 1
    return _raw_legal_fallback(obs_dict)
