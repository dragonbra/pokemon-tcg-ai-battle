"""Generation 0 Residual LSTM policy with rule-only safety fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from cg.api import OptionType, to_observation_class
from src import engine
from src.encoding import encode
from src.net import A3CLSTMNet
from src.net.model import pad_decision
from src.agents.ver64 import Ver64Agent


WALL_IDS = {344, 345}
DRAGAPULT_IDS = {119, 120, 121}
CUBCHOO_CONTROL_IDS = {506}
LOW_DECK_THINNING_PLAY_IDS = {1102, 1142, 1152, 1197, 1213, 1227}


def load_generation0_policy(path: Union[str, Path], device: str = "cpu") -> A3CLSTMNet:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    net = A3CLSTMNet(
        hidden=int(ckpt.get("hidden", 384)),
        residual_context=bool(ckpt.get("residual_context", True)),
    ).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net


class Generation0PolicyAgent:
    def __init__(self, net: A3CLSTMNet, deck: List[int], fallback, device: str = "cpu"):
        self.net = net
        self.deck = deck
        self.fallback = fallback
        self.device = device
        self._adb = engine.attack_db()
        self.state = None
        self.last_decision_info: Dict[str, Any] = {}
        self.stats = {
            "total_decisions": 0,
            "generation0_policy_decisions": 0,
            "rule_fallback_decisions": 0,
            "first_legal_fallback_decisions": 0,
            "policy_exceptions": 0,
            "illegal_policy_actions": 0,
        }
        self.fallback_reasons: Dict[str, int] = {}

    def reset(self) -> None:
        self.state = self.net.initial_state(1, self.device)
        self.last_decision_info = {}
        if hasattr(self.fallback, "reset"):
            self.fallback.reset()

    def _record_fallback_reason(self, reason: str) -> None:
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1

    def _is_policy_supported(self, obs) -> Tuple[bool, str]:
        sel = obs.select
        if sel is None:
            return False, "deck_request"
        if not sel.option:
            return False, "empty_options"
        if sel.maxCount != 1 or sel.minCount > 1:
            return False, "multi_select"
        if len(sel.option) > 64:
            return False, "too_many_options"
        return True, "supported_single_select"

    def choose_policy(self, obs_dict: Dict[str, Any]) -> Optional[List[int]]:
        obs = to_observation_class(obs_dict)
        supported, reason = self._is_policy_supported(obs)
        if not supported:
            self._record_fallback_reason(reason)
            self.last_decision_info = {
                "decision_type": "rule_fallback",
                "fallback_used": True,
                "fallback_reason": reason,
            }
            return None

        if self.state is None:
            self.state = self.net.initial_state(1, self.device)

        try:
            encoded = encode(obs, attack_db=self._adb)
            glob, options, mask = pad_decision(encoded)
            with torch.no_grad():
                gt = torch.as_tensor(glob, dtype=torch.float32, device=self.device).unsqueeze(0)
                ot = torch.as_tensor(options, dtype=torch.float32, device=self.device).unsqueeze(0)
                mt = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
                logits_t, value_t, next_state = self.net.forward_step(gt, ot, mt, self.state)
                n = int(encoded.n_options)
                logits = logits_t[0, :n].detach().float().cpu().numpy()
                probs = F.softmax(logits_t[0, :n], dim=-1).detach().float().cpu().numpy()
            if n <= 0 or probs.size != n or not np.isfinite(probs).all() or probs.sum() <= 0:
                raise ValueError("nonfinite_or_empty_policy")
            action = int(np.argmax(probs))
            self.state = next_state
            self.stats["generation0_policy_decisions"] += 1
            self.last_decision_info = {
                "decision_type": "generation0_policy",
                "fallback_used": False,
                "fallback_reason": None,
                "policy_probability": float(probs[action]),
                "value_prediction": float(value_t.item()),
                "policy_top1_action": action,
                "policy_logits": [float(x) for x in logits],
            }
            return [action]
        except Exception as exc:
            self.stats["policy_exceptions"] += 1
            reason = f"policy_exception:{type(exc).__name__}"
            self._record_fallback_reason(reason)
            self.last_decision_info = {
                "decision_type": "rule_fallback",
                "fallback_used": True,
                "fallback_reason": reason,
            }
            return None

    def choose_rule_fallback(self, obs_dict: Dict[str, Any]) -> List[int]:
        self.stats["rule_fallback_decisions"] += 1
        return self.fallback(obs_dict)

    def __call__(self, obs_dict: Dict[str, Any]) -> List[int]:
        if isinstance(obs_dict, dict) and obs_dict.get("select") is None:
            self.reset()
            self.last_decision_info = {"decision_type": "deck", "fallback_used": False}
            return [int(card) for card in self.deck]
        self.stats["total_decisions"] += 1
        choice = self.choose_policy(obs_dict)
        if choice is not None:
            return choice
        return self.choose_rule_fallback(obs_dict)


class Generation0TacticalOverlayPolicyAgent(Generation0PolicyAgent):
    """Gen0 policy plus a narrow wall/Dragapult logit overlay.

    This keeps ordinary single-select decisions on the original neural policy.
    Only when the opponent has publicly revealed a Crustle wall line or a
    Dragapult line do we let the Ver64 tactical scorer add a small fixed logit
    bonus to its top action, and only when the scorer margin is clear.
    """

    def __init__(
        self,
        net: A3CLSTMNet,
        deck: List[int],
        fallback,
        device: str = "cpu",
        overlay_bonus: float = 1.25,
        min_rule_margin: float = 1000.0,
    ):
        super().__init__(net, deck, fallback, device=device)
        self.tactical = Ver64Agent(deck=deck)
        self.overlay_bonus = float(overlay_bonus)
        self.min_rule_margin = float(min_rule_margin)
        self.stats["tactical_overlay_decisions"] = 0
        self.stats["wall_signal_decisions"] = 0
        self.stats["dragapult_signal_decisions"] = 0
        self.stats["cubchoo_signal_decisions"] = 0
        self.stats["cubchoo_guard_decisions"] = 0

    def reset(self) -> None:
        super().reset()
        if hasattr(self.tactical, "reset"):
            self.tactical.reset()

    def _opponent_has_any(self, obs, ids: set[int]) -> bool:
        state = getattr(obs, "current", None)
        if state is None:
            return False
        opp = state.players[1 - state.yourIndex]
        zones = (
            list(getattr(opp, "active", []) or []),
            list(getattr(opp, "bench", []) or []),
            list(getattr(opp, "discard", []) or []),
        )
        return any(card is not None and getattr(card, "id", None) in ids for zone in zones for card in zone)

    def _signal(self, obs) -> str | None:
        if self._opponent_has_any(obs, WALL_IDS):
            return "wall"
        if self._opponent_has_any(obs, DRAGAPULT_IDS):
            return "dragapult"
        if self._opponent_has_any(obs, CUBCHOO_CONTROL_IDS):
            return "cubchoo_control"
        return None

    def _hand_card_id_for_option(self, obs, op) -> int | None:
        try:
            idx = getattr(op, "index", None)
            if idx is None:
                return None
            hand = obs.current.players[obs.current.yourIndex].hand or []
            if idx < 0 or idx >= len(hand):
                return None
            return int(getattr(hand[idx], "id", -1))
        except Exception:
            return None

    def _apply_cubchoo_deckout_guard(self, obs, logits: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        state = getattr(obs, "current", None)
        sel = getattr(obs, "select", None)
        if state is None or sel is None or logits.size <= 1:
            return logits, {"route": "cubchoo_control", "overlay_used": False}
        deck_count = int(getattr(state.players[state.yourIndex], "deckCount", 99))
        if deck_count > 12:
            return logits, {
                "route": "cubchoo_control",
                "overlay_used": False,
                "cubchoo_guard_checked": True,
                "deck_count": deck_count,
                "cubchoo_guard_used": False,
            }

        penalty = 10.0 if deck_count <= 8 else 5.0
        adjusted = logits.copy()
        penalized = 0
        n = min(len(sel.option), adjusted.size)
        for idx in range(n):
            op = sel.option[idx]
            op_type = getattr(op, "type", None)
            if op_type == OptionType.ABILITY:
                adjusted[idx] -= penalty
                penalized += 1
                continue
            if op_type == OptionType.PLAY and self._hand_card_id_for_option(obs, op) in LOW_DECK_THINNING_PLAY_IDS:
                adjusted[idx] -= penalty
                penalized += 1
        if penalized:
            self.stats["cubchoo_guard_decisions"] += 1
        return adjusted, {
            "route": "cubchoo_control",
            "overlay_used": False,
            "cubchoo_guard_checked": True,
            "cubchoo_guard_used": penalized > 0,
            "cubchoo_guard_penalized": penalized,
            "cubchoo_guard_penalty": penalty,
            "deck_count": deck_count,
        }

    def _adjust_logits(self, obs, logits: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        signal = self._signal(obs)
        if signal is None or logits.size <= 1:
            return logits, {"route": "base", "overlay_used": False}
        if signal == "wall":
            self.stats["wall_signal_decisions"] += 1
        elif signal == "dragapult":
            self.stats["dragapult_signal_decisions"] += 1
        elif signal == "cubchoo_control":
            self.stats["cubchoo_signal_decisions"] += 1
            return self._apply_cubchoo_deckout_guard(obs, logits)

        try:
            scores = np.asarray(self.tactical.scores_for(obs), dtype=np.float32)
        except Exception as exc:
            self._record_fallback_reason(f"overlay_score_exception:{type(exc).__name__}")
            return logits, {"route": signal, "overlay_used": False}
        n = min(int(logits.size), int(scores.size))
        if n <= 1 or not np.isfinite(scores[:n]).all():
            return logits, {"route": signal, "overlay_used": False}
        order = np.argsort(scores[:n])[::-1]
        teacher = int(order[0])
        base = int(np.argmax(logits[:n]))
        second = float(scores[order[1]])
        margin = float(scores[teacher] - second)
        if teacher == base or margin < self.min_rule_margin:
            return logits, {
                "route": signal,
                "overlay_used": False,
                "teacher_action": teacher,
                "base_action": base,
                "rule_margin": margin,
            }
        adjusted = logits.copy()
        adjusted[teacher] += self.overlay_bonus
        self.stats["tactical_overlay_decisions"] += 1
        return adjusted, {
            "route": signal,
            "overlay_used": True,
            "teacher_action": teacher,
            "base_action": base,
            "rule_margin": margin,
        }

    def choose_policy(self, obs_dict: Dict[str, Any]) -> Optional[List[int]]:
        obs = to_observation_class(obs_dict)
        supported, reason = self._is_policy_supported(obs)
        if not supported:
            self._record_fallback_reason(reason)
            self.last_decision_info = {
                "decision_type": "rule_fallback",
                "fallback_used": True,
                "fallback_reason": reason,
            }
            return None

        if self.state is None:
            self.state = self.net.initial_state(1, self.device)

        try:
            encoded = encode(obs, attack_db=self._adb)
            glob, options, mask = pad_decision(encoded)
            with torch.no_grad():
                gt = torch.as_tensor(glob, dtype=torch.float32, device=self.device).unsqueeze(0)
                ot = torch.as_tensor(options, dtype=torch.float32, device=self.device).unsqueeze(0)
                mt = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
                logits_t, value_t, next_state = self.net.forward_step(gt, ot, mt, self.state)
                n = int(encoded.n_options)
                logits = logits_t[0, :n].detach().float().cpu().numpy()
            if n <= 0 or logits.size != n or not np.isfinite(logits).all():
                raise ValueError("nonfinite_or_empty_policy")
            adjusted, overlay_info = self._adjust_logits(obs, logits)
            probs = F.softmax(torch.as_tensor(adjusted, dtype=torch.float32), dim=-1).numpy()
            if probs.size != n or not np.isfinite(probs).all() or probs.sum() <= 0:
                raise ValueError("nonfinite_or_empty_adjusted_policy")
            action = int(np.argmax(probs))
            self.state = next_state
            self.stats["generation0_policy_decisions"] += 1
            self.last_decision_info = {
                "decision_type": "tactical_overlay_policy" if overlay_info.get("overlay_used") else "generation0_policy",
                "fallback_used": False,
                "fallback_reason": None,
                "policy_probability": float(probs[action]),
                "value_prediction": float(value_t.item()),
                "policy_top1_action": action,
                "policy_logits": [float(x) for x in logits],
                "adjusted_policy_logits": [float(x) for x in adjusted],
                **overlay_info,
            }
            return [action]
        except Exception as exc:
            self.stats["policy_exceptions"] += 1
            reason = f"policy_exception:{type(exc).__name__}"
            self._record_fallback_reason(reason)
            self.last_decision_info = {
                "decision_type": "rule_fallback",
                "fallback_used": True,
                "fallback_reason": reason,
            }
            return None
