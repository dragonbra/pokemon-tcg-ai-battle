"""Kaggle-safe dedicated Iwapalace/Crustle RL submission agent.

This file is intentionally self-contained for submission use.
It combines a deterministic deck-specific heuristic with an optional
Tabular Q-learning model loaded from q_model_submit.json.

Deck concept:
- Dwebble / Crustle (Iwapalace) setup and tank plan.
- Hero's Mantle + Grow Grass Energy + Mist Energy for durability.
- Munkidori with Dark Energy for damage-counter pressure.
- Boss's Orders / Great Scissor to close prizes.

No Shaymin package is used in this dedicated build. The extra slot is converted
into a third PokePad so the policy stays focused on Crustle + Munkidori.

Public entry points supported:
    agent(state, legal_actions)
    agent.select_action(state, legal_actions)
    agent.act(state, legal_actions)
    choose_action(state, legal_actions)

The selected object is one of the provided legal action objects by default.
Set environment variable IWAPALACE_RETURN_INDEX=1 if your local environment
expects an action index instead of an action object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import os
import random
import re
from typing import Any, Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Card IDs from the supplied card-id list / optimized deck CSV.
# ---------------------------------------------------------------------------
DWEBBLE = 344
CRUSTLE = 345
MUNKIDORI = 112
BUDDY_POFFIN = 1086
BUG_CATCHING_SET = 1094
POKEPAD = 1152
ULTRA_BALL = 1121
POKEGEAR = 1122
NIGHT_STRETCHER = 1097
HERO_MANTLE = 1159
AIR_BALLOON = 1174
JUMBO_ICE = 1147

AKAMATSU = 1198
TOUKO = 1225
ROCKET_LAMBDA = 1219
BOSS_ORDERS = 1182
JUDGE = 1213
LILLIE_DETERMINATION = 1227
VITALITY_FOREST = 1261

GRASS_ENERGY = 1
DARK_ENERGY = 7
MIST_ENERGY = 11
GROW_GRASS_ENERGY = 18

# Attack IDs observed in the existing project Q-models. If the simulator uses
# names instead of IDs, the text matcher below still handles them.
AWAKENING_ATTACK = 478
GREAT_SCISSOR_ATTACK = 479

DECKLIST: dict[int, int] = {
    DWEBBLE: 4,
    CRUSTLE: 4,
    MUNKIDORI: 3,
    BUDDY_POFFIN: 4,
    BUG_CATCHING_SET: 3,
    POKEPAD: 3,
    ULTRA_BALL: 2,
    POKEGEAR: 3,
    NIGHT_STRETCHER: 1,
    HERO_MANTLE: 1,
    AIR_BALLOON: 2,
    JUMBO_ICE: 1,
    AKAMATSU: 3,
    TOUKO: 2,
    ROCKET_LAMBDA: 3,
    BOSS_ORDERS: 3,
    JUDGE: 1,
    LILLIE_DETERMINATION: 2,
    VITALITY_FOREST: 2,
    GRASS_ENERGY: 4,
    DARK_ENERGY: 4,
    MIST_ENERGY: 3,
    GROW_GRASS_ENERGY: 2,
}

CARD_NAMES: dict[int, tuple[str, ...]] = {
    DWEBBLE: ("イシズマイ", "Dwebble"),
    CRUSTLE: ("イワパレス", "Crustle"),
    MUNKIDORI: ("マシマシラ", "Munkidori"),
    BUDDY_POFFIN: ("なかよしポフィン", "Buddy-Buddy Poffin", "Buddy Poffin"),
    BUG_CATCHING_SET: ("むしとりセット", "Bug Catching Set"),
    POKEPAD: ("ポケパッド", "PokePad", "Poke Pad"),
    ULTRA_BALL: ("ハイパーボール", "Ultra Ball"),
    POKEGEAR: ("ポケギア", "Pokegear"),
    NIGHT_STRETCHER: ("夜のタンカ", "Night Stretcher"),
    HERO_MANTLE: ("ヒーローマント", "Hero's Cape", "Hero Cape", "Hero Mantle"),
    AIR_BALLOON: ("ふうせん", "Air Balloon"),
    JUMBO_ICE: ("ジャンボアイス", "Jumbo Ice"),
    AKAMATSU: ("アカマツ", "Akamatsu", "Briar"),
    TOUKO: ("トウコ", "Touko", "Hilda"),
    ROCKET_LAMBDA: ("ロケット団のラムダ", "Team Rocket's Lambda", "Lambda"),
    BOSS_ORDERS: ("ボスの指令", "Boss's Orders", "Boss Orders"),
    JUDGE: ("ジャッジマン", "Judge"),
    LILLIE_DETERMINATION: ("リーリエの決心", "Lillie's Determination"),
    VITALITY_FOREST: ("活力の森", "Vitality Forest"),
    GRASS_ENERGY: ("基本【草】エネルギー", "Basic {G} Energy", "Grass Energy"),
    DARK_ENERGY: ("基本【悪】エネルギー", "Basic {D} Energy", "Darkness Energy", "Dark Energy"),
    MIST_ENERGY: ("ミストエネルギー", "Mist Energy"),
    GROW_GRASS_ENERGY: ("グロウ【草】エネルギー", "Grow Grass Energy"),
}

ENERGY_CARDS = {GRASS_ENERGY, DARK_ENERGY, MIST_ENERGY, GROW_GRASS_ENERGY}

# Cards that tend to spend cards from hand, redraw hands, or search without
# directly improving an already-built board. These are useful during setup,
# but should be deprioritized once Crustle + Munkidori are online.
HAND_REFRESH_OR_FILTER_CARDS = {BUDDY_POFFIN, BUG_CATCHING_SET, POKEPAD, ULTRA_BALL, POKEGEAR, ROCKET_LAMBDA, JUDGE, LILLIE_DETERMINATION, TOUKO}
# Cards that spend the hand/deck tempo.  Akamatsu is included here even though
# it can be useful, because once the board is complete it must justify itself
# by improving energy/backup development rather than being played automatically.
RESOURCE_SPEND_CARDS = HAND_REFRESH_OR_FILTER_CARDS | {AKAMATSU}
HAND_DISCARD_COST_CARDS = {ULTRA_BALL, LILLIE_DETERMINATION}
BENCH_GROWTH_TARGETS = {DWEBBLE, CRUSTLE, MUNKIDORI}
POKEMON_CARDS = {DWEBBLE, CRUSTLE, MUNKIDORI}
SEARCH_CARDS = {BUDDY_POFFIN, BUG_CATCHING_SET, POKEPAD, ULTRA_BALL, POKEGEAR, ROCKET_LAMBDA, TOUKO, AKAMATSU}
SUPPORTERS = {AKAMATSU, TOUKO, ROCKET_LAMBDA, BOSS_ORDERS, JUDGE, LILLIE_DETERMINATION}
TOOLS = {HERO_MANTLE, AIR_BALLOON}
SETUP_CARDS = {
    DWEBBLE, CRUSTLE, MUNKIDORI, BUDDY_POFFIN, BUG_CATCHING_SET,
    POKEPAD, ULTRA_BALL, AKAMATSU, TOUKO, ROCKET_LAMBDA,
    VITALITY_FOREST, GRASS_ENERGY, DARK_ENERGY, GROW_GRASS_ENERGY,
}

BASE_CARD_PRIORITY: dict[int, float] = {
    CRUSTLE: 9.4,
    DWEBBLE: 8.9,
    MUNKIDORI: 7.2,
    BUDDY_POFFIN: 6.3,
    BUG_CATCHING_SET: 5.5,
    POKEPAD: 6.5,
    ULTRA_BALL: 4.9,
    POKEGEAR: 3.5,
    NIGHT_STRETCHER: 2.0,
    HERO_MANTLE: 9.2,
    AIR_BALLOON: 3.2,
    JUMBO_ICE: 2.1,
    AKAMATSU: 6.1,
    TOUKO: 5.8,
    ROCKET_LAMBDA: 5.2,
    BOSS_ORDERS: 3.4,
    JUDGE: 1.5,
    LILLIE_DETERMINATION: 3.4,
    VITALITY_FOREST: 5.6,
    GRASS_ENERGY: 3.5,
    DARK_ENERGY: 4.4,
    MIST_ENERGY: 3.6,
    GROW_GRASS_ENERGY: 4.8,
}


@dataclass
class StateInfo:
    turn: int | None = None
    active_ids: set[int] = field(default_factory=set)
    bench_ids: set[int] = field(default_factory=set)
    hand_ids: set[int] = field(default_factory=set)
    discard_ids: set[int] = field(default_factory=set)
    all_ids: set[int] = field(default_factory=set)
    deck_count: int | None = None
    hand_count: int | None = None
    bench_count: int | None = None
    supporter_played: bool | None = None
    energy_attached: bool | None = None
    your_prize: int | None = None
    opp_prize: int | None = None

    @property
    def early(self) -> bool:
        return self.turn is None or self.turn <= 3 or self.your_prize in (None, 6)

    @property
    def mid_or_late(self) -> bool:
        return not self.early

    @property
    def has_dwebble(self) -> bool:
        return DWEBBLE in self.active_ids or DWEBBLE in self.bench_ids

    @property
    def has_crustle(self) -> bool:
        return CRUSTLE in self.active_ids or CRUSTLE in self.bench_ids

    @property
    def has_munkidori(self) -> bool:
        return MUNKIDORI in self.active_ids or MUNKIDORI in self.bench_ids

    @property
    def behind_on_prizes(self) -> bool:
        if self.your_prize is None or self.opp_prize is None:
            return False
        # In Pokemon TCG, fewer remaining prizes means winning. If opponent has
        # fewer prizes remaining, we are behind.
        return self.opp_prize < self.your_prize


class IwapalaceRLAgent:
    """Heuristic + tabular Q-learning agent."""

    deck_name = "Iwapalace/Crustle dedicated durability + Munkidori damage-move"
    # New version writes deck-aware Q keys, while still reading old v1.1 keys
    # as a fallback so existing q_model_mirror_v3_lossfix.json remains useful.
    policy_version = "iwapalace_rl_v1.4_resource_balance"
    legacy_policy_version = "iwapalace_rl_v1.1_no_shaymin"

    def __init__(
        self,
        *,
        model_path: str | None = None,
        training: bool = False,
        seed: int = 20260620,
        alpha: float = 0.08,
        gamma: float = 0.97,
        epsilon: float | None = None,
        q_weight: float = 6.0,
        return_index: bool | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.training = training
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 0.12 if training else 0.0 if epsilon is None else epsilon
        if epsilon is not None:
            self.epsilon = float(epsilon)
        self.q_weight = q_weight
        self.q_values: dict[str, float] = {}
        self.return_index = self._env_bool("IWAPALACE_RETURN_INDEX", False) if return_index is None else return_index
        self._load_model(model_path)
        if not training:
            self.epsilon = 0.0

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.select_action(*args, **kwargs)

    def act(self, *args: Any, **kwargs: Any) -> Any:
        return self.select_action(*args, **kwargs)

    def choose_action(self, *args: Any, **kwargs: Any) -> Any:
        return self.select_action(*args, **kwargs)

    def policy(self, *args: Any, **kwargs: Any) -> Any:
        return self.select_action(*args, **kwargs)

    def select(self, *args: Any, **kwargs: Any) -> Any:
        return self.select_action(*args, **kwargs)

    def select_action(self, *args: Any, **kwargs: Any) -> Any:
        state, actions = self._split_state_actions(args, kwargs)
        if actions is None:
            candidate_source = state if state is not None else kwargs
            actions = self._get_any(candidate_source, ("legal_actions", "actions", "options", "choices", "valid_actions"))
        action_list = self._normalize_actions(actions)
        if not action_list:
            return None

        state_info = self.summarize_state(state)

        if self.training and self.epsilon > 0 and self.rng.random() < self.epsilon:
            idx = self.rng.randrange(len(action_list))
            return idx if self.return_index else action_list[idx]

        best_idx = 0
        best_score = -10**18
        for idx, action in enumerate(action_list):
            score = self.total_score(state_info, action)
            score += self.rng.random() * 1e-8  # deterministic tie noise
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx if self.return_index else action_list[best_idx]

    def score_action(self, state: Any, action: Any) -> float:
        return self.total_score(self.summarize_state(state), action)

    # ------------------------------------------------------------------
    # Q-learning update helpers for train.py
    # ------------------------------------------------------------------
    def update_q(
        self,
        prev_state: Any,
        action: Any,
        reward: float,
        next_state: Any | None,
        next_actions: Iterable[Any] | None,
        done: bool,
    ) -> float:
        key = self.state_action_key(prev_state, action)
        legacy_key = self.legacy_state_action_key(prev_state, action)
        old = self.q_values.get(key, self.q_values.get(legacy_key, 0.0))
        if done or next_state is None or not next_actions:
            target = reward
        else:
            next_values = [self.q_value_for(next_state, a) for a in list(next_actions)]
            target = reward + self.gamma * (max(next_values) if next_values else 0.0)
        new_value = old + self.alpha * (target - old)
        # Clamp to keep the submitted model stable even after noisy local training.
        self.q_values[key] = max(-5.0, min(5.0, new_value))
        return self.q_values[key]

    def save_model(self, path: str | os.PathLike[str]) -> None:
        data = {
            "metadata": {
                "policy_version": self.policy_version,
                "deck_name": self.deck_name,
                "decklist": DECKLIST,
                "notes": "Tabular Q-values are an overlay on top of deterministic deck heuristics. v1.4 keeps discard/deck-pressure gains but rolls back over-search pressure; deck/resource/board-need buckets retain legacy v1.1 fallback.",
            },
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": 0.0,  # submission should not explore
            "q_weight": self.q_weight,
            "q_values": self.q_values,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def total_score(self, state: StateInfo, action: Any) -> float:
        heuristic = self.heuristic_score(state, action)
        q = self.q_value_for_info(state, action)
        return heuristic + self.q_weight * q

    def q_value_for(self, state: Any, action: Any) -> float:
        return self.q_value_for_info(self.summarize_state(state), action)

    def q_value_for_info(self, state: StateInfo, action: Any) -> float:
        # Prefer the new deck-aware key, but fall back to the old model keys.
        # This lets old trained Q-values work immediately, while further
        # training can learn deck/resource-aware behavior.
        key = self.state_action_key_from_info(state, action)
        if key in self.q_values:
            return self.q_values[key]
        return self.q_values.get(self.legacy_state_action_key_from_info(state, action), 0.0)

    def heuristic_score(self, state: StateInfo, action: Any) -> float:
        text = self._action_text(action)
        text_lower = text.lower()
        card_ids = self.extract_card_ids(action)
        target_ids = self.extract_target_ids(action)
        attack_ids = self.extract_attack_ids(action)
        bucket = self.action_bucket(action)

        score = 0.0

        if bucket == "end":
            score -= 3.2 if state.early else 0.9
        elif bucket == "attack":
            score += 3.0
        elif bucket in {"search", "select"}:
            score += 1.2
        elif bucket == "energy":
            score += 1.0

        for cid in card_ids:
            score += BASE_CARD_PRIORITY.get(cid, 0.0)
            if state.early and cid in SETUP_CARDS:
                score += 1.5
            if state.mid_or_late and cid in {BOSS_ORDERS, NIGHT_STRETCHER, JUMBO_ICE}:
                score += 1.0

        # Search/select choice priority.
        if bucket in {"search", "select"} or self._looks_like_search(text_lower):
            score += self._search_choice_score(state, card_ids)

        # Dwebble -> Crustle is plan A.
        if CRUSTLE in card_ids:
            score += 6.0 if state.has_dwebble and not state.has_crustle else 2.0
        if DWEBBLE in card_ids:
            # Before the first Crustle, Dwebble is essential. After Crustle is
            # established, extra Dwebble is only a low-priority backup.
            if not state.has_dwebble and not state.has_crustle:
                score += 6.0
            elif state.has_crustle:
                score += 0.3
            else:
                score += 1.0
        if MUNKIDORI in card_ids:
            score += 5.6 if (state.has_crustle and not state.has_munkidori) else (4.5 if not state.has_munkidori else 0.8)

        # Early search cards are useful only while they advance setup.
        if card_ids & SEARCH_CARDS:
            if not state.has_dwebble:
                score += 2.4
            if state.has_dwebble and not state.has_crustle:
                score += 2.2
            if not state.has_munkidori:
                score += 0.8

        # Stadium.
        if VITALITY_FOREST in card_ids:
            score += 4.0 if state.has_dwebble and not state.has_crustle else 0.4

        # Hero Mantle should go to Crustle only. If the action has an
        # explicit target and that target is not Crustle, it is a serious
        # misplay even when a Crustle exists elsewhere on the board.
        if HERO_MANTLE in card_ids or "hero" in text_lower or "ヒーローマント" in text:
            if target_ids:
                if CRUSTLE in target_ids:
                    score += 10.5
                else:
                    score -= 28.0
            else:
                # Searching/selecting Hero Mantle before target selection.
                # v3 loss-log: no wrong Hero target remained, but missing Hero
                # Mantle was still common, so raise the search priority.
                score += 5.0 if state.has_crustle else 0.4

        # Air Balloon is a pivot card, but attaching it to Dwebble is also
        # bad: when Dwebble evolves, Crustle inherits the tool and can no
        # longer receive Hero Mantle. v3 loss-log showed Balloon->Dwebble
        # remained, so Dwebble and Crustle are both hard negatives.
        if AIR_BALLOON in card_ids:
            if target_ids & {CRUSTLE, DWEBBLE}:
                score -= 13.0
            elif MUNKIDORI in target_ids:
                score += 2.8
            else:
                score -= 0.4

        # Energy attachment.
        if (card_ids & ENERGY_CARDS) or bucket == "energy" or self._looks_like_attach(text_lower):
            score += self._energy_score(state, card_ids, target_ids, text_lower)

        # Supporter timing.
        if state.supporter_played is True and (card_ids & SUPPORTERS or bucket == "supporter"):
            score -= 7.0
        if AKAMATSU in card_ids:
            score += 2.3 if not state.has_munkidori else 1.2
        if TOUKO in card_ids:
            score += 2.3 if state.has_dwebble and not state.has_crustle else 0.7
        if ROCKET_LAMBDA in card_ids:
            score += 1.7
        if BOSS_ORDERS in card_ids:
            score += 0.2 if state.early else 3.2
        if JUDGE in card_ids:
            score += 2.0 if state.behind_on_prizes else (-0.8 if state.early else 0.3)
        if LILLIE_DETERMINATION in card_ids:
            score += 1.4 if state.early else 0.4

        # Recovery / healing.
        if JUMBO_ICE in card_ids:
            score += 2.4 if CRUSTLE in state.active_ids else -0.4
        if NIGHT_STRETCHER in card_ids:
            score += -0.3 if state.early else 1.8

        # Attacks / abilities by ID or text.
        if AWAKENING_ATTACK in attack_ids or "かくせい" in text or "awaken" in text_lower:
            score += 13.0 if DWEBBLE in state.active_ids and not state.has_crustle else 3.0
        if GREAT_SCISSOR_ATTACK in attack_ids or "グレートシザー" in text or "great scissor" in text_lower:
            # v3 loss-log: 34/94 losses still never used Great Scissor.
            score += 10.8 if CRUSTLE in state.active_ids else 2.0
        if "アドレナブレイン" in text or "adrena" in text_lower or "damage counter" in text_lower or "ダメカン" in text:
            score += 6.0 if state.has_munkidori else 1.5

        # Prize pressure.
        if any(w in text_lower for w in ("knock", "ko", "prize")) or any(w in text for w in ("きぜつ", "サイド")):
            score += 3.0

        return score

    def _search_choice_score(self, state: StateInfo, card_ids: set[int]) -> float:
        score = 0.0
        if CRUSTLE in card_ids:
            score += 10.0 if state.has_dwebble and not state.has_crustle else 4.5
        if DWEBBLE in card_ids:
            if not state.has_dwebble and not state.has_crustle:
                score += 10.0
            elif state.has_crustle:
                score += 0.8
            else:
                score += 2.0
        if MUNKIDORI in card_ids:
            score += 9.0 if (state.has_crustle and not state.has_munkidori) else (7.0 if not state.has_munkidori else 1.2)
        if HERO_MANTLE in card_ids:
            # v3 loss-log: 35/94 losses still had no Hero Mantle attached.
            # When Crustle exists, finding Hero Mantle is a top priority.
            score += 11.0 if state.has_crustle else 3.0
        if VITALITY_FOREST in card_ids:
            score += 5.5 if state.has_dwebble and not state.has_crustle else 1.8
        if AKAMATSU in card_ids:
            score += 5.3
        if TOUKO in card_ids:
            score += 5.0 if state.has_dwebble else 3.0
        if BOSS_ORDERS in card_ids:
            score += 0.6 if state.early else 4.5
        if DARK_ENERGY in card_ids:
            score += 4.6 if state.has_munkidori else 2.2
        if GROW_GRASS_ENERGY in card_ids:
            score += 4.5 if state.has_crustle or state.has_dwebble else 1.9
        if GRASS_ENERGY in card_ids:
            score += 3.2
        if MIST_ENERGY in card_ids:
            score += 2.8 if state.has_crustle else 1.2
        return score

    def _energy_score(self, state: StateInfo, card_ids: set[int], target_ids: set[int], text_lower: str) -> float:
        if state.energy_attached is True and self._looks_like_attach(text_lower):
            return -6.0
        score = 0.0
        explicit_target = bool(target_ids)

        if DARK_ENERGY in card_ids:
            if explicit_target:
                if MUNKIDORI in target_ids:
                    score += 8.0
                else:
                    # Dark Energy is mainly for Munkidori. v3 loss-log still
                    # had Dark Energy attached to Dwebble/Crustle, so make
                    # non-Munkidori targets a hard negative.
                    score -= 8.0
            else:
                score += 3.5 if state.has_munkidori else 0.8

        if GROW_GRASS_ENERGY in card_ids:
            if explicit_target:
                if target_ids & {CRUSTLE, DWEBBLE}:
                    score += 7.0
                elif MUNKIDORI in target_ids:
                    # Grow Grass belongs on the Crustle line, not Munkidori.
                    score -= 7.0
                else:
                    score -= 1.5
            elif CRUSTLE in state.active_ids or DWEBBLE in state.active_ids:
                score += 4.0
            else:
                score += 1.2

        if GRASS_ENERGY in card_ids:
            if explicit_target:
                if target_ids & {CRUSTLE, DWEBBLE}:
                    score += 5.0
                elif MUNKIDORI in target_ids:
                    score -= 6.0
                else:
                    score -= 0.8
            elif CRUSTLE in state.active_ids or DWEBBLE in state.active_ids:
                score += 3.2
            else:
                score += 0.8

        if MIST_ENERGY in card_ids:
            if explicit_target:
                if target_ids & {CRUSTLE, DWEBBLE}:
                    score += 5.2
                elif MUNKIDORI in target_ids:
                    score -= 7.0
                else:
                    score -= 0.8
            elif CRUSTLE in state.active_ids:
                score += 3.0
            else:
                score += 0.6

        if not (card_ids & ENERGY_CARDS) and self._looks_like_attach(text_lower):
            score += 1.0
        return score

    # ------------------------------------------------------------------
    # State/action abstractions for Q-table
    # ------------------------------------------------------------------
    def state_action_key(self, state: Any, action: Any) -> str:
        return self.state_action_key_from_info(self.summarize_state(state), action)

    def legacy_state_action_key(self, state: Any, action: Any) -> str:
        return self.legacy_state_action_key_from_info(self.summarize_state(state), action)

    def state_action_key_from_info(self, state: StateInfo, action: Any) -> str:
        return self._state_action_key_from_info(state, action, version=self.policy_version, include_deck=True)

    def legacy_state_action_key_from_info(self, state: StateInfo, action: Any) -> str:
        return self._state_action_key_from_info(state, action, version=self.legacy_policy_version, include_deck=False)

    def _state_action_key_from_info(self, state: StateInfo, action: Any, *, version: str, include_deck: bool) -> str:
        card_ids = sorted(self.extract_card_ids(action))
        attack_ids = sorted(self.extract_attack_ids(action))
        target_ids = sorted(self.extract_target_ids(action))
        primary_card = card_ids[0] if card_ids else None
        primary_attack = attack_ids[0] if attack_ids else None
        primary_target = target_ids[0] if target_ids else None
        flags = "".join([
            "D" if state.has_dwebble else "d",
            "C" if state.has_crustle else "c",
            "M" if state.has_munkidori else "m",
        ])
        deck_part = ""
        if include_deck:
            # v1.3 Q keys expose resource pressure and board readiness, so
            # training can learn when thin-deck search is still correct instead
            # of relying on a fixed deck-count ban.
            deck_part = (
                f"db={self._deck_bucket(state.deck_count)}|"
                f"bc={int(state.has_crustle and state.has_munkidori)}|"
                f"bn={self._board_need_bucket(state)}|"
            )
        return (
            f"v={version}|tb={self._turn_bucket(state.turn)}|"
            f"yp={self._prize_bucket(state.your_prize)}|op={self._prize_bucket(state.opp_prize)}|"
            f"{deck_part}"
            f"ea={int(bool(state.energy_attached))}|sp={int(bool(state.supporter_played))}|"
            f"act={self.action_bucket(action)}|card={primary_card}|atk={primary_attack}|tgt={primary_target}|flags={flags}"
        )

    def action_bucket(self, action: Any) -> str:
        """Classify action without letting field names like option_card look like search.

        The official cg bridge represents every candidate as a dict containing
        option_type/option_card/target_card.  The old string fallback treated the
        word "option" itself as a search signal, which polluted Q-keys and made
        generic unidentified actions too attractive.  Prefer explicit cg option
        types first, then fall back to text only for non-cg environments.
        """
        if isinstance(action, Mapping):
            opt_type = action.get("option_type")
            card_ids0 = self.extract_card_ids(action)
            if opt_type == OPTION_END:
                return "end"
            if opt_type == OPTION_ATTACK:
                return "attack"
            if opt_type == OPTION_ABILITY:
                return "ability"
            if opt_type == OPTION_RETREAT:
                return "switch"
            if opt_type == OPTION_ATTACH:
                if card_ids0 & TOOLS:
                    return "tool"
                return "energy"
            if opt_type == OPTION_EVOLVE:
                # Keep "search" for backward compatibility with existing Q keys
                # such as card=345,target=344.
                return "search"
            if opt_type == OPTION_PLAY:
                if card_ids0 & ENERGY_CARDS:
                    return "energy"
                if card_ids0 & TOOLS:
                    return "tool"
                if card_ids0 & SUPPORTERS:
                    return "supporter"
                if VITALITY_FOREST in card_ids0:
                    return "stadium"
                if NIGHT_STRETCHER in card_ids0 or JUMBO_ICE in card_ids0:
                    return "recovery"
                if card_ids0 & SEARCH_CARDS:
                    return "search"
                return "play"
            if opt_type in {OPTION_CARD, OPTION_TOOL_CARD, OPTION_ENERGY_CARD, OPTION_SKILL}:
                return "search"
            if opt_type == OPTION_ENERGY:
                return "energy"
            if opt_type == OPTION_DISCARD:
                return "discard"

        text = self._action_text(action)
        tl = text.lower()
        card_ids = self.extract_card_ids(action)
        if self._looks_like_end(tl):
            return "end"
        if self.extract_attack_ids(action) or any(x in tl for x in ("attack", "ワザ", "かくせい", "グレートシザー")):
            return "attack"
        if "ability" in tl or "特性" in text or "アドレナ" in text:
            return "ability"
        if card_ids & ENERGY_CARDS or self._looks_like_attach(tl):
            return "energy"
        if card_ids & TOOLS or "tool" in tl or "どうぐ" in text:
            return "tool"
        if VITALITY_FOREST in card_ids or "stadium" in tl or "スタジアム" in text:
            return "stadium"
        if card_ids & SUPPORTERS or "supporter" in tl or "サポート" in text:
            return "supporter"
        if NIGHT_STRETCHER in card_ids or JUMBO_ICE in card_ids or "recover" in tl or "回復" in text:
            return "recovery"
        if self._looks_like_search(tl):
            return "search"
        if any(x in tl for x in ("switch", "retreat")) or any(x in text for x in ("にげ", "逃げ", "入れ替")):
            return "switch"
        if any(x in tl for x in ("select", "choose")) or "選" in text:
            return "select"
        return "other"

    @staticmethod
    def _turn_bucket(turn: int | None) -> str:
        if turn is None:
            return "u"
        if turn <= 1:
            return "1"
        if turn <= 3:
            return "2-3"
        if turn <= 6:
            return "4-6"
        return "7+"

    @staticmethod
    def _prize_bucket(prize: int | None) -> str:
        if prize is None:
            return "u"
        if prize >= 5:
            return "5-6"
        if prize >= 3:
            return "3-4"
        return "0-2"

    @staticmethod
    def _deck_bucket(deck_count: int | None) -> str:
        # Coarse deck-count bucket for learning deck-out / resource management.
        # This is intentionally a state feature, not a hard-coded stop rule.
        if deck_count is None:
            return "u"
        if deck_count <= 0:
            return "0"
        if deck_count <= 3:
            return "1-3"
        if deck_count <= 6:
            return "4-6"
        if deck_count <= 10:
            return "7-10"
        if deck_count <= 20:
            return "11-20"
        return "21+"

    @staticmethod
    def _board_need_bucket(state: StateInfo) -> str:
        # Lightweight board-need feature for Q-learning.  This deliberately
        # avoids a hard rule like "deck <= 5 never search"; it lets the model
        # learn whether search is correct in thin-deck states depending on how
        # incomplete the board still is.
        need = 0
        if not state.has_dwebble and not state.has_crustle:
            need += 3
        if state.has_dwebble and not state.has_crustle:
            need += 3
        if not state.has_munkidori:
            need += 2
        if state.has_crustle and not state.has_munkidori:
            need += 1
        if state.behind_on_prizes:
            need += 1
        if need >= 5:
            return "high"
        if need >= 2:
            return "mid"
        return "low"

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------
    def summarize_state(self, state: Any) -> StateInfo:
        info = StateInfo()
        if state is None:
            return info
        info.turn = self._first_int(state, ("turn", "turn_count", "current_turn", "turn_number"))
        info.supporter_played = self._first_bool(state, ("supporter_played", "used_supporter", "supporter_used"))
        info.energy_attached = self._first_bool(state, ("energy_attached", "attached_energy", "energy_played", "energy_used"))
        info.your_prize = self._first_int(state, ("your_prize", "my_prize", "player_prize", "prizes_remaining", "my_prizes_remaining"))
        info.opp_prize = self._first_int(state, ("opp_prize", "opponent_prize", "enemy_prize", "opponent_prizes_remaining"))
        info.active_ids = self._ids_from_zones(state, ("active", "active_pokemon", "your_active", "my_active", "battle_pokemon"))
        info.bench_ids = self._ids_from_zones(state, ("bench", "your_bench", "my_bench", "benched"))
        info.hand_ids = self._ids_from_zones(state, ("hand", "your_hand", "my_hand"))
        info.discard_ids = self._ids_from_zones(state, ("discard", "trash", "discard_pile", "your_discard"))
        info.deck_count = self._first_int(state, ("deck_count", "deck_remaining", "my_deck_count", "your_deck_count"))
        if info.deck_count is None:
            deck_zone = self._get_any(state, ("deck", "your_deck", "my_deck"))
            if isinstance(deck_zone, Sequence) and not isinstance(deck_zone, (str, bytes, bytearray)):
                info.deck_count = len(deck_zone)
        info.hand_count = self._first_int(state, ("hand_count", "my_hand_count", "your_hand_count"))
        if info.hand_count is None:
            hand_zone = self._get_any(state, ("hand", "your_hand", "my_hand"))
            if isinstance(hand_zone, Sequence) and not isinstance(hand_zone, (str, bytes, bytearray)):
                info.hand_count = len(hand_zone)
        info.bench_count = self._first_int(state, ("bench_count", "my_bench_count", "your_bench_count"))
        if info.bench_count is None:
            bench_zone = self._get_any(state, ("bench", "your_bench", "my_bench", "benched"))
            if isinstance(bench_zone, Sequence) and not isinstance(bench_zone, (str, bytes, bytearray)):
                info.bench_count = len(bench_zone)
        info.all_ids = self._extract_all_card_ids(state)
        return info

    def extract_card_ids(self, obj: Any) -> set[int]:
        """Return the card being played/selected, not its target.

        Official cg features carry `option_card` and `target_card` separately.
        The previous text fallback also read names contained in `target=...`,
        causing actions such as "attach Hero Mantle to Crustle" to be keyed as
        both card=1159 and card=345.  For mapping actions, trust explicit
        card-like fields only.
        """
        ids: set[int] = set()
        if isinstance(obj, Mapping):
            for key in ("option_card", "card", "card_id", "cardId", "selected_card", "played_card"):
                value = obj.get(key)
                if isinstance(value, int):
                    ids.add(value)
                elif isinstance(value, Mapping):
                    cid = value.get("id", value.get("cardId"))
                    if isinstance(cid, int):
                        ids.add(cid)
            return ids & set(DECKLIST)

        self._collect_card_like_ids(obj, ids, max_depth=5)
        text = self._action_text(obj)
        for cid, names in CARD_NAMES.items():
            if any(name and name in text for name in names):
                ids.add(cid)
        return ids & set(DECKLIST)


    def extract_attack_ids(self, obj: Any) -> set[int]:
        ids: set[int] = set()
        if isinstance(obj, Mapping):
            value = obj.get("option_attack")
            if isinstance(value, int):
                ids.add(value)
            value = obj.get("attackId")
            if isinstance(value, int):
                ids.add(value)
            return ids | {i for i in ids if i in {AWAKENING_ATTACK, GREAT_SCISSOR_ATTACK}}
        self._collect_ids_by_hint(obj, ids, hints=("attack", "option_attack"), max_depth=5)
        all_ints: set[int] = set()
        self._collect_all_ints(obj, all_ints, max_depth=4)
        return ids | {i for i in all_ints if i in {AWAKENING_ATTACK, GREAT_SCISSOR_ATTACK}}

    def extract_target_ids(self, obj: Any) -> set[int]:
        """Return explicit target Pokemon IDs only.

        Do not infer target IDs from free text. Otherwise the played card name
        itself (e.g. Hero Mantle) can become `target=1159`, which is impossible
        and poisons Q-keys and scoring.
        """
        ids: set[int] = set()
        if isinstance(obj, Mapping):
            for key in ("target_card", "target", "target_id", "targetId", "in_play_card", "pokemon"):
                value = obj.get(key)
                if isinstance(value, int):
                    ids.add(value)
                elif isinstance(value, Mapping):
                    cid = value.get("id", value.get("cardId"))
                    if isinstance(cid, int):
                        ids.add(cid)
            return ids & set(DECKLIST)

        self._collect_ids_by_hint(obj, ids, hints=("target", "in_play"), max_depth=5)
        return ids & set(DECKLIST)

    def _extract_all_card_ids(self, obj: Any) -> set[int]:
        ids: set[int] = set()
        self._collect_all_ints(obj, ids, max_depth=6)
        text = self._action_text(obj)
        for cid, names in CARD_NAMES.items():
            if any(name and name in text for name in names):
                ids.add(cid)
        return {i for i in ids if i in DECKLIST}

    def _ids_from_zones(self, obj: Any, zone_names: Iterable[str]) -> set[int]:
        ids: set[int] = set()
        for name in zone_names:
            zone = self._get_any(obj, (name,))
            if zone is not None:
                ids |= self._extract_all_card_ids(zone)
        return ids

    def _collect_card_like_ids(self, obj: Any, out: set[int], *, max_depth: int) -> None:
        if obj is None or max_depth < 0:
            return
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                key = str(k).lower()
                # Include explicit card selectors; exclude targets and in-play references.
                is_card_field = (
                    key in {"card", "card_id", "option_card", "selected_card", "played_card"}
                    or ("option_card" in key)
                    or ("card_id" in key)
                )
                is_target_field = any(h in key for h in ("target", "attach", "to", "in_play", "pokemon")) and "option_card" not in key
                if is_card_field and not is_target_field:
                    self._collect_all_ints(v, out, max_depth=1)
                else:
                    self._collect_card_like_ids(v, out, max_depth=max_depth - 1)
        elif isinstance(obj, (list, tuple, set, frozenset)):
            for v in list(obj)[:120]:
                self._collect_card_like_ids(v, out, max_depth=max_depth - 1)
        elif hasattr(obj, "__dict__"):
            self._collect_card_like_ids(vars(obj), out, max_depth=max_depth - 1)

    def _collect_ids_by_hint(self, obj: Any, out: set[int], *, hints: tuple[str, ...], max_depth: int) -> None:
        if obj is None or max_depth < 0:
            return
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                key = str(k).lower()
                if any(h in key for h in hints):
                    self._collect_all_ints(v, out, max_depth=2)
                self._collect_ids_by_hint(v, out, hints=hints, max_depth=max_depth - 1)
        elif isinstance(obj, (list, tuple, set, frozenset)):
            for v in list(obj)[:120]:
                self._collect_ids_by_hint(v, out, hints=hints, max_depth=max_depth - 1)
        elif hasattr(obj, "__dict__"):
            self._collect_ids_by_hint(vars(obj), out, hints=hints, max_depth=max_depth - 1)

    def _collect_all_ints(self, obj: Any, out: set[int], *, max_depth: int) -> None:
        if obj is None or max_depth < 0 or isinstance(obj, bool):
            return
        if isinstance(obj, int):
            out.add(obj)
            return
        if isinstance(obj, float) and obj.is_integer():
            out.add(int(obj))
            return
        if isinstance(obj, str):
            # Pull explicit ids only; avoid treating arbitrary text numbers as IDs.
            for token in re.findall(r"(?:card_id|option_card|attack_id|option_attack)\D+(\d+)", obj, flags=re.IGNORECASE):
                try:
                    out.add(int(token))
                except ValueError:
                    pass
            return
        if isinstance(obj, Mapping):
            for v in obj.values():
                self._collect_all_ints(v, out, max_depth=max_depth - 1)
        elif isinstance(obj, (list, tuple, set, frozenset)):
            for v in list(obj)[:160]:
                self._collect_all_ints(v, out, max_depth=max_depth - 1)
        elif hasattr(obj, "__dict__"):
            self._collect_all_ints(vars(obj), out, max_depth=max_depth - 1)

    # ------------------------------------------------------------------
    # Generic utilities
    # ------------------------------------------------------------------
    def _load_model(self, model_path: str | None) -> None:
        base = Path(__file__).resolve().parent
        candidates: list[Path] = []
        if model_path:
            candidates.append(Path(model_path))
        env_path = os.environ.get("IWAPALACE_MODEL_PATH")
        if env_path:
            candidates.append(Path(env_path))
        candidates.extend([
            base / "q_model_submit.json",
            base / "q_model_training.json",
            base / "rl_q_model.json",
        ])
        for path in candidates:
            try:
                if path.exists():
                    data = json.loads(path.read_text(encoding="utf-8"))
                    self.q_values = {str(k): float(v) for k, v in data.get("q_values", {}).items()}
                    self.alpha = float(data.get("alpha", self.alpha))
                    self.gamma = float(data.get("gamma", self.gamma))
                    self.q_weight = float(data.get("q_weight", self.q_weight))
                    if self.training:
                        self.epsilon = float(data.get("epsilon_train", self.epsilon))
                    return
            except Exception:
                continue

    def _split_state_actions(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> tuple[Any | None, Any | None]:
        state = self._get_any(kwargs, ("state", "game_state", "observation", "obs"))
        actions = self._get_any(kwargs, ("legal_actions", "actions", "options", "choices", "valid_actions"))
        if actions is not None:
            return state, actions
        if len(args) >= 2:
            if self._looks_like_action_sequence(args[1]):
                return args[0], args[1]
            if self._looks_like_action_sequence(args[0]):
                return args[1], args[0]
            return args[0], args[1]
        if len(args) == 1:
            return args[0], None
        return state, actions

    @staticmethod
    def _normalize_actions(actions: Any) -> list[Any]:
        if actions is None:
            return []
        if isinstance(actions, Mapping):
            return list(actions.values())
        if isinstance(actions, (str, bytes)):
            return [actions]
        try:
            return list(actions)
        except TypeError:
            return [actions]

    @staticmethod
    def _get_any(obj: Any, names: Iterable[str], default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, Mapping):
            lowered = {str(k).lower(): k for k in obj.keys()}
            for name in names:
                if name in obj:
                    return obj[name]
                key = lowered.get(name.lower())
                if key is not None:
                    return obj[key]
        for name in names:
            if hasattr(obj, name):
                try:
                    return getattr(obj, name)
                except Exception:
                    pass
        return default

    def _first_int(self, obj: Any, names: Iterable[str]) -> int | None:
        value = self._get_any(obj, names)
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return None

    def _first_bool(self, obj: Any, names: Iterable[str]) -> bool | None:
        value = self._get_any(obj, names)
        return value if isinstance(value, bool) else None

    @staticmethod
    def _looks_like_action_sequence(value: Any) -> bool:
        if value is None or isinstance(value, Mapping) or isinstance(value, (str, bytes)):
            return False
        return isinstance(value, Sequence) or hasattr(value, "__iter__")

    @staticmethod
    def _looks_like_end(text_lower: str) -> bool:
        return any(x in text_lower for x in ("end", "pass", "skip", "done")) or any(x in text_lower for x in ("番を終", "終了", "パス"))

    @staticmethod
    def _looks_like_search(text_lower: str) -> bool:
        # Do not include the word "option": official cg feature keys contain
        # option_card/option_type for every action, which would classify almost
        # everything as search.
        return any(x in text_lower for x in ("search", "select", "choose", "find", "deck")) or any(x in text_lower for x in ("選", "山札", "手札に加", "ベンチに出"))

    @staticmethod
    def _looks_like_attach(text_lower: str) -> bool:
        return any(x in text_lower for x in ("attach", "energy")) or any(x in text_lower for x in ("エネルギー", "つける", "付ける"))

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _action_text(self, obj: Any) -> str:
        try:
            if obj is None:
                return ""
            if isinstance(obj, str):
                return obj
            if isinstance(obj, Mapping):
                parts: list[str] = []
                for k, v in obj.items():
                    if isinstance(v, (str, int, float, bool)):
                        parts.append(f"{k}={v}")
                    elif str(k).lower() in {"name", "card_name", "attack_name", "description", "text"}:
                        parts.append(str(v))
                return "|".join(parts) or str(obj)
            return str(obj)
        except Exception:
            return ""

    def explain_plan(self) -> str:
        return (
            "Plan: set up Dwebble, evolve into Crustle quickly, attach Hero's Mantle/Grow Grass/Mist for durability, "
            "enable Munkidori with Dark Energy, then use damage-counter pressure plus Boss's Orders and Great Scissor. "
            "Shaymin is intentionally not part of this policy; PokePad is the extra consistency slot."
        )




# ---------------------------------------------------------------------------
# Official Kaggle cg API bridge
# ---------------------------------------------------------------------------
# The official simulator passes a single Observation-like dict and expects
# list[int] containing selected option indices.  This bridge converts cg.Option
# dictionaries into the abstract action dictionaries consumed by IwapalaceRLAgent.

# Numeric constants copied from cg.api to keep submission self-contained.
AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6
AREA_STADIUM = 7
AREA_ENERGY = 8
AREA_TOOL = 9
AREA_PRE_EVOLUTION = 10
AREA_PLAYER = 11
AREA_LOOKING = 12

SELECT_MAIN = 0
SELECT_CARD = 1
SELECT_YES_NO = 9
SELECT_COUNT = 8

CONTEXT_SETUP_ACTIVE = 1
CONTEXT_SETUP_BENCH = 2
CONTEXT_IS_FIRST = 41
CONTEXT_MULLIGAN = 42
CONTEXT_ACTIVATE = 43
CONTEXT_DAMAGE_COUNTER = 13
CONTEXT_DAMAGE_COUNTER_ANY = 14
CONTEXT_ATTACK = 35
CONTEXT_DRAW_COUNT = 38

OPTION_NUMBER = 0
OPTION_YES = 1
OPTION_NO = 2
OPTION_CARD = 3
OPTION_TOOL_CARD = 4
OPTION_ENERGY_CARD = 5
OPTION_ENERGY = 6
OPTION_PLAY = 7
OPTION_ATTACH = 8
OPTION_EVOLVE = 9
OPTION_ABILITY = 10
OPTION_DISCARD = 11
OPTION_RETREAT = 12
OPTION_ATTACK = 13
OPTION_END = 14
OPTION_SKILL = 15


def _dget(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _card_id(card: Any) -> int | None:
    if card is None:
        return None
    value = _dget(card, "id", None)
    if isinstance(value, int):
        return value
    value = _dget(card, "cardId", None)
    return value if isinstance(value, int) else None


def _player_state(observation: Any, player_index: int | None = None) -> Any:
    cur = _dget(observation, "current", {}) or {}
    players = _dget(cur, "players", []) or []
    if player_index is None:
        player_index = _dget(cur, "yourIndex", 0) or 0
    try:
        return players[player_index]
    except Exception:
        return {}


def _current_player_index(observation: Any) -> int:
    cur = _dget(observation, "current", {}) or {}
    idx = _dget(cur, "yourIndex", 0)
    return idx if isinstance(idx, int) else 0


def cg_state_for_agent(observation: Any) -> dict[str, Any]:
    """Convert official cg observation into the loose state shape used by the model."""
    cur = _dget(observation, "current", None)
    if cur is None:
        return {}
    yi = _current_player_index(observation)
    me = _player_state(observation, yi)
    opp = _player_state(observation, 1 - yi)
    return {
        "turn": _dget(cur, "turn", None),
        "turn_action_count": _dget(cur, "turnActionCount", None),
        "supporter_played": _dget(cur, "supporterPlayed", None),
        "energy_attached": _dget(cur, "energyAttached", None),
        "active": _dget(me, "active", []) or [],
        "bench": _dget(me, "bench", []) or [],
        "hand": _dget(me, "hand", []) or [],
        "deck": _dget(me, "deck", []) or [],
        "deck_count": len(_dget(me, "deck", []) or []),
        "discard": _dget(me, "discard", []) or [],
        "your_prize": len(_dget(me, "prize", []) or []),
        "opp_prize": len(_dget(opp, "prize", []) or []),
    }


def _cards_from_area(observation: Any, area: int | None, index: int | None, player_index: int | None = None) -> Any:
    if area is None or index is None:
        return None
    select = _dget(observation, "select", {}) or {}
    cur = _dget(observation, "current", {}) or {}
    if player_index is None:
        player_index = _current_player_index(observation)
    if area == AREA_DECK and _dget(select, "deck", None) is not None:
        cards = _dget(select, "deck", []) or []
        return cards[index] if 0 <= index < len(cards) else None
    if area == AREA_LOOKING:
        cards = _dget(cur, "looking", []) or []
        return cards[index] if 0 <= index < len(cards) else None
    if area == AREA_STADIUM:
        cards = _dget(cur, "stadium", []) or []
        return cards[index] if 0 <= index < len(cards) else None
    ps = _player_state(observation, player_index)
    zone_name = {
        AREA_HAND: "hand",
        AREA_DISCARD: "discard",
        AREA_ACTIVE: "active",
        AREA_BENCH: "bench",
        AREA_PRIZE: "prize",
    }.get(area)
    if not zone_name:
        return None
    cards = _dget(ps, zone_name, []) or []
    return cards[index] if 0 <= index < len(cards) else None


def _pokemon_from_area(observation: Any, area: int | None, index: int | None, player_index: int | None = None) -> Any:
    return _cards_from_area(observation, area, index, player_index)


def enrich_cg_option(observation: Any, option: Any, idx: int | None = None) -> dict[str, Any]:
    """Make a feature dict with real card IDs from official cg option indices."""
    select = _dget(observation, "select", {}) or {}
    yi = _current_player_index(observation)
    typ = _dget(option, "type", None)
    feature: dict[str, Any] = {
        "official_index": idx,
        "select_type": _dget(select, "type", None),
        "context": _dget(select, "context", None),
        "option_type": typ,
        "option_card": None,
        "target_card": None,
        "option_attack": None,
        "number": _dget(option, "number", None),
        "source_area": _dget(option, "area", None),
        "source_index": _dget(option, "index", None),
        "target_area": _dget(option, "inPlayArea", None),
        "target_index": _dget(option, "inPlayIndex", None),
    }

    card = None
    target = None

    if typ == OPTION_PLAY:
        card = _cards_from_area(observation, AREA_HAND, _dget(option, "index", None), yi)
    elif typ == OPTION_ATTACH:
        card = _cards_from_area(observation, _dget(option, "area", None), _dget(option, "index", None), yi)
        target = _pokemon_from_area(observation, _dget(option, "inPlayArea", None), _dget(option, "inPlayIndex", None), yi)
    elif typ == OPTION_EVOLVE:
        card = _cards_from_area(observation, _dget(option, "area", None), _dget(option, "index", None), yi)
        target = _pokemon_from_area(observation, _dget(option, "inPlayArea", None), _dget(option, "inPlayIndex", None), yi)
    elif typ in {OPTION_CARD, OPTION_TOOL_CARD, OPTION_ENERGY_CARD}:
        pi = _dget(option, "playerIndex", yi)
        card = _cards_from_area(observation, _dget(option, "area", None), _dget(option, "index", None), pi)
        if typ == OPTION_TOOL_CARD and card is not None:
            tools = _dget(card, "tools", []) or []
            ti = _dget(option, "toolIndex", None)
            if isinstance(ti, int) and 0 <= ti < len(tools):
                card = tools[ti]
        if typ == OPTION_ENERGY_CARD and card is not None:
            energies = _dget(card, "energyCards", []) or []
            ei = _dget(option, "energyIndex", None)
            if isinstance(ei, int) and 0 <= ei < len(energies):
                card = energies[ei]
    elif typ == OPTION_ENERGY:
        target = _pokemon_from_area(observation, _dget(option, "area", None), _dget(option, "index", None), _dget(option, "playerIndex", yi))
    elif typ in {OPTION_ABILITY, OPTION_DISCARD}:
        card = _cards_from_area(observation, _dget(option, "area", None), _dget(option, "index", None), yi)
    elif typ == OPTION_ATTACK:
        feature["option_attack"] = _dget(option, "attackId", None)
    elif typ == OPTION_SKILL:
        feature["option_card"] = _dget(option, "cardId", None)

    cid = _card_id(card)
    tid = _card_id(target)
    if cid is not None:
        feature["option_card"] = cid
        names = CARD_NAMES.get(cid)
        if names:
            feature["card_name"] = names[0]
    if tid is not None:
        feature["target_card"] = tid
        names = CARD_NAMES.get(tid)
        if names:
            feature["target_name"] = names[0]

    # Add readable action name for text matching fallback.
    names: list[str] = []
    if typ == OPTION_END:
        names.append("end turn")
    elif typ == OPTION_RETREAT:
        names.append("retreat")
    elif typ == OPTION_YES:
        names.append("yes")
    elif typ == OPTION_NO:
        names.append("no")
    elif feature.get("option_attack"):
        names.append(f"attack={feature['option_attack']}")
    if feature.get("option_card") in CARD_NAMES:
        names.append(CARD_NAMES[feature["option_card"]][0])
    if feature.get("target_card") in CARD_NAMES:
        names.append("target=" + CARD_NAMES[feature["target_card"]][0])
    feature["name"] = "|".join(names)
    return feature


def _zone_len(player_state: Any, name: str) -> int:
    zone = _dget(player_state, name, []) or []
    try:
        return len(zone)
    except Exception:
        return 0



def _attached_energy_ids(card: Any) -> list[int]:
    energies = _dget(card, "energyCards", None)
    if energies is None:
        energies = _dget(card, "energies", None)
    result: list[int] = []
    for e in _as_list(energies):
        cid = _card_id(e)
        if cid is not None:
            result.append(cid)
    return result


def _inplay_cards(me: Any) -> list[Any]:
    cards: list[Any] = []
    cards.extend(_as_list(_dget(me, "active", []) or []))
    cards.extend(_as_list(_dget(me, "bench", []) or []))
    return cards


def _has_tool(card: Any, tool_id: int) -> bool:
    tools = _dget(card, "tools", None)
    if tools is None:
        tools = _dget(card, "toolCards", None)
    return any(_card_id(t) == tool_id for t in _as_list(tools))


def _count_inplay(me: Any, card_id: int) -> int:
    return sum(1 for c in _inplay_cards(me) if _card_id(c) == card_id)


def _munkidori_has_dark(me: Any) -> bool:
    for c in _inplay_cards(me):
        if _card_id(c) == MUNKIDORI and DARK_ENERGY in _attached_energy_ids(c):
            return True
    return False


def _target_energy_count(observation: Any, feat: Mapping[str, Any]) -> int:
    yi = _current_player_index(observation)
    target = _pokemon_from_area(observation, feat.get("target_area"), feat.get("target_index"), yi)
    return len(_attached_energy_ids(target)) if target is not None else 0


def _cg_board_features(observation: Any, model: IwapalaceRLAgent) -> dict[str, Any]:
    """Official-cg board summary for adaptive timing/resource rules.

    The important change from the previous patch is that deck management is
    exposed as a learnable bucket in the Q key and as a soft pressure value here,
    not as a hard "deck <= 5 means never search" rule.  Search/hand-refresh is
    still allowed when it fixes a real board need.
    """
    yi = _current_player_index(observation)
    me = _player_state(observation, yi)
    state = cg_state_for_agent(observation)
    info = model.summarize_state(state)
    deck_count = _zone_len(me, "deck")
    hand_count = _zone_len(me, "hand")
    bench_count = _zone_len(me, "bench")
    board_complete = bool(info.has_crustle and info.has_munkidori)
    active_crustle = CRUSTLE in info.active_ids
    inplay = _inplay_cards(me)
    dwebble_count = sum(1 for c in inplay if _card_id(c) == DWEBBLE)
    crustle_count = sum(1 for c in inplay if _card_id(c) == CRUSTLE)
    munkidori_count = sum(1 for c in inplay if _card_id(c) == MUNKIDORI)
    hero_on_crustle = any(_card_id(c) == CRUSTLE and _has_tool(c, HERO_MANTLE) for c in inplay)
    munkidori_has_dark = _munkidori_has_dark(me)
    active_cards = _as_list(_dget(me, "active", []) or [])
    bench_cards = _as_list(_dget(me, "bench", []) or [])
    active_crustle_energy = 0
    for c in active_cards:
        if _card_id(c) == CRUSTLE:
            active_crustle_energy = max(active_crustle_energy, len(_attached_energy_ids(c)))
    line_energy_total = sum(len(_attached_energy_ids(c)) for c in inplay if _card_id(c) in {DWEBBLE, CRUSTLE})
    backup_line_needs_energy = any(
        _card_id(c) in {DWEBBLE, CRUSTLE} and len(_attached_energy_ids(c)) < 2
        for c in bench_cards
    )
    active_needs_energy = bool(active_crustle and active_crustle_energy < 3)

    # Soft deck pressure: 0 when the deck is still comfortable, approaches 1 as
    # the deck becomes very thin. It is a score modifier, not a hard rule.
    if deck_count <= 0:
        deck_pressure = 1.0
    else:
        deck_pressure = max(0.0, min(1.0, (12.0 - float(deck_count)) / 12.0))

    board_need = 0.0
    if not info.has_dwebble and not info.has_crustle:
        board_need += 3.0
    if info.has_dwebble and not info.has_crustle:
        board_need += 2.6
    if info.has_crustle and not hero_on_crustle:
        board_need += 2.0
    if not info.has_munkidori:
        board_need += 1.4
    if info.has_munkidori and not munkidori_has_dark:
        board_need += 1.6
    if board_complete and (dwebble_count + crustle_count) < 2:
        board_need += 1.4
    # Do not add pure energy-deficit to generic board_need.
    # Energy/back-up development is handled by attachment scoring, not by
    # increasing the value of generic search/hand-refresh cards.
    if bench_count <= 1:
        board_need += 1.2
    if info.behind_on_prizes:
        board_need += 0.8

    return {
        "info": info,
        "deck_count": deck_count,
        "hand_count": hand_count,
        "bench_count": bench_count,
        "board_complete": board_complete,
        "active_crustle": active_crustle,
        "dwebble_count": dwebble_count,
        "crustle_count": crustle_count,
        "munkidori_count": munkidori_count,
        "hero_on_crustle": hero_on_crustle,
        "munkidori_has_dark": munkidori_has_dark,
        "active_crustle_energy": active_crustle_energy,
        "line_energy_total": line_energy_total,
        "backup_line_needs_energy": backup_line_needs_energy,
        "active_needs_energy": active_needs_energy,
        "deck_pressure": deck_pressure,
        "board_need": board_need,
        "deck_bucket": IwapalaceRLAgent._deck_bucket(deck_count),
        "energy_attached": info.energy_attached is True,
        "behind_on_prizes": info.behind_on_prizes,
    }


def _is_good_energy_attach(feat: Mapping[str, Any], board: Mapping[str, Any] | None = None) -> bool:
    if feat.get("option_type") != OPTION_ATTACH:
        return False
    oc = feat.get("option_card")
    tc = feat.get("target_card")
    board = board or {}
    munkidori_has_dark = bool(board.get("munkidori_has_dark"))
    target_energy_count = int(feat.get("target_energy_count") or 0)
    if oc == DARK_ENERGY:
        # First Dark enables Munkidori.  After that, Dark can be used as a
        # colorless attack-cost attachment for the backup Dwebble/Crustle line.
        if tc == MUNKIDORI:
            return not munkidori_has_dark and target_energy_count == 0
        return tc in {DWEBBLE, CRUSTLE}
    if oc in {GRASS_ENERGY, MIST_ENERGY, GROW_GRASS_ENERGY}:
        return tc in {DWEBBLE, CRUSTLE}
    return False


def _is_hand_refresh_or_filter_card(card_id: int | None) -> bool:
    return card_id in HAND_REFRESH_OR_FILTER_CARDS


def _search_need_for_card(card_id: int | None, board: Mapping[str, Any]) -> float:
    """How much a selected/played card actually fixes the current board."""
    info: StateInfo = board.get("info")  # type: ignore[assignment]
    if info is None:
        return 0.0
    hero_on_crustle = bool(board.get("hero_on_crustle"))
    munkidori_has_dark = bool(board.get("munkidori_has_dark"))
    board_complete = bool(board.get("board_complete"))
    bench_count = int(board.get("bench_count") or 0)
    dwebble_count = int(board.get("dwebble_count") or 0)
    crustle_count = int(board.get("crustle_count") or 0)
    active_needs_energy = bool(board.get("active_needs_energy"))
    backup_line_needs_energy = bool(board.get("backup_line_needs_energy"))

    if card_id == DWEBBLE:
        if not info.has_dwebble and not info.has_crustle:
            return 3.5
        if board_complete and (dwebble_count + crustle_count) < 2:
            return 1.8
        if bench_count <= 1:
            return 1.1
        return 0.2
    if card_id == CRUSTLE:
        if info.has_dwebble and not info.has_crustle:
            return 3.8
        if board_complete and dwebble_count > 0:
            return 1.6
        return 0.4
    if card_id == MUNKIDORI:
        return 2.4 if not info.has_munkidori else 0.2
    if card_id == HERO_MANTLE:
        return 3.0 if info.has_crustle and not hero_on_crustle else 0.2
    if card_id == DARK_ENERGY:
        return 2.3 if info.has_munkidori and not munkidori_has_dark else (1.0 if info.has_crustle or info.has_dwebble else 0.2)
    if card_id in {GRASS_ENERGY, MIST_ENERGY, GROW_GRASS_ENERGY}:
        return 1.8 if (info.has_crustle or info.has_dwebble) else 0.4
    if card_id in {BOSS_ORDERS, JUMBO_ICE, NIGHT_STRETCHER}:
        return 1.4 if board_complete else 0.5
    if card_id == AKAMATSU:
        # Good when it genuinely fixes the first Dark Energy or a clearly
        # incomplete energy plan.  Once the board is complete, previous logs
        # showed Akamatsu became deck churn, so keep its generic search value low.
        if not bool(board.get("energy_attached")) and not munkidori_has_dark:
            return 2.0
        if (not board_complete) and (active_needs_energy or backup_line_needs_energy):
            return 1.4
        return 0.15 if board_complete else 0.9
    if card_id == TOUKO:
        if info.has_dwebble and not info.has_crustle:
            return 2.8
        if board_complete and dwebble_count > 0 and crustle_count < 2:
            return 1.2
        return 0.15 if board_complete else 1.0
    if card_id in {BUDDY_POFFIN, BUG_CATCHING_SET, POKEPAD, ULTRA_BALL, POKEGEAR, ROCKET_LAMBDA}:
        # Search/filter card as the played card: useful during setup, but once
        # board is online it must solve a real missing piece.  Energy deficits
        # are not enough reason to churn the deck.
        base_need = min(3.0, float(board.get("board_need") or 0.0))
        if board_complete:
            return min(base_need, 0.45) if base_need < 2.4 else base_need
        return base_need
    return 0.0


def _discipline_score(feat: Mapping[str, Any], board: Mapping[str, Any], *, good_energy_attach_available: bool) -> float:
    """Adaptive resource discipline.

    This is intentionally not a fixed rule such as "deck <= 5 never search".
    It scores each action by whether it improves the board now, whether energy
    attachment is still unused, and how high the deck-out pressure is.  The new
    deck bucket in the Q key lets training learn the exact boundary over time.
    """
    score = 0.0
    typ = feat.get("option_type")
    oc = feat.get("option_card")
    tc = feat.get("target_card")
    target_area = feat.get("target_area")
    board_complete = bool(board.get("board_complete"))
    active_crustle = bool(board.get("active_crustle"))
    energy_attached = bool(board.get("energy_attached"))
    behind = bool(board.get("behind_on_prizes"))
    bench_count = int(board.get("bench_count") or 0)
    deck_pressure = float(board.get("deck_pressure") or 0.0)
    board_need = float(board.get("board_need") or 0.0)
    active_needs_energy = bool(board.get("active_needs_energy"))
    backup_line_needs_energy = bool(board.get("backup_line_needs_energy"))
    need = _search_need_for_card(oc, board)

    if typ == OPTION_ATTACH and oc in ENERGY_CARDS:
        if not energy_attached:
            score += 5.0
        if _is_good_energy_attach(feat, board):
            score += 4.0 + min(2.0, need)
            if board_complete and target_area == AREA_BENCH:
                score += 3.5
            if board_complete and active_crustle and tc in {DWEBBLE, CRUSTLE}:
                score += 2.0
            if oc == DARK_ENERGY and tc in {DWEBBLE, CRUSTLE} and board.get("munkidori_has_dark"):
                score += 2.2
        else:
            # Do not over-penalize Dark-to-Crustle when Munkidori is already on,
            # but strongly reject repeated Dark on Munkidori or color energy on Munkidori.
            if oc == DARK_ENERGY and tc == MUNKIDORI and board.get("munkidori_has_dark"):
                score -= 10.0
            elif oc in {GRASS_ENERGY, MIST_ENERGY, GROW_GRASS_ENERGY} and tc == MUNKIDORI:
                score -= 12.0
            else:
                score -= 2.5

    if typ == OPTION_DISCARD:
        # Discard search/filter cards first when they no longer solve a real
        # board need.  Protect energy before attachment and all core pieces.
        if oc in ENERGY_CARDS:
            score -= 12.0 if not energy_attached else 3.0
        if oc in {DWEBBLE, CRUSTLE, MUNKIDORI, HERO_MANTLE}:
            score -= 11.0
        if oc in HAND_REFRESH_OR_FILTER_CARDS and (board_complete or deck_pressure > 0.45):
            score += 3.5
        if oc in {BOSS_ORDERS, NIGHT_STRETCHER, JUMBO_ICE} and deck_pressure < 0.55:
            score -= 2.5

    if typ == OPTION_PLAY and oc in RESOURCE_SPEND_CARDS:
        # Search/hand refresh is positive only when it fixes missing pieces.
        # This keeps Resource Pressure's deck-aware scoring, but rolls back the
        # over-search behaviour that raised select count and completed-board churn.
        score += need * 1.2
        if oc == AKAMATSU and not energy_attached and not good_energy_attach_available and (active_needs_energy or backup_line_needs_energy or not board.get("munkidori_has_dark")):
            score += 1.8
        if good_energy_attach_available and not energy_attached:
            score -= 9.0 if need < 2.8 else 4.0
        if board_complete and not behind and need < 2.0:
            score -= 7.5
        if board_complete and bench_count >= 3 and need < 2.0:
            score -= 4.0
        if board_complete and need < 1.0:
            score -= 3.5
        # Smooth deck-pressure penalty.  No hard deck<=5 ban: high real need can
        # still override this, but pointless cycling collapses as deck shrinks.
        score -= deck_pressure * (10.0 if need < 1.5 else 4.0 if need < 2.5 else 1.2)

    if typ == OPTION_PLAY and oc in HAND_DISCARD_COST_CARDS:
        if good_energy_attach_available and not energy_attached and need < 2.5:
            score -= 7.5
        if board_complete and need < 1.5:
            score -= 3.2

    if typ in {OPTION_CARD, OPTION_TOOL_CARD, OPTION_ENERGY_CARD, OPTION_SKILL}:
        # Search result selection: take cards that answer the board, not another
        # round of cycling.  Keep the pressure patch's wider churn-card list,
        # but reduce over-selection on completed boards.
        score += need * 2.0
        churn_cards = {BUDDY_POFFIN, BUG_CATCHING_SET, POKEPAD, POKEGEAR, ULTRA_BALL, ROCKET_LAMBDA, JUDGE, LILLIE_DETERMINATION, TOUKO, AKAMATSU}
        if board_complete and oc in churn_cards and need < 2.0:
            score -= 6.5
        if deck_pressure > 0.0 and oc in churn_cards:
            score -= deck_pressure * (8.0 if need < 1.5 else 4.0 if need < 2.5 else 1.0)
        if board_complete and oc in ENERGY_CARDS:
            score += 2.2
            if backup_line_needs_energy or active_needs_energy:
                score += 1.0
        if board_complete and oc in {DWEBBLE, CRUSTLE}:
            score += 1.8
        if board_complete and oc == BOSS_ORDERS and not behind:
            score += 1.2

    # End becomes more acceptable only when board need is low and deck pressure
    # is high.  This encourages natural deck-out avoidance without a hard rule.
    if typ == OPTION_END:
        if deck_pressure > 0.55 and board_need < 1.5:
            score += 2.0 * deck_pressure

    return score


def _discard_context_score(feat: Mapping[str, Any], board: Mapping[str, Any]) -> float:
    """Score cards during explicit discard/cost selections.

    The official cg log represents discard choices as OPTION_CARD with
    context == Discard, not OPTION_DISCARD.  Earlier resource discipline
    therefore failed to protect core cards during Ultra Ball / similar discard
    costs.  This layer is still adaptive: it protects resources according to
    board state instead of using a fixed deck-count ban.
    """
    oc = feat.get("option_card")
    if oc is None:
        return 0.0

    info: StateInfo | None = board.get("info")  # type: ignore[assignment]
    board_complete = bool(board.get("board_complete"))
    hero_on_crustle = bool(board.get("hero_on_crustle"))
    munkidori_has_dark = bool(board.get("munkidori_has_dark"))
    energy_attached = bool(board.get("energy_attached"))
    deck_pressure = float(board.get("deck_pressure") or 0.0)
    board_need = float(board.get("board_need") or 0.0)
    dwebble_count = int(board.get("dwebble_count") or 0)
    crustle_count = int(board.get("crustle_count") or 0)
    line_count = dwebble_count + crustle_count

    score = 0.0

    # Hero Mantle is one-of ACE SPEC.  If Crustle does not already have it,
    # discarding it is almost always unrecoverable tempo loss.
    if oc == HERO_MANTLE:
        score -= 32.0 if not hero_on_crustle else 6.0

    # Keep at least one Crustle line and preserve backup attackers.
    if oc == CRUSTLE:
        if info is not None and not info.has_crustle:
            score -= 32.0
        elif line_count < 2:
            score -= 26.0
        elif not board_complete:
            score -= 18.0
        else:
            # Even after setup, discarding Crustle was still too common.
            score -= 12.0
    if oc == DWEBBLE:
        if info is not None and not info.has_dwebble and not info.has_crustle:
            score -= 26.0
        elif line_count < 2:
            score -= 20.0
        else:
            score -= 7.0
    if oc == MUNKIDORI:
        if info is not None and not info.has_munkidori:
            score -= 18.0
        elif not munkidori_has_dark:
            score -= 10.0
        else:
            score -= 4.0

    # Energy is not a free discard while attachment for the turn remains or
    # while a backup Crustle is not charged.  Dark after Munkidori is online may
    # become attack-cost energy for the Crustle line, so still protect it.
    if oc in ENERGY_CARDS:
        if not energy_attached:
            score -= 18.0
        elif not board_complete:
            score -= 11.0
        elif line_count < 2:
            score -= 9.0
        else:
            score -= 4.0
        if oc == DARK_ENERGY and not munkidori_has_dark:
            score -= 6.0

    # Finishers / recovery should not be thrown away freely in mid-late game.
    if oc == BOSS_ORDERS:
        score -= 9.0 if board_complete else 4.0
    if oc == NIGHT_STRETCHER:
        score -= 9.0 if board_complete or deck_pressure > 0.35 else 3.0
    if oc == JUMBO_ICE:
        score -= 6.0 if board_complete else 2.0

    # Prefer discarding redundant filter cards and low-impact cards, especially
    # when board need is low or deck pressure is high.
    if oc in {POKEGEAR, BUDDY_POFFIN, BUG_CATCHING_SET, POKEPAD, ROCKET_LAMBDA, JUDGE, LILLIE_DETERMINATION}:
        score += 8.0 if board_complete else 3.0
        score += deck_pressure * 5.0
        if board_need < 1.5:
            score += 3.0
    if oc == ULTRA_BALL:
        score += 2.0 if board_complete else 0.5
    if oc == VITALITY_FOREST and (board_complete or (info is not None and info.has_crustle)):
        score += 3.0
    if oc == AIR_BALLOON:
        score += 2.0 if board_complete else 0.5
    if oc == AKAMATSU and board_complete and munkidori_has_dark:
        score += 4.0
    if oc == TOUKO and (board_complete or (info is not None and info.has_crustle)):
        score += 4.0

    return score



def _is_discard_context(context: Any) -> bool:
    if context in {8, CONTEXT_DAMAGE_COUNTER, "Discard", "discard", "DISCARD"}:
        # CONTEXT_DAMAGE_COUNTER can also use card-like selections, but it is
        # not a hand discard.  Keep numeric 8 for the observed cg discard
        # context and use strings for safer future logs.
        return context == 8 or (isinstance(context, str) and "discard" in context.lower())
    if isinstance(context, str):
        return "discard" in context.lower() or "トラッシュ" in context or "捨" in context
    return False


class OfficialCGPolicy:
    """Choose official cg select indices using the dedicated Iwapalace model."""

    def __init__(self, model: IwapalaceRLAgent) -> None:
        self.model = model

    def select_indices(self, observation: Any) -> list[int]:
        select = _dget(observation, "select", None)
        if not select:
            return []
        options = _dget(select, "option", []) or []
        min_count = _dget(select, "minCount", 0) or 0
        max_count = _dget(select, "maxCount", len(options)) or len(options)
        context = _dget(select, "context", None)
        select_type = _dget(select, "type", None)
        if not options or max_count <= 0:
            return []

        # Count selections. Draw/mulligan count should take the maximum,
        # but discard/cost counts should not blindly choose the largest number.
        if select_type == SELECT_COUNT:
            nums = []
            for i, opt in enumerate(options):
                n = _dget(opt, "number", None)
                nums.append((n if isinstance(n, int) else -1, i))
            if not nums:
                return [0]
            if context in {CONTEXT_DRAW_COUNT, CONTEXT_MULLIGAN}:
                nums.sort(reverse=True)
            else:
                nums.sort()
            return [nums[0][1]]

        # Yes/No policy. In cg mirror tests, choosing first is more stable
        # than forcing second: it wins the tempo race to Crustle and avoids
        # giving the mirror opponent the first evolution window.
        if select_type == SELECT_YES_NO:
            yes_idx = next((i for i, o in enumerate(options) if _dget(o, "type", None) == OPTION_YES), None)
            no_idx = next((i for i, o in enumerate(options) if _dget(o, "type", None) == OPTION_NO), None)
            if context == CONTEXT_IS_FIRST and yes_idx is not None:
                return [yes_idx]
            if yes_idx is not None:
                return [yes_idx]
            return [0]

        state = cg_state_for_agent(observation)
        board = _cg_board_features(observation, self.model)
        feats = [enrich_cg_option(observation, opt, i) for i, opt in enumerate(options)]
        for f in feats:
            if f.get("option_type") == OPTION_ATTACH:
                f["target_energy_count"] = _target_energy_count(observation, f)
        good_energy_attach_available = (not board.get("energy_attached")) and any(_is_good_energy_attach(f, board) for f in feats)
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for i, feat in enumerate(feats):
            score = self.model.score_action(state, feat)
            # v3 loss-log hard safety layer. These are attached-option
            # penalties, so they do not punish selecting the same card from a
            # search effect where target_card is None.
            if feat.get("option_type") == OPTION_ATTACH:
                oc = feat.get("option_card")
                tc = feat.get("target_card")
                if oc == AIR_BALLOON and tc in {DWEBBLE, CRUSTLE}:
                    score -= 30.0
                if oc == HERO_MANTLE and tc != CRUSTLE:
                    score -= 35.0
                if oc == DARK_ENERGY and tc == MUNKIDORI and board.get("munkidori_has_dark"):
                    score -= 24.0
                if oc == DARK_ENERGY and tc in {DWEBBLE, CRUSTLE} and board.get("munkidori_has_dark"):
                    score += 6.0
                if oc in {GRASS_ENERGY, MIST_ENERGY, GROW_GRASS_ENERGY} and tc == MUNKIDORI:
                    score -= 18.0
            # New discipline layer: attach energy before spending the hand,
            # grow the bench once the main board is complete, and stop needless
            # hand refresh/search loops.
            score += _discipline_score(feat, board, good_energy_attach_available=good_energy_attach_available)
            # Explicit discard/cost context.  In cg this is encoded as
            # OPTION_CARD + context=Discard, so it needs its own resource guard.
            if _is_discard_context(context):
                score += _discard_context_score(feat, board)
            # Extra official-context rules for multi-select setup.
            if context == CONTEXT_SETUP_ACTIVE:
                if feat.get("option_card") == DWEBBLE:
                    score += 20
                elif feat.get("option_card") == MUNKIDORI:
                    score += 5
            elif context == CONTEXT_SETUP_BENCH:
                if feat.get("option_card") in {DWEBBLE, MUNKIDORI}:
                    score += 12
            scored.append((score, i, feat))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Discard/cost selectors should choose only the required number of
        # cards.  If discard is optional (minCount == 0), choose nothing rather
        # than throwing away extra resources.
        if _is_discard_context(context):
            if min_count <= 0:
                return []
            return [i for _, i, _ in scored[:min_count]]

        # Single-choice selectors.
        if max_count == 1:
            return [scored[0][1]]

        selected: list[int] = []
        # Always satisfy min count.
        for _, i, _ in scored[:min_count]:
            selected.append(i)
        # Optional multi-select: add only clearly useful cards.
        for score, i, feat in scored[min_count:]:
            if len(selected) >= max_count:
                break
            core_cards = {DWEBBLE, CRUSTLE, MUNKIDORI, GRASS_ENERGY, DARK_ENERGY, MIST_ENERGY, GROW_GRASS_ENERGY, HERO_MANTLE}
            useful = feat.get("option_card") in core_cards
            # Permit high-scoring identified setup/search choices, but do not
            # multi-select unidentified generic actions just because old Q keys
            # gave card=None a high value.
            if feat.get("option_card") is not None and score > 6.0:
                useful = True
            if context == CONTEXT_SETUP_BENCH:
                useful = feat.get("option_card") in {DWEBBLE, MUNKIDORI}
            if feat.get("option_card") in RESOURCE_SPEND_CARDS:
                board = _cg_board_features(observation, self.model)
                need = _search_need_for_card(feat.get("option_card"), board)
                if board.get("board_complete") and board.get("bench_count", 0) >= 3 and need < 2.0:
                    useful = False
                if board.get("board_complete") and need < 1.6:
                    useful = False
                if float(board.get("deck_pressure") or 0.0) > 0.35 and need < 2.2:
                    useful = False
            if useful:
                selected.append(i)

        selected = selected[:max_count]
        if len(selected) < min_count:
            for _, i, _ in scored:
                if i not in selected:
                    selected.append(i)
                    if len(selected) >= min_count:
                        break
        return selected


class SubmittedIwapalaceAgent:
    """Callable object that supports both official cg and old local APIs."""

    def __init__(self) -> None:
        self.model = IwapalaceRLAgent(training=False)
        self.cg = OfficialCGPolicy(self.model)
        self.deck_name = self.model.deck_name

    def __call__(self, observation: Any, legal_actions: Iterable[Any] | None = None) -> Any:
        if legal_actions is None and isinstance(observation, Mapping) and "select" in observation:
            return self.cg.select_indices(observation)
        if legal_actions is None:
            return deck_list()
        return self.model.select_action(observation, legal_actions)

    def select_action(self, observation: Any, legal_actions: Iterable[Any] | None = None) -> Any:
        return self.__call__(observation, legal_actions)

    def act(self, observation: Any, legal_actions: Iterable[Any] | None = None) -> Any:
        return self.__call__(observation, legal_actions)

    def choose_action(self, observation: Any, legal_actions: Iterable[Any] | None = None) -> Any:
        return self.__call__(observation, legal_actions)

    def explain_plan(self) -> str:
        return self.model.explain_plan()


agent = SubmittedIwapalaceAgent()


def deck_list() -> list[int]:
    deck: list[int] = []
    for cid, count in DECKLIST.items():
        deck.extend([int(cid)] * int(count))
    return deck


def choose_action(observation: Any, legal_actions: Iterable[Any] | None = None) -> Any:
    return agent(observation, legal_actions)


def select_action(observation: Any, legal_actions: Iterable[Any] | None = None) -> Any:
    return agent(observation, legal_actions)


if __name__ == "__main__":
    print(agent.deck_name)
    print(agent.explain_plan())
