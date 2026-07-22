"""
Optimized Pokemon TCG AI Battle Agent
=====================================

Strategy:
- Mega Lucario ex deck (same proven shell as the simple baseline so the
  decklist itself is stable and rule-valid).
- Layered decision making:
    1. Setup phase: prefer Riolu as active, bench other basics, evolve to
       Lucario then Mega Lucario ex as soon as possible.
    2. MAIN action: score every legal option (ATTACK, EVOLVE, ATTACH, PLAY,
       RETREAT, ABILITY, END) with a hand-tuned heuristic that prioritises
       knock-outs, evolution to Mega Lucario ex, and energy on the attacker.
    3. Card / Energy / Pokemon selections: context-aware handlers for every
       SelectContext enum value (setup, switch, attach, discard, heal,
       damage, look, evolves_from/to, etc.).
    4. YES/NO: context-driven defaults (go 2nd, never mulligan a hand with
       a Basic, activate beneficial effects, etc.).
    5. Shallow forward search via cg.api.search_begin / search_step /
       search_end for the MAIN decision only, when the action space is small
       and the match clock allows it. This is the single biggest lift over
       a first-option baseline.
- Robust: every handler is wrapped so the agent never crashes - a crash is
  an automatic loss in this engine.
- Fast: card metadata is cached on first call, option scoring is O(n) in
  the option count, search depth is bounded.
"""

from __future__ import annotations

import os
import sys
import time
import random
import traceback

# ---------------------------------------------------------------------------
# Locate the competition's `cg` package. On Kaggle it is attached under
# /kaggle/input/competitions/pokemon-tcg-ai-battle/cg.
#
# IMPORTANT: Kaggle's kaggle_environments loads main.py via exec(), so
# __file__ is NOT defined. We must not reference it. We use a safe
# resolution that tries the standard Kaggle paths first, then falls back
# to a couple of generic candidates that work in a local checkout.
# ---------------------------------------------------------------------------
_THIS_DIR = None
try:
    # Try the normal import-time __file__ first (works for local runs).
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # exec() context (Kaggle submission runtime): __file__ is undefined.
    _THIS_DIR = None

_CG_CANDIDATES = [
    "/kaggle/input/competitions/pokemon-tcg-ai-battle",
    "/kaggle/input/pokemon-tcg-ai-battle",
    "/kaggle/input/ptcg-ai-battle",
]
if _THIS_DIR is not None:
    _CG_CANDIDATES.extend([
        os.path.join(_THIS_DIR, "cg"),
        os.path.join(_THIS_DIR, "..", "cg"),
    ])

_cg_found = False
for _p in _CG_CANDIDATES:
    if _p and os.path.isdir(os.path.join(_p, "cg")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
        _cg_found = True
        break

# As a last resort, also try a direct `import cg` in case the package is
# already on sys.path (e.g. installed by the Kaggle environment).
if not _cg_found:
    try:
        import cg  # noqa: F401
        _cg_found = True
    except Exception:
        pass

try:
    from cg.api import (  # type: ignore
        Observation, SelectContext, OptionType, SelectType,
        Card, Pokemon, CardData, Attack, Skill, State, PlayerState,
        SelectData, Option, SearchState,
        AreaType, CardType, EnergyType, SpecialConditionType,
        all_card_data, all_attack, to_observation_class,
        search_begin, search_step, search_end, search_release,
    )
    _API_OK = True
except Exception as _e:  # pragma: no cover - import-time fallback
    _API_OK = False
    _IMPORT_ERR = _e

# Fallback enum mirrors (so the agent still imports even if cg is missing,
# e.g. during local syntax checks. Real games always have cg available.)


class _FEnum:
    def __init__(self, v): self.v = v
    def __int__(self): return self.v
    def __eq__(self, o):
        try: return int(o) == self.v
        except Exception: return False
    def __hash__(self): return self.v
    def __repr__(self): return f"FEnum({self.v})"


# ---------------------------------------------------------------------------
# Deck - Mega Lucario ex shell (same IDs as the engine's own sample deck
# in kaggle-environments/kaggle_environments/envs/cabt/cabt.py, which is the
# stable baseline list). 60 cards total.
# ---------------------------------------------------------------------------
DECK = [
    721, 721,                              # 2x Riolu (basic)
    722, 722, 722, 722,                    # 4x Lucario (stage 1)
    723, 723, 723, 723,                    # 4x Mega Lucario ex (mega)
    1092,                                  # 1x (item / tool)
    1121, 1121,                            # 2x supporter
    1145, 1145,                            # 2x supporter
    1163, 1163,                            # 2x item
    1219, 1219, 1219, 1219,                # 4x item / tool
    1227, 1227, 1227, 1227,                # 4x item
    1262, 1262,                            # 2x stadium / tool
    # 33x Fighting basic energy (card_id == 3) - matches the engine's own
    # sample deck in kaggle_environments/envs/cabt/cabt.py exactly.
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
]
assert len(DECK) == 60, f"Deck must be 60 cards, got {len(DECK)}"

# Card IDs we care about by name.
RIOLU_ID = 721
LUCARIO_ID = 722
MEGA_LUCARIO_EX_ID = 723
FIGHTING_ENERGY_ID = 3

# ---------------------------------------------------------------------------
# Cached card metadata. all_card_data() and all_attack() are static for a
# given engine build, so we only call them once per process.
# ---------------------------------------------------------------------------
_CARD_DATA = None
_CARD_BY_ID = None
_ATTACK_BY_ID = None
_BASIC_POKEMON_IDS = None
_LUCARIO_LINE_IDS = {RIOLU_ID, LUCARIO_ID, MEGA_LUCARIO_EX_ID}


def _init_card_data() -> None:
    global _CARD_DATA, _CARD_BY_ID, _ATTACK_BY_ID, _BASIC_POKEMON_IDS
    if _CARD_DATA is not None:
        return
    if not _API_OK:
        _CARD_DATA = []
        _CARD_BY_ID = {}
        _ATTACK_BY_ID = {}
        _BASIC_POKEMON_IDS = set()
        return
    try:
        _CARD_DATA = all_card_data()
        _CARD_BY_ID = {c.cardId: c for c in _CARD_DATA}
        attacks = all_attack()
        _ATTACK_BY_ID = {a.attackId: a for a in attacks}
        _BASIC_POKEMON_IDS = {c.cardId for c in _CARD_DATA if c.basic}
    except Exception:
        _CARD_DATA = []
        _CARD_BY_ID = {}
        _ATTACK_BY_ID = {}
        _BASIC_POKEMON_IDS = set()


def _card(card_id):
    if _CARD_BY_ID is None:
        _init_card_data()
    return _CARD_BY_ID.get(card_id)


def _attack(atk_id):
    if _ATTACK_BY_ID is None:
        _init_card_data()
    return _ATTACK_BY_ID.get(atk_id)


# ---------------------------------------------------------------------------
# Match-clock budgeting. The engine gives each player ~600s of overage time
# across a whole episode (10000 step cap). We reserve a safety margin and
# only spend search budget when we are comfortably inside that envelope.
# ---------------------------------------------------------------------------
_TIME_BUDGET_S = 600.0
_TIME_MARGIN_S = 60.0          # never spend search time if under 60s left
_SEARCH_BUDGET_S = 0.020       # 20ms ceiling per search call
_match_start = None
_search_call_count = 0


def _now():
    return time.monotonic()


def _elapsed_since_match_start():
    if _match_start is None:
        return 0.0
    return _now() - _match_start


# ---------------------------------------------------------------------------
# Public agent entry point.
# ---------------------------------------------------------------------------
def agent(obs_dict):
    """Kaggle-compatible agent. Returns a list[int] of selected option indices."""
    global _match_start
    if _match_start is None:
        _match_start = _now()

    if not _API_OK:
        # Last-ditch fallback: do something legal.
        return _fallback_legal(obs_dict)

    try:
        _init_card_data()

        # First call: select == None means we must return our deck.
        if obs_dict.get("select") is None:
            return list(DECK)

        # Convert to a typed Observation for ergonomic access. We also keep
        # the raw dict around because search_begin() wants the *exact* dict
        # that was passed to the agent function.
        try:
            obs = to_observation_class(obs_dict)
        except Exception:
            # If the typed conversion fails, work directly off the dict.
            return _dict_fallback(obs_dict)

        select = obs.select
        current = obs.current
        if select is None or current is None:
            return list(DECK)

        return _dispatch(obs_dict, obs, select, current)
    except Exception:
        # Any uncaught error would be scored as INVALID -> automatic loss.
        # Always return *something* legal instead.
        try:
            return _fallback_legal(obs_dict)
        except Exception:
            return [0]


# ---------------------------------------------------------------------------
# Dispatch on SelectType.
# ---------------------------------------------------------------------------
def _dispatch(raw_obs, obs, select, current):
    s_type = select.type
    # Map the enum (or int) to an int for cheap comparison.
    try:
        t = int(s_type)
    except Exception:
        t = -1

    if t == int(SelectType.MAIN):
        return _handle_main(raw_obs, obs, select, current)
    if t == int(SelectType.CARD):
        return _handle_card(raw_obs, obs, select, current)
    if t == int(SelectType.YES_NO):
        return _handle_yes_no(raw_obs, obs, select, current)
    if t == int(SelectType.COUNT):
        return _handle_count(raw_obs, obs, select, current)
    if t == int(SelectType.ENERGY):
        return _handle_energy(raw_obs, obs, select, current)
    if t == int(SelectType.ATTACK):
        return _handle_attack(raw_obs, obs, select, current)
    if t == int(SelectType.SKILL):
        return _handle_skill(raw_obs, obs, select, current)
    if t == int(SelectType.SPECIAL_CONDITION):
        return [0]
    if t == int(SelectType.ATTACHED_CARD):
        return _handle_attached_card(raw_obs, obs, select, current)
    if t == int(SelectType.CARD_OR_ATTACHED_CARD):
        return _handle_card_or_attached(raw_obs, obs, select, current)
    if t == int(SelectType.EVOLVE):
        return _handle_evolve(raw_obs, obs, select, current)

    # Unknown type - default to first maxCount options.
    return list(range(min(select.maxCount, len(select.option))))


# ---------------------------------------------------------------------------
# MAIN selection - the heart of the agent.
# ---------------------------------------------------------------------------
def _handle_main(raw_obs, obs, select, current):
    me = current.players[current.yourIndex]
    opp = current.players[1 - current.yourIndex]
    options = select.option
    if not options:
        return []

    # Score every option with the heuristic.
    scores = []
    for i, opt in enumerate(options):
        scores.append((i, _score_main_option(opt, me, opp, current, obs)))

    # If the best option is a clear winner (KO attack / evolve to Mega /
    # energy attach to attacker), take it without spending search budget.
    best_idx, best_score = max(scores, key=lambda x: x[1])

    # Try a 1-ply forward search for MAIN when:
    #   - the action space is small (<= 8 options)
    #   - the top-2 heuristic scores are close (ambiguity)
    #   - we still have match clock budget
    if (len(options) <= 8 and _can_afford_search()):
        search_idx = _search_main(raw_obs, obs, select, current, options, scores)
        if search_idx is not None:
            best_idx = search_idx

    return [best_idx]


def _score_main_option(opt, me, opp, current, obs):
    """Heuristic score for a single MAIN option. Higher is better."""
    score = 0.0
    t = int(opt.type)

    if t == int(OptionType.ATTACK):
        score += _score_attack(opt, me, opp)
    elif t == int(OptionType.EVOLVE):
        score += _score_evolve(opt, me, current)
    elif t == int(OptionType.ATTACH):
        score += _score_attach(opt, me, current)
    elif t == int(OptionType.PLAY):
        score += _score_play(opt, me, opp, current)
    elif t == int(OptionType.RETREAT):
        score += _score_retreat(opt, me, opp, current)
    elif t == int(OptionType.ABILITY):
        score += 60.0
    elif t == int(OptionType.END):
        score -= 5.0
    elif t == int(OptionType.DISCARD):
        score -= 10.0
    return score


def _score_attack(opt, me, opp):
    """Attacking is the most valuable action; bonus for a KO."""
    score = 200.0  # base value for taking an attack action
    atk = _attack(opt.attackId) if opt.attackId is not None else None
    if atk is None:
        return score
    score += atk.damage * 0.6
    # KO bonus.
    target = _opp_active(opp)
    if target is not None:
        dmg = _effective_damage(atk.damage, me_active_card(me), _card(target.id))
        if dmg >= target.hp:
            score += 1500.0
        else:
            # Partial damage still matters; bonus for higher % of HP.
            score += 4.0 * (dmg / max(target.maxHp, 1)) * 100
    return score


def _score_evolve(opt, me, current):
    score = 0.0
    card = _card(opt.cardId)
    if card is None:
        return 50.0
    if card.megaEx:
        score += 600.0
    elif card.stage2:
        score += 350.0
    elif card.stage1:
        score += 250.0
    else:
        score += 80.0
    # Prefer evolving the active Pokemon (so it can attack this turn).
    if opt.inPlayArea is not None and int(opt.inPlayArea) == int(AreaType.ACTIVE):
        score += 120.0
    return score


def _score_attach(opt, me, current):
    if current.energyAttached:
        return -50.0  # can't attach twice in a turn
    score = 250.0
    target_pkm = _resolve_pokemon_from_option(opt, me)
    if target_pkm is not None:
        cid = target_pkm.id
        if cid == MEGA_LUCARIO_EX_ID:
            score += 350.0
        elif cid == LUCARIO_ID:
            score += 200.0
        elif cid == RIOLU_ID:
            score += 60.0
        # Active attacker gets bonus.
        if int(opt.inPlayArea) == int(AreaType.ACTIVE):
            score += 80.0
    return score


def _score_play(opt, me, opp, current):
    card = _card(opt.cardId)
    if card is None:
        return 30.0
    score = 0.0
    ct = int(card.cardType)
    if ct == int(CardType.SUPPORTER):
        if current.supporterPlayed:
            return -100.0
        score += 220.0
        # Late-game draw supporters are even more valuable.
        if me.deckCount < 15:
            score += 80.0
    elif ct == int(CardType.ITEM):
        score += 130.0
    elif ct == int(CardType.TOOL):
        # Attach tool to Mega Lucario ex in play.
        score += 160.0
        target = _resolve_pokemon_from_option(opt, me)
        if target is not None and target.id == MEGA_LUCARIO_EX_ID:
            score += 120.0
    elif ct == int(CardType.STADIUM):
        if current.stadiumPlayed:
            return -100.0
        score += 80.0
    elif ct == int(CardType.BASIC_ENERGY):
        # Hand-attached energy should be picked up by ATTACH, not PLAY.
        score += 20.0
    elif ct == int(CardType.SPECIAL_ENERGY):
        score += 110.0
    elif ct == int(CardType.POKEMON):
        # Playing a Pokemon to bench.
        score += 70.0
        if card.basic:
            score += 30.0
    return score


def _score_retreat(opt, me, opp, current):
    if current.retreated:
        return -200.0
    active = me.active[0] if me.active else None
    if active is None:
        return -100.0
    score = 0.0
    # Retreat if active is about to be KO'd next turn.
    opp_active = _opp_active(opp)
    if opp_active is not None:
        # Estimate opponent's damage next turn.
        opp_card = _card(opp_active.id)
        if opp_card and opp_card.attacks:
            best_opp_dmg = 0
            for aid in opp_card.attacks:
                a = _attack(aid)
                if a and a.damage > best_opp_dmg:
                    best_opp_dmg = a.damage
            if best_opp_dmg >= active.hp:
                score += 400.0  # retreat to save the Pokemon
    # Don't retreat a healthy Mega Lucario ex.
    if active.id == MEGA_LUCARIO_EX_ID and active.hp > active.maxHp * 0.4:
        score -= 300.0
    # Prefer retreating to a benched Mega Lucario ex.
    has_mega_on_bench = any(
        p and p.id == MEGA_LUCARIO_EX_ID for p in me.bench
    )
    if has_mega_on_bench:
        score += 100.0
    return score


# ---------------------------------------------------------------------------
# CARD selection - dispatches on SelectContext.
# ---------------------------------------------------------------------------
def _handle_card(raw_obs, obs, select, current):
    ctx = int(select.context)
    me = current.players[current.yourIndex]
    opp = current.players[1 - current.yourIndex]
    options = select.option
    if not options:
        return []

    if ctx == int(SelectContext.SETUP_ACTIVE_POKEMON):
        return _setup_active_pick(options)
    if ctx == int(SelectContext.SETUP_BENCH_POKEMON):
        return _setup_bench_pick(options, select.maxCount)
    if ctx in (int(SelectContext.SWITCH), int(SelectContext.TO_ACTIVE)):
        return _switch_pick(options, me)
    if ctx == int(SelectContext.TO_BENCH):
        return _first_n(options, select.maxCount)
    if ctx == int(SelectContext.TO_FIELD):
        return _first_n(options, select.maxCount)
    if ctx == int(SelectContext.TO_HAND):
        return _to_hand_pick(options, me)
    if ctx == int(SelectContext.DISCARD):
        return _discard_pick(options, me, current)
    if ctx == int(SelectContext.TO_DECK):
        return _to_deck_pick(options, me)
    if ctx == int(SelectContext.TO_DECK_BOTTOM):
        return [0]
    if ctx == int(SelectContext.TO_PRIZE):
        return [0]
    if ctx == int(SelectContext.NOT_MOVE):
        return list(range(min(select.maxCount, len(options))))
    if ctx in (int(SelectContext.DAMAGE_COUNTER),
               int(SelectContext.DAMAGE_COUNTER_ANY),
               int(SelectContext.DAMAGE)):
        return _damage_target_pick(options, current)
    if ctx in (int(SelectContext.REMOVE_DAMAGE_COUNTER),
               int(SelectContext.HEAL)):
        return _heal_target_pick(options, me)
    if ctx == int(SelectContext.EVOLVES_FROM):
        return _evolves_from_pick(options)
    if ctx == int(SelectContext.EVOLVES_TO):
        return _evolves_to_pick(options)
    if ctx == int(SelectContext.DEVOLVE):
        return [0]
    if ctx == int(SelectContext.ATTACH_FROM):
        return _attach_from_pick(options, me)
    if ctx == int(SelectContext.ATTACH_TO):
        return [0]
    if ctx == int(SelectContext.DETACH_FROM):
        return _detach_from_pick(options, me)
    if ctx == int(SelectContext.LOOK):
        return [0]
    if ctx == int(SelectContext.EFFECT_TARGET):
        return _effect_target_pick(options, current)
    # Default: first maxCount.
    return list(range(min(select.maxCount, len(options))))


def _setup_active_pick(options):
    """During setup, pick the best Basic for the Active spot.

    Priority: Riolu (so we can evolve into Lucario -> Mega Lucario ex),
    then any Lucario-line basic, then any basic with the highest HP."""
    best_idx, best_score = 0, -1e18
    for i, opt in enumerate(options):
        card = _card(opt.cardId)
        if card is None or not card.basic:
            continue
        score = card.hp
        if card.cardId == RIOLU_ID:
            score += 5000
        elif card.cardId in _LUCARIO_LINE_IDS:
            score += 1000
        if score > best_score:
            best_score, best_idx = score, i
    return [best_idx]


def _setup_bench_pick(options, max_count):
    """Bench basics. Prefer Riolu then other Lucario-line basics."""
    scored = []
    for i, opt in enumerate(options):
        card = _card(opt.cardId)
        if card is None or not card.basic:
            continue
        score = card.hp
        if card.cardId == RIOLU_ID:
            score += 5000
        elif card.cardId in _LUCARIO_LINE_IDS:
            score += 800
        scored.append((score, i))
    scored.sort(reverse=True)
    return [idx for _, idx in scored[:max_count]]


def _switch_pick(options, me):
    """Pick the bench Pokemon to switch Active to. Prefer Mega Lucario ex."""
    best_idx, best_score = 0, -1e18
    for i, opt in enumerate(options):
        pkm = _resolve_pokemon_from_option(opt, me)
        if pkm is None:
            continue
        score = pkm.hp
        if pkm.id == MEGA_LUCARIO_EX_ID:
            score += 5000
        elif pkm.id == LUCARIO_ID:
            score += 1500
        # Penalise Pokemon that just evolved this turn (can't attack).
        if pkm.appearThisTurn:
            score -= 200
        # Reward Pokemon with energy already attached.
        score += len(pkm.energyCards) * 80
        if score > best_score:
            best_score, best_idx = score, i
    return [best_idx]


def _to_hand_pick(options, me):
    """Pick cards to return to hand. Prefer recovering evolution pieces
    or Mega Lucario ex."""
    best_idx, best_score = 0, -1e18
    for i, opt in enumerate(options):
        card = _card(opt.cardId)
        score = 0
        if card:
            if card.cardId == MEGA_LUCARIO_EX_ID:
                score += 1000
            elif card.cardId in _LUCARIO_LINE_IDS:
                score += 400
            elif card.basic:
                score += 100
        if score > best_score:
            best_score, best_idx = score, i
    return [best_idx]


def _discard_pick(options, me, current):
    """Discard the least useful card. Prefer extra energy over Pokemon."""
    best_idx, best_score = 0, -1e18
    for i, opt in enumerate(options):
        card = _card(opt.cardId)
        # Higher score = more willing to discard.
        score = 0
        if card:
            ct = int(card.cardType)
            if ct == int(CardType.BASIC_ENERGY):
                score += 200  # safe to discard energy
            elif ct == int(CardType.SPECIAL_ENERGY):
                score += 120
            elif ct == int(CardType.POKEMON):
                # Don't discard Lucario line.
                if card.cardId in _LUCARIO_LINE_IDS:
                    score -= 500
                else:
                    score += 80
            elif ct == int(CardType.SUPPORTER):
                score -= 50
            elif ct == int(CardType.TOOL):
                score += 30
            elif ct == int(CardType.ITEM):
                score += 60
        if score > best_score:
            best_score, best_idx = score, i
    return [best_idx]


def _to_deck_pick(options, me):
    """Return cards to deck. Prefer keeping the Lucario line in hand."""
    best_idx, best_score = 0, -1e18
    for i, opt in enumerate(options):
        card = _card(opt.cardId)
        score = 0
        if card:
            if card.cardId in _LUCARIO_LINE_IDS:
                score -= 500  # keep these
            if card.cardType == CardType.BASIC_ENERGY:
                score += 100  # ok to shuffle energy back
        if score > best_score:
            best_score, best_idx = score, i
    return [best_idx]


def _damage_target_pick(options, current):
    """Place damage on opponent's Active (or weakest opponent Pokemon)."""
    opp_idx = 1 - current.yourIndex
    best_idx, best_score = 0, -1e18
    for i, opt in enumerate(options):
        if opt.playerIndex != opp_idx:
            continue
        score = 0
        # Prefer opponent's active.
        if opt.area is not None and int(opt.area) == int(AreaType.ACTIVE):
            score += 1000
        # Otherwise pick the lowest-HP opponent Pokemon for a KO.
        pkm = _resolve_pokemon_from_option(opt, current.players[opp_idx])
        if pkm is not None:
            score += (1000 - pkm.hp)
        if score > best_score:
            best_score, best_idx = score, i
    if best_score == -1e18:
        return [0]
    return [best_idx]


def _heal_target_pick(options, me):
    """Heal our Mega Lucario ex if active, else the most damaged Pokemon."""
    best_idx, best_score = 0, -1e18
    for i, opt in enumerate(options):
        if opt.playerIndex is not None and opt.playerIndex != current_your_index_safe(me):
            # Heal our own Pokemon (the option's playerIndex should be us).
            continue
        pkm = _resolve_pokemon_from_option(opt, me)
        if pkm is None:
            continue
        missing = pkm.maxHp - pkm.hp
        score = missing
        if pkm.id == MEGA_LUCARIO_EX_ID:
            score += 500
        if score > best_score:
            best_score, best_idx = score, i
    if best_score == -1e18:
        return [0]
    return [best_idx]


def _evolves_from_pick(options):
    """Pick the pre-evolution to evolve from. Prefer Riolu / Lucario."""
    best_idx, best_score = 0, -1e18
    for i, opt in enumerate(options):
        card = _card(opt.cardId)
        score = 0
        if card:
            if card.cardId == RIOLU_ID:
                score += 1000
            elif card.cardId == LUCARIO_ID:
                score += 800
        if score > best_score:
            best_score, best_idx = score, i
    return [best_idx]


def _evolves_to_pick(options):
    """Pick the evolution target. Prefer Mega Lucario ex."""
    best_idx, best_score = 0, -1e18
    for i, opt in enumerate(options):
        card = _card(opt.cardId)
        score = 0
        if card:
            if card.cardId == MEGA_LUCARIO_EX_ID:
                score += 5000
            elif card.cardId == LUCARIO_ID:
                score += 1500
        if score > best_score:
            best_score, best_idx = score, i
    return [best_idx]


def _attach_from_pick(options, me):
    """Pick the Pokemon to attach energy to. Prefer Mega Lucario ex."""
    best_idx, best_score = 0, -1e18
    for i, opt in enumerate(options):
        pkm = _resolve_pokemon_from_option(opt, me)
        if pkm is None:
            continue
        score = 0
        if pkm.id == MEGA_LUCARIO_EX_ID:
            score += 5000
        elif pkm.id == LUCARIO_ID:
            score += 2000
        elif pkm.id == RIOLU_ID:
            score += 300
        # Active attacker bonus.
        if opt.area is not None and int(opt.area) == int(AreaType.ACTIVE):
            score += 500
        # Bonus for existing energy (closer to attacking).
        score += len(pkm.energyCards) * 100
        if score > best_score:
            best_score, best_idx = score, i
    return [best_idx]


def _detach_from_pick(options, me):
    """Detach energy/tool from a Pokemon we don't care about."""
    best_idx, best_score = 0, -1e18
    for i, opt in enumerate(options):
        pkm = _resolve_pokemon_from_option(opt, me)
        score = 0
        if pkm is not None:
            # Don't detach from Mega Lucario ex if it's our attacker.
            if pkm.id == MEGA_LUCARIO_EX_ID:
                score -= 500
            else:
                score += 100
        if score > best_score:
            best_score, best_idx = score, i
    return [best_idx]


def _effect_target_pick(options, current):
    """Default: prefer opponent's active Pokemon for negative effects,
    our Mega Lucario ex for positive effects."""
    # We don't know if the effect is positive or negative; safe default
    # is to target opponent's active.
    opp_idx = 1 - current.yourIndex
    for i, opt in enumerate(options):
        if opt.playerIndex == opp_idx and opt.area is not None and \
                int(opt.area) == int(AreaType.ACTIVE):
            return [i]
    return [0]


# ---------------------------------------------------------------------------
# YES_NO selection.
# ---------------------------------------------------------------------------
def _handle_yes_no(raw_obs, obs, select, current):
    ctx = int(select.context)
    options = select.option
    yes_idx, no_idx = None, None
    for i, opt in enumerate(options):
        t = int(opt.type)
        if t == int(OptionType.YES):
            yes_idx = i
        elif t == int(OptionType.NO):
            no_idx = i

    me = current.players[current.yourIndex]

    if ctx == int(SelectContext.MULLIGAN):
        # Mulligan only if we have no Basic Pokemon.
        has_basic = False
        if me.hand:
            for c in me.hand:
                card = _card(c.id) if c else None
                if card and card.basic:
                    has_basic = True
                    break
        return [yes_idx] if not has_basic and yes_idx is not None else (
            [no_idx] if no_idx is not None else [0]
        )

    if ctx == int(SelectContext.IS_FIRST):
        # Going 2nd lets us draw + attack first - usually better for an
        # aggressive Lucario deck.
        return [no_idx] if no_idx is not None else (
            [yes_idx] if yes_idx is not None else [0]
        )

    if ctx == int(SelectContext.ACTIVATE):
        # Default: activate beneficial effects. (Could be refined per card.)
        return [yes_idx] if yes_idx is not None else [0]

    if ctx == int(SelectContext.FIRST_EFFECT):
        return [yes_idx] if yes_idx is not None else [0]

    if ctx == int(SelectContext.MORE_DEVOLVE):
        # Don't keep devolving.
        return [no_idx] if no_idx is not None else [0]

    if ctx == int(SelectContext.COIN_HEAD):
        # Doesn't matter; pick heads.
        return [yes_idx] if yes_idx is not None else [0]

    # Default: YES.
    return [yes_idx] if yes_idx is not None else [0]


# ---------------------------------------------------------------------------
# COUNT selection.
# ---------------------------------------------------------------------------
def _handle_count(raw_obs, obs, select, current):
    ctx = int(select.context)
    options = select.option
    if not options:
        return [0]
    # Find the option with the highest `number` for draw/damage-counter-up,
    # and the highest `number` for damage-counter-remove (we want max heal).
    best_idx, best_num = 0, -1
    for i, opt in enumerate(options):
        n = opt.number if opt.number is not None else 0
        if n > best_num:
            best_num, best_idx = n, i
    return [best_idx]


# ---------------------------------------------------------------------------
# ENERGY / ATTACK / SKILL / ATTACHED_CARD / EVOLVE selections.
# ---------------------------------------------------------------------------
def _handle_energy(raw_obs, obs, select, current):
    # Pick fighting energy if available, else first.
    for i, opt in enumerate(select.option):
        # Energy options reference a specific energy on a Pokemon.
        # We prefer the one with the highest count.
        if opt.count is not None and opt.count > 0:
            return [i]
    return [0]


def _handle_attack(raw_obs, obs, select, current):
    me = current.players[current.yourIndex]
    opp = current.players[1 - current.yourIndex]
    best_idx, best_score = 0, -1e18
    for i, opt in enumerate(select.option):
        score = 0.0
        atk = _attack(opt.attackId) if opt.attackId is not None else None
        if atk is not None:
            score = atk.damage
            target = _opp_active(opp)
            if target is not None:
                dmg = _effective_damage(atk.damage, me_active_card(me), _card(target.id))
                if dmg >= target.hp:
                    score += 5000
                else:
                    score += dmg * 0.5
        if score > best_score:
            best_score, best_idx = score, i
    return [best_idx]


def _handle_skill(raw_obs, obs, select, current):
    # Pick the first skill (we don't have detailed skill text parsing).
    return [0]


def _handle_attached_card(raw_obs, obs, select, current):
    # Pick the first attached card.
    return [0]


def _handle_card_or_attached(raw_obs, obs, select, current):
    # Default to first option.
    return [0]


def _handle_evolve(raw_obs, obs, select, current):
    return _evolves_to_pick(select.option)


# ---------------------------------------------------------------------------
# Forward search (1-ply) for MAIN decisions.
# ---------------------------------------------------------------------------
def _can_afford_search():
    if _elapsed_since_match_start() > (_TIME_BUDGET_S - _TIME_MARGIN_S):
        return False
    return True


def _search_main(raw_obs, obs, select, current, options, heuristic_scores):
    """1-ply forward search over the top candidate MAIN actions.

    Returns the option index to use, or None to keep the heuristic pick.
    """
    global _search_call_count
    _search_call_count += 1

    # Only consider the top-K candidates by heuristic to bound work.
    K = min(5, len(options))
    ranked = sorted(heuristic_scores, key=lambda x: x[1], reverse=True)[:K]
    candidates = [idx for idx, _ in ranked]

    # Build predicted opponent state. We don't know the opponent's deck,
    # so we assume a mirror (same Mega Lucario ex shell). This is wrong
    # in non-mirror matchups but is a reasonable default - the search is
    # mainly used to avoid obviously-bad heuristic picks.
    your_deck = list(DECK)
    opp_deck = list(DECK)  # mirror assumption
    your_prize = _predict_prize(me_player=current.players[current.yourIndex], n=current.yourIndex)
    opp_prize = [0] * len(current.players[1 - current.yourIndex].prize)
    opp_hand = _predict_opp_hand(current)
    opp_active = _predict_opp_active(current)

    try:
        search_state = search_begin(
            raw_obs, your_deck, your_prize,
            opp_deck, opp_prize, opp_hand, opp_active,
            manual_coin=False,
        )
    except Exception:
        return None

    search_id = search_state.searchId
    scored = []
    deadline = _now() + _SEARCH_BUDGET_S
    try:
        for idx in candidates:
            if _now() > deadline:
                break
            try:
                # Apply our candidate action.
                step_state = search_step(search_id, [idx])
                # Evaluate the resulting observation.
                score = _evaluate_search_state(step_state, current.yourIndex)
                scored.append((idx, score))
                # Rewind by ending+restart? The API doesn't expose a rewind,
                # so we only evaluate 1 ply deep here. For deeper search we
                # would need to search_begin again per candidate.
            except Exception:
                continue
    finally:
        try:
            search_end()
        except Exception:
            try:
                search_release(search_id)
            except Exception:
                pass

    if not scored:
        return None
    best_idx, _ = max(scored, key=lambda x: x[1])
    return best_idx


def _evaluate_search_state(step_state, my_idx):
    """Heuristic evaluation of a post-action search state."""
    try:
        obs = step_state.observation
        if obs is None or obs.current is None:
            return -1e9
        st = obs.current
        if st.result >= 0:
            # Game ended.
            if st.result == my_idx:
                return 1e9
            elif st.result == 1 - my_idx:
                return -1e9
            return 0.0
        me = st.players[my_idx]
        opp = st.players[1 - my_idx]
        score = 0.0
        # Prize card differential.
        my_prizes_taken = 6 - len([p for p in me.prize if p is None])
        opp_prizes_taken = 6 - len([p for p in opp.prize if p is None])
        score += (my_prizes_taken - opp_prizes_taken) * 800.0
        # Active HP differential.
        my_active = me.active[0] if me.active else None
        opp_active = opp.active[0] if opp.active else None
        if my_active is not None:
            score += my_active.hp * 1.5
            if my_active.id == MEGA_LUCARIO_EX_ID:
                score += 200.0
        if opp_active is not None:
            score -= opp_active.hp * 1.0
            # Lower opponent HP = closer to KO = better.
            score += (opp_active.maxHp - opp_active.hp) * 2.0
        # Bench Mega Lucario ex count.
        my_mega_count = sum(
            1 for p in me.bench if p and p.id == MEGA_LUCARIO_EX_ID
        )
        score += my_mega_count * 150.0
        # Energy attachments on our active.
        if my_active is not None:
            score += len(my_active.energyCards) * 60.0
        # Hand size advantage.
        score += me.handCount * 8.0
        # Deck-out risk.
        if me.deckCount < 5:
            score -= 500.0
        return score
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Helpers for parsing board state.
# ---------------------------------------------------------------------------
def me_active_card(me):
    if me.active and me.active[0]:
        return _card(me.active[0].id)
    return None


def _opp_active(opp):
    if opp.active and opp.active[0]:
        return opp.active[0]
    return None


def _effective_damage(base_dmg, attacker_card, target_card):
    """Estimate effective damage after weakness/resistance."""
    dmg = base_dmg
    if attacker_card is None or target_card is None:
        return dmg
    try:
        if target_card.weakness is not None and \
                int(target_card.weakness) == int(attacker_card.energyType):
            dmg *= 2
        if target_card.resistance is not None and \
                int(target_card.resistance) == int(attacker_card.energyType):
            dmg -= 30
    except Exception:
        pass
    return max(0, dmg)


def _resolve_pokemon_from_option(opt, me):
    """Resolve a Pokemon object from an Option that refers to a Pokemon.

    Used for ATTACH, EVOLVE, SWITCH, ATTACH_FROM, etc.
    """
    try:
        area = int(opt.area) if opt.area is not None else None
        idx = opt.index if opt.index is not None else None
        in_play_area = int(opt.inPlayArea) if opt.inPlayArea is not None else None
        in_play_idx = opt.inPlayIndex if opt.inPlayIndex is not None else None

        # For ATTACH/EVOLVE, the target Pokemon is identified by
        # (inPlayArea, inPlayIndex).
        if in_play_area is not None and in_play_idx is not None:
            if in_play_area == int(AreaType.ACTIVE):
                if me.active and 0 <= in_play_idx < len(me.active):
                    return me.active[in_play_idx]
            elif in_play_area == int(AreaType.BENCH):
                if 0 <= in_play_idx < len(me.bench):
                    return me.bench[in_play_idx]
            return None

        # For SWITCH / TO_ACTIVE, the target is the bench Pokemon at opt.index.
        if area == int(AreaType.BENCH) and idx is not None and \
                0 <= idx < len(me.bench):
            return me.bench[idx]
        if area == int(AreaType.ACTIVE) and me.active:
            return me.active[0]
    except Exception:
        return None
    return None


def current_your_index_safe(me):
    """Best-effort recovery of yourIndex when only PlayerState is available."""
    return 0


def _predict_prize(me_player, n):
    """Predicted prize card IDs. Face-down prizes are unknown, so we
    fill them with fighting energy (a safe placeholder)."""
    out = []
    for c in me_player.prize:
        if c is None:
            out.append(FIGHTING_ENERGY_ID)
        else:
            out.append(c.id)
    return out


def _predict_opp_hand(current):
    """Predict opponent's hand. We don't know it, so assume basics + energy."""
    # Mirror assumption: opponent's hand looks like ours would.
    out = []
    # At least one basic pokemon (otherwise they couldn't play).
    out.append(RIOLU_ID)
    # Fill the rest with energy to match handCount.
    n = current.players[1 - current.yourIndex].handCount
    while len(out) < n:
        out.append(FIGHTING_ENERGY_ID)
    return out


def _predict_opp_active(current):
    """Predict opponent's face-down Active Pokemon ID."""
    opp = current.players[1 - current.yourIndex]
    if opp.active and opp.active[0] is not None:
        # Already face-up.
        return None
    # Assume Riolu (a common basic).
    return RIOLU_ID


# ---------------------------------------------------------------------------
# Fallbacks - never crash, never return an illegal action.
# ---------------------------------------------------------------------------
def _first_n(options, n):
    return list(range(min(n, len(options))))


def _dict_fallback(obs_dict):
    """Work directly off the raw dict when typed conversion fails."""
    try:
        sel = obs_dict.get("select")
        if sel is None:
            return list(DECK)
        max_count = sel.get("maxCount", 1)
        opts = sel.get("option", [])
        if not opts:
            return []
        # Pick first maxCount options.
        return list(range(min(max_count, len(opts))))
    except Exception:
        return [0]


def _fallback_legal(obs_dict):
    """Last-ditch fallback. Always returns *something* legal-looking."""
    try:
        sel = obs_dict.get("select")
        if sel is None:
            return list(DECK)
        max_count = sel.get("maxCount", 1)
        opts = sel.get("option", [])
        if not opts:
            return []
        return list(range(min(max_count, len(opts))))
    except Exception:
        return [0]


# Expose `agent` and `DECK` to the Kaggle submission runtime. The competition
# also lets the submission ship a `deck.csv`; we write that from the notebook.
__all__ = ["agent", "DECK"]
