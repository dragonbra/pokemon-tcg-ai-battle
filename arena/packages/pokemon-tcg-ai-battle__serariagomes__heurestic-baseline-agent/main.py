"""
Pokémon TCG AI Battle Challenge — heuristic agent (baseline v1).

Submission entrypoint: the Kaggle runner imports `agent` from this file.
The agent receives an observation dict each turn and returns a list of
option indices (see https://matsuoinstitute.github.io/cabt/).

Design:
  * Deck selection step (obs["select"] is None) -> return deck from deck.csv
  * MAIN selections    -> priority-scored actions (evolve > play > attach > ability > attack > end)
  * CARD selections    -> context-aware scoring (setup, discard, heal, damage targets, deck search...)
  * YES/NO, COUNT, ENERGY, SPECIAL_CONDITION -> sensible fixed policies

Card knowledge comes from cards.json / attacks.json (dumped from the engine's
AllCard/AllAttack). Everything degrades gracefully if the DBs are missing.
"""

import json
import os
import random

# ----------------------------------------------------------------------------
# Constants mirroring the cabt enums (ints, so no engine import needed)
# ----------------------------------------------------------------------------

# AreaType
AREA_DECK, AREA_HAND, AREA_DISCARD, AREA_ACTIVE, AREA_BENCH = 1, 2, 3, 4, 5
AREA_PRIZE, AREA_STADIUM, AREA_LOOKING = 6, 7, 12

# CardType
CT_POKEMON, CT_ITEM, CT_TOOL, CT_SUPPORTER, CT_STADIUM = 0, 1, 2, 3, 4
CT_BASIC_ENERGY, CT_SPECIAL_ENERGY = 5, 6

# SelectType
ST_MAIN, ST_CARD, ST_ATTACHED_CARD, ST_CARD_OR_ATTACHED = 0, 1, 2, 3
ST_ENERGY, ST_SKILL, ST_ATTACK, ST_EVOLVE, ST_COUNT, ST_YES_NO, ST_SPECIAL = 4, 5, 6, 7, 8, 9, 10

# OptionType
OT_NUMBER, OT_YES, OT_NO, OT_CARD, OT_TOOL_CARD, OT_ENERGY_CARD, OT_ENERGY = 0, 1, 2, 3, 4, 5, 6
OT_PLAY, OT_ATTACH, OT_EVOLVE, OT_ABILITY, OT_DISCARD, OT_RETREAT = 7, 8, 9, 10, 11, 12
OT_ATTACK, OT_END, OT_SKILL, OT_SPECIAL_CONDITION = 13, 14, 15, 16

# SelectContext (only the ones we branch on)
CTX_MAIN = 0
CTX_SETUP_ACTIVE, CTX_SETUP_BENCH = 1, 2
CTX_SWITCH, CTX_TO_ACTIVE, CTX_TO_BENCH, CTX_TO_FIELD, CTX_TO_HAND = 3, 4, 5, 6, 7
CTX_DISCARD, CTX_TO_DECK, CTX_TO_DECK_BOTTOM, CTX_TO_PRIZE, CTX_NOT_MOVE = 8, 9, 10, 11, 12
CTX_DMG_COUNTER, CTX_DMG_COUNTER_ANY, CTX_DAMAGE = 13, 14, 15
CTX_REMOVE_DMG, CTX_HEAL = 16, 17
CTX_EVOLVES_FROM, CTX_EVOLVES_TO = 18, 19
CTX_ATTACH_FROM, CTX_ATTACH_TO = 21, 22
CTX_EFFECT_TARGET = 25
CTX_DISCARD_ENERGY = 30
CTX_IS_FIRST, CTX_MULLIGAN, CTX_ACTIVATE, CTX_FIRST_EFFECT, CTX_MORE_DEVOLVE = 41, 42, 43, 44, 45
CTX_COIN_HEAD = 46
CTX_AFFECT_COND, CTX_RECOVER_COND = 47, 48

# SpecialConditionType priority when afflicting the opponent (higher = better)
AFFLICT_PRIORITY = {3: 5, 2: 4, 4: 3, 0: 2, 1: 1}  # paralyze > sleep > confuse > poison > burn

# ----------------------------------------------------------------------------
# Static data: deck + card database (all optional, with graceful fallback)
# ----------------------------------------------------------------------------

_BASE_DIRS = [
    os.path.dirname(os.path.abspath(__file__)),
    "/kaggle_simulations/agent",
    os.getcwd(),
]


def _find(fname):
    for d in _BASE_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return None


def _load_deck():
    p = _find("deck.csv")
    if p:
        with open(p) as f:
            deck = [int(line.strip()) for line in f if line.strip()]
        if len(deck) == 60:
            return deck
    # Fallback: engine sample deck (Kyogre / Mega Abomasnow ex + Water energy)
    return ([721] * 2 + [722] * 4 + [723] * 4 + [1092] + [1121] * 2 + [1145] * 2
            + [1163] * 2 + [1219] * 4 + [1227] * 4 + [1262] * 2 + [3] * 33)


def _load_json(fname):
    p = _find(fname)
    if not p:
        return []
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return []


DECK = _load_deck()
CARD_DB = {c["cardId"]: c for c in _load_json("cards.json")}
ATTACK_DB = {a["attackId"]: a for a in _load_json("attacks.json")}

# per-episode safety counters (module state persists within one episode process)
_ability_uses = {}

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _card(card_id):
    return CARD_DB.get(card_id)


def _card_value(card_id):
    """Generic 'how much do I want this card' score (searching / keeping)."""
    cd = _card(card_id)
    if not cd:
        return 2.0
    t = cd["cardType"]
    if t == CT_POKEMON:
        v = 5.0 + cd.get("hp", 0) / 50.0
        if cd.get("ex"):
            v += 2.0
        if cd.get("basic"):
            v += 0.5
        return v
    if t == CT_SUPPORTER:
        return 5.0
    if t == CT_ITEM:
        return 4.0
    if t == CT_TOOL:
        return 3.0
    if t == CT_STADIUM:
        return 2.5
    return 1.5  # energy is replaceable


def _me(cur):
    return cur["yourIndex"] if cur else 0


def _resolve_card_id(obs, opt):
    """Best-effort: figure out which cardId an Option refers to."""
    if opt.get("cardId") is not None:
        return opt["cardId"]
    cur = obs.get("current")
    sel = obs.get("select") or {}
    area, idx = opt.get("area"), opt.get("index")
    pidx = opt.get("playerIndex")
    if area is None or idx is None:
        return None
    if area == AREA_DECK:
        deck = sel.get("deck")
        if deck and 0 <= idx < len(deck):
            return deck[idx].get("id")
        return None
    if not cur:
        return None
    if area == AREA_LOOKING:
        looking = cur.get("looking")
        if looking and 0 <= idx < len(looking) and looking[idx]:
            return looking[idx].get("id")
        return None
    if pidx is None:
        pidx = _me(cur)
    try:
        player = cur["players"][pidx]
    except Exception:
        return None
    if area == AREA_HAND:
        hand = player.get("hand")
        if hand and 0 <= idx < len(hand):
            return hand[idx].get("id")
    elif area == AREA_DISCARD:
        pile = player.get("discard") or []
        if 0 <= idx < len(pile):
            return pile[idx].get("id")
    elif area == AREA_ACTIVE:
        act = player.get("active") or []
        if act and act[0]:
            return act[0].get("id")
    elif area == AREA_BENCH:
        bench = player.get("bench") or []
        if 0 <= idx < len(bench) and bench[idx]:
            return bench[idx].get("id")
    return None


def _resolve_pokemon(obs, opt):
    """Return the in-play Pokemon dict an option points to, if any."""
    cur = obs.get("current")
    if not cur:
        return None
    area, idx = opt.get("area"), opt.get("index")
    pidx = opt.get("playerIndex")
    if pidx is None:
        pidx = _me(cur)
    try:
        player = cur["players"][pidx]
    except Exception:
        return None
    if area == AREA_ACTIVE:
        act = player.get("active") or []
        return act[0] if act else None
    if area == AREA_BENCH:
        bench = player.get("bench") or []
        return bench[idx] if 0 <= idx < len(bench) else None
    return None


def _pick_top(options, scores, k):
    order = sorted(range(len(options)), key=lambda i: -scores[i])
    return [options[i]["_i"] for i in order[:k]]


# ----------------------------------------------------------------------------
# MAIN action selection
# ----------------------------------------------------------------------------


def _score_main_option(obs, opt):
    cur = obs.get("current")
    me = _me(cur)
    t = opt["type"]

    if t == OT_EVOLVE:
        return 90.0
    if t == OT_PLAY:
        cid = _resolve_card_id(obs, {"area": AREA_HAND, "index": opt.get("index"),
                                     "playerIndex": me})
        cd = _card(cid)
        if cd:
            ct = cd["cardType"]
            if ct == CT_POKEMON:
                return 82.0 + cd.get("hp", 0) / 100.0
            if ct == CT_SUPPORTER:
                return 70.0
            if ct == CT_ITEM:
                return 65.0
            if ct == CT_STADIUM:
                return 40.0
        return 60.0
    if t == OT_ATTACH:
        # prefer powering up the active Pokémon
        return 78.0 if opt.get("inPlayArea") == AREA_ACTIVE else 62.0
    if t == OT_ABILITY:
        # loop guard: don't spam the same ability endlessly in one turn
        turn = cur.get("turn", -1) if cur else -1
        key = (turn, opt.get("area"), opt.get("index"))
        if _ability_uses.get(key, 0) >= 3:
            return 0.5
        return 72.0
    if t == OT_ATTACK:
        atk = ATTACK_DB.get(opt.get("attackId"))
        dmg = atk.get("damage", 0) if atk else 0
        return 20.0 + dmg / 100.0  # below all setup actions, above END/retreat
    if t == OT_RETREAT:
        # retreat only if the active Pokémon is nearly dead and a healthier
        # benched Pokémon exists
        try:
            player = cur["players"][me]
            act = (player.get("active") or [None])[0]
            if act and act["maxHp"] and act["hp"] / act["maxHp"] < 0.25:
                bench = [b for b in (player.get("bench") or []) if b]
                if any(b["hp"] > act["hp"] for b in bench):
                    return 55.0
        except Exception:
            pass
        return 0.8
    if t == OT_DISCARD:
        return 5.0
    if t == OT_END:
        return 0.0
    return 10.0


# ----------------------------------------------------------------------------
# CARD selection scoring per context
# ----------------------------------------------------------------------------

# contexts where picking MORE is good for us
_GREEDY_CTX = {
    CTX_SETUP_BENCH, CTX_TO_HAND, CTX_TO_FIELD, CTX_HEAL, CTX_REMOVE_DMG,
    CTX_DMG_COUNTER, CTX_DMG_COUNTER_ANY, CTX_DAMAGE, CTX_LOOK if (CTX_LOOK := 24) else 24,
}
# contexts where each pick costs us something -> take the minimum required
_COSTLY_CTX = {
    CTX_DISCARD, CTX_TO_DECK, CTX_TO_DECK_BOTTOM, CTX_TO_PRIZE,
    CTX_DISCARD_ENERGY, 26, 27, 29,  # discard energy/tool/attached
}


def _score_card_option(obs, opt, ctx):
    cur = obs.get("current")
    me = _me(cur)
    opp_owned = opt.get("playerIndex") is not None and opt.get("playerIndex") != me

    # --- targets on the board ---
    pkm = _resolve_pokemon(obs, opt)
    if ctx in (CTX_SETUP_ACTIVE, CTX_SETUP_BENCH, CTX_EVOLVES_TO, CTX_TO_FIELD):
        cid = _resolve_card_id(obs, opt)
        cd = _card(cid)
        if cd:
            return cd.get("hp", 0) + (20 if cd.get("ex") else 0)
        return 10.0
    if ctx in (CTX_SWITCH, CTX_TO_ACTIVE):
        if pkm:
            return pkm.get("hp", 0) + 15.0 * len(pkm.get("energies") or [])
        return 1.0
    if ctx in (CTX_DAMAGE, CTX_DMG_COUNTER, CTX_DMG_COUNTER_ANY):
        # hit the opponent's most killable target; never our own if avoidable
        if pkm:
            base = 500.0 - pkm.get("hp", 0)
            return base + (300.0 if opp_owned else -300.0)
        return 0.0
    if ctx in (CTX_HEAL, CTX_REMOVE_DMG):
        if pkm:
            dmg_taken = pkm.get("maxHp", 0) - pkm.get("hp", 0)
            return dmg_taken + (100.0 if not opp_owned else -100.0)
        return 0.0

    # --- picking cards (hand / deck / discard) ---
    cid = _resolve_card_id(obs, opt)
    val = _card_value(cid)
    if ctx in _COSTLY_CTX:
        return -val  # discard the least valuable
    return val  # searching / drawing: take the most valuable


# ----------------------------------------------------------------------------
# The agent
# ----------------------------------------------------------------------------


def agent(obs, config=None):
    try:
        return _agent(obs)
    except Exception:
        # never crash: fall back to a legal random choice
        sel = obs.get("select")
        if sel is None:
            return DECK
        n, mx = len(sel["option"]), sel["maxCount"]
        return random.sample(range(n), min(mx, n))


def _agent(obs):
    sel = obs.get("select")
    if sel is None:                      # deck submission step
        return list(DECK)

    options = sel["option"]
    for i, o in enumerate(options):      # remember original indices
        o["_i"] = i
    min_c, max_c = sel.get("minCount", 1), sel.get("maxCount", 1)
    stype, ctx = sel.get("type", ST_MAIN), sel.get("context", CTX_MAIN)

    # ---- YES / NO ----
    if stype == ST_YES_NO:
        want_yes = True
        if ctx == CTX_MULLIGAN:
            want_yes = False             # keep our hand when we get a choice
        elif ctx == CTX_MORE_DEVOLVE:
            want_yes = False
        target = OT_YES if want_yes else OT_NO
        for o in options:
            if o["type"] == target:
                return [o["_i"]]
        return [options[0]["_i"]]

    # ---- COUNT: take the max (draw as much as possible etc.) ----
    if stype == ST_COUNT:
        best = max(options, key=lambda o: o.get("number") or 0)
        return [best["_i"]]

    # ---- SPECIAL CONDITION ----
    if stype == ST_SPECIAL:
        if ctx == CTX_RECOVER_COND:
            # recover the most crippling condition first
            best = max(options, key=lambda o: AFFLICT_PRIORITY.get(o.get("specialConditionType"), 0))
        else:
            best = max(options, key=lambda o: AFFLICT_PRIORITY.get(o.get("specialConditionType"), 0))
        return [best["_i"]]

    # ---- MAIN menu ----
    if stype == ST_MAIN:
        scores = [_score_main_option(obs, o) for o in options]
        pick = _pick_top(options, scores, 1)
        chosen = options[[o["_i"] for o in options].index(pick[0])]
        if chosen["type"] == OT_ABILITY:
            cur = obs.get("current")
            turn = cur.get("turn", -1) if cur else -1
            key = (turn, chosen.get("area"), chosen.get("index"))
            _ability_uses[key] = _ability_uses.get(key, 0) + 1
        return pick

    # ---- ATTACK choice: highest raw damage ----
    if stype == ST_ATTACK:
        def dmg(o):
            a = ATTACK_DB.get(o.get("attackId"))
            return a.get("damage", 0) if a else 0
        best = max(options, key=dmg)
        return [best["_i"]]

    # ---- everything card-like ----
    scores = [_score_card_option(obs, o, ctx) for o in options]
    k = min_c if ctx in _COSTLY_CTX else max_c
    k = max(min_c, min(k, max_c, len(options)))
    return _pick_top(options, scores, k)
