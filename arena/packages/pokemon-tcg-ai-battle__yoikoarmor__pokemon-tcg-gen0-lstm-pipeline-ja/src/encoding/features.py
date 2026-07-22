"""Feature builders. Pure numpy -- no torch dependency."""
from dataclasses import dataclass
import numpy as np

from cg.api import OptionType, AreaType
from src import engine

MAX_OPTIONS = 64
_NUM_OPTION_TYPES = 17  # OptionType 0..16
_NUM_CARD_TYPES = 7     # CardType 0..6

# role sets (mirror Ver5's; let the net learn the heuristic's card priorities)
_GUST_IDS = {1182, 1124, 1088, 1143}
_DRAW_SUP_IDS = {1192, 1227, 1224, 1236}
_DMG_BOOST_IDS = {1141}
_HP_TOOL_IDS = {1159, 1158}
_KEY_CARD_IDS = [
    6, 20,                 # Fighting energies
    673, 674, 675, 676, 677, 678,  # Mega Lucario package
    1102, 1123, 1141, 1142, 1152, 1159, 1182,
    1197, 1213, 1227, 1229, 1252,
    344, 345,              # Dwebble / Crustle
    119, 120, 121,         # Dragapult line
]
_KEY_ATTACK_IDS = [978, 979, 980, 981, 982, 983, 153, 154, 478, 479]
_MY_BOARD_IDS = [673, 674, 675, 676, 677, 678]
_OPP_ACTIVE_IDS = [344, 345, 119, 120, 121, 677, 678]


def _ref_card_id(op, me_p):
    """CardData id the option refers to (PLAY/ATTACH/EVOLVE/CARD source), or None."""
    if me_p is None or op.index is None:
        return None
    # PLAY has only `index` (hand); ATTACH/EVOLVE/CARD carry an explicit area
    if op.type in (OptionType.PLAY, OptionType.ATTACH, OptionType.EVOLVE):
        return me_p.hand[op.index].id if (me_p.hand and op.index < len(me_p.hand)) else None
    if op.area == AreaType.HAND:
        return me_p.hand[op.index].id if (me_p.hand and op.index < len(me_p.hand)) else None
    if op.area == AreaType.BENCH:
        return me_p.bench[op.index].id if op.index < len(me_p.bench) else None
    if op.area == AreaType.ACTIVE:
        return me_p.active[0].id if (me_p.active and me_p.active[0]) else None
    return None


def _card_role_features(cid):
    """Card identity and coarse role features."""
    out = [0.0] * (_NUM_CARD_TYPES + 4)
    cd = engine.card_db().get(cid) if cid is not None else None
    if cd is not None and 0 <= int(cd.cardType) < _NUM_CARD_TYPES:
        out[int(cd.cardType)] = 1.0
    base = _NUM_CARD_TYPES
    out[base + 0] = 1.0 if cid in _GUST_IDS else 0.0
    out[base + 1] = 1.0 if cid in _DRAW_SUP_IDS else 0.0
    out[base + 2] = 1.0 if cid in _DMG_BOOST_IDS else 0.0
    out[base + 3] = 1.0 if cid in _HP_TOOL_IDS else 0.0
    out += [
        _ratio(cid, 1300),
        _ratio(cd.hp, 340) if cd is not None else 0.0,
        1.0 if (cd is not None and cd.basic) else 0.0,
        1.0 if (cd is not None and cd.stage1) else 0.0,
        1.0 if (cd is not None and cd.stage2) else 0.0,
        1.0 if (cd is not None and cd.ex) else 0.0,
        1.0 if (cd is not None and cd.megaEx) else 0.0,
    ]
    out += [1.0 if cid == key else 0.0 for key in _KEY_CARD_IDS]
    return out


def _prize_of(cd):
    if cd is None:
        return 1
    return 3 if cd.megaEx else 2 if cd.ex else 1


def _wadj_damage(base, my_type, target_cd):
    """Weakness x2 / resistance -30 adjusted damage (the heuristic's core signal)."""
    d = base
    if target_cd is not None and my_type is not None:
        if target_cd.weakness is not None and int(target_cd.weakness) == int(my_type):
            d *= 2
        if target_cd.resistance is not None and int(target_cd.resistance) == int(my_type):
            d -= 30
    return max(0, d)


def hint_from_scores(scores):
    """Min-max normalise a teacher's per-option scores to [0,1] (best option -> 1).
    Used by the optional 'hint feature' experiment to test the info-gap hypothesis."""
    s = np.asarray(scores, dtype=np.float32)
    if s.size == 0:
        return s
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-6:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


def _attack_stats(op, my_type, opp_active, attack_db, cdb):
    """For an ATTACK option: (ko_active, wadj_damage/300, prize_if_ko/3)."""
    if op.type != OptionType.ATTACK or op.attackId is None or attack_db is None or opp_active is None:
        return 0.0, 0.0, 0.0
    a = attack_db.get(op.attackId)
    if a is None:
        return 0.0, 0.0, 0.0
    tc = cdb.get(opp_active.id) if cdb else None
    d = _wadj_damage(a.damage, my_type, tc)
    ko = 1.0 if opp_active.hp <= d else 0.0
    prize = (_prize_of(tc) / 3.0) if ko else 0.0
    return ko, min(d / 300.0, 1.5), prize


def _ratio(x, d):
    if x is None or d == 0:
        return 0.0
    return float(x) / float(d)


def _player_features(p) -> list:
    """Summarize one PlayerState into a small fixed vector."""
    if p is None:
        return [0.0] * 11
    active = p.active[0] if p.active else None
    has_active = 1.0 if active is not None else 0.0
    hp_ratio = _ratio(active.hp, active.maxHp) if active else 0.0
    active_energy = float(len(active.energies)) / 6.0 if active else 0.0
    bench_energy = sum(len(b.energies) for b in p.bench) if p.bench else 0
    return [
        _ratio(len(p.prize), 6),
        _ratio(p.deckCount, 60),
        _ratio(p.handCount, 12),
        _ratio(len(p.bench), 5),
        has_active,
        hp_ratio,
        active_energy,
        _ratio(bench_energy, 20),
        _ratio(len(p.discard), 60),
        1.0 if (p.poisoned or p.burned) else 0.0,
        1.0 if (p.asleep or p.paralyzed or p.confused) else 0.0,
    ]


def _board_count(p, cid):
    if p is None:
        return 0
    return sum(1 for card in list(p.active or []) + list(p.bench or []) if card is not None and card.id == cid)


def _board_has_any(p, ids):
    if p is None:
        return 0.0
    zones = (p.active or []), (p.bench or []), (p.discard or [])
    return 1.0 if any(card is not None and card.id in ids for zone in zones for card in zone) else 0.0


def global_features(obs, attack_db=None) -> np.ndarray:
    st = obs.current
    sel = obs.select
    if st is None:
        return np.zeros(global_dim(), dtype=np.float32)
    me = st.players[st.yourIndex]
    opp = st.players[1 - st.yourIndex]
    feats = [
        _ratio(st.turn, 40),
        _ratio(st.turnActionCount, 20),
        1.0 if st.supporterPlayed else 0.0,
        1.0 if st.stadiumPlayed else 0.0,
        1.0 if st.energyAttached else 0.0,
        1.0 if st.retreated else 0.0,
        1.0 if st.stadium else 0.0,
    ]
    feats += _player_features(me)
    feats += _player_features(opp)
    # select type one-hot (11 SelectType values)
    sel_type = [0.0] * 11
    if sel is not None and 0 <= int(sel.type) < 11:
        sel_type[int(sel.type)] = 1.0
    feats += sel_type
    if sel is not None:
        feats += [_ratio(sel.context, 49), _ratio(sel.minCount, 6), _ratio(sel.maxCount, 6)]
    else:
        feats += [0.0, 0.0, 0.0]

    # heuristic-derived combat features (matchup awareness)
    cdb = engine.card_db()
    my_active = me.active[0] if me.active else None
    opp_active = opp.active[0] if opp.active else None
    my_type = cdb[my_active.id].energyType if (my_active and my_active.id in cdb) else None
    opp_cd = cdb.get(opp_active.id) if opp_active else None
    weak_match = 1.0 if (opp_cd is not None and my_type is not None
                         and opp_cd.weakness is not None and int(opp_cd.weakness) == int(my_type)) else 0.0
    opp_prize = _ratio(_prize_of(opp_cd), 3) if opp_active else 0.0
    opp_energy = _ratio(len(opp_active.energies), 6) if opp_active else 0.0
    can_ohko = 0.0
    if my_active and opp_active and attack_db is not None:
        for aid in (cdb[my_active.id].attacks if my_active.id in cdb else []):
            a = attack_db.get(aid)
            if a and _wadj_damage(a.damage, my_type, opp_cd) >= opp_active.hp:
                can_ohko = 1.0
                break
    feats += [opp_prize, weak_match, opp_energy, can_ohko]
    my_active_id = my_active.id if my_active else None
    opp_active_id = opp_active.id if opp_active else None
    feats += [1.0 if my_active_id == cid else 0.0 for cid in _MY_BOARD_IDS]
    feats += [1.0 if opp_active_id == cid else 0.0 for cid in _OPP_ACTIVE_IDS]
    feats += [_ratio(_board_count(me, cid), 4) for cid in _MY_BOARD_IDS]
    feats += [
        _board_has_any(opp, {344, 345}),
        _board_has_any(opp, {119, 120, 121}),
        _board_has_any(opp, {677, 678}),
    ]
    return np.asarray(feats, dtype=np.float32)


def option_features(obs, attack_db=None, hint=None) -> np.ndarray:
    """(n_options, F) feature matrix. `hint` (optional, len n) = teacher score signal."""
    sel = obs.select
    if sel is None or not sel.option:
        return np.zeros((0, option_dim()), dtype=np.float32)
    st = obs.current
    opp = (1 - st.yourIndex) if st is not None else 1
    cdb = engine.card_db()
    me_p = st.players[st.yourIndex] if st is not None else None
    opp_p = st.players[opp] if st is not None else None
    my_active = me_p.active[0] if (me_p and me_p.active) else None
    opp_active = opp_p.active[0] if (opp_p and opp_p.active) else None
    my_type = cdb[my_active.id].energyType if (my_active and my_active.id in cdb) else None
    rows = []
    for oi, op in enumerate(sel.option):
        oh = [0.0] * _NUM_OPTION_TYPES
        if 0 <= int(op.type) < _NUM_OPTION_TYPES:
            oh[int(op.type)] = 1.0
        dmg = 0.0
        if op.attackId is not None and attack_db is not None:
            a = attack_db.get(op.attackId)
            if a:
                dmg = a.damage / 300.0
        is_opp = 1.0 if op.playerIndex == opp else 0.0
        ko, wadj, prize = _attack_stats(op, my_type, opp_active, attack_db, cdb)
        row = oh + [
            is_opp,
            dmg,
            _ratio(op.area, 12),
            _ratio(op.index, 20),
            _ratio(op.inPlayIndex, 5),
            _ratio(op.energyIndex, 6),
            _ratio(op.count, 6),
            _ratio(op.number, 6),
            ko,        # this attack KOs opp active
            wadj,      # weakness-adjusted damage vs opp active /300
            prize,     # prizes gained if KO /3
            float(hint[oi]) if (hint is not None and oi < len(hint)) else 0.0,  # teacher hint
            1.0 if (op.inPlayArea is not None and int(op.inPlayArea) == int(AreaType.ACTIVE)) else 0.0,
            _ratio(op.attackId, 1100),
        ]
        row += [1.0 if op.attackId == key else 0.0 for key in _KEY_ATTACK_IDS]
        ref_cid = _ref_card_id(op, me_p)
        if ref_cid is None and sel.deck and op.index is not None and op.index < len(sel.deck):
            ref_cid = sel.deck[op.index].id
        row += _card_role_features(ref_cid)
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def global_dim() -> int:
    # base 47 + board identity/archetype flags.
    return 7 + 11 + 11 + 11 + 3 + 4 + len(_MY_BOARD_IDS) + len(_OPP_ACTIVE_IDS) + len(_MY_BOARD_IDS) + 3


def option_dim() -> int:
    # 17 type + 8 base + 3 attack(ko/wadj/prize) + 1 hint + 1 attach-active
    # + 1 attack-id + key attack flags + expanded card identity/role features.
    return (
        _NUM_OPTION_TYPES
        + 8
        + 3
        + 1
        + 1
        + 1
        + len(_KEY_ATTACK_IDS)
        + (_NUM_CARD_TYPES + 4 + 7 + len(_KEY_CARD_IDS))
    )


@dataclass
class EncodedDecision:
    glob: np.ndarray          # (global_dim,)
    options: np.ndarray       # (n, option_dim)
    n_options: int
    sel_type: int
    min_count: int
    max_count: int


def encode(obs, attack_db=None, hint=None) -> EncodedDecision:
    g = global_features(obs, attack_db=attack_db)
    o = option_features(obs, attack_db=attack_db, hint=hint)
    sel = obs.select
    return EncodedDecision(
        glob=g,
        options=o,
        n_options=o.shape[0],
        sel_type=int(sel.type) if sel is not None else -1,
        min_count=sel.minCount if sel is not None else 0,
        max_count=sel.maxCount if sel is not None else 0,
    )
