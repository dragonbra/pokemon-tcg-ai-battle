"""Ver5 — attack-planning heuristic (learned from LB-800+ Mega Lucario agents).

Core idea (the source of the reference agents' strength): before choosing an
action, *plan the best attack* this turn — over (my attacker × its attacks ×
target), applying weakness x2 / resistance -30, KO detection and prize math
(mega=3 / ex=2 / else=1), including gusting a benched target with Boss's Orders.
Then score every option so that board setup (evolve → energy → develop) happens
first and the planned attack fires last. Written generically over card_db so it
works on any deck; tuned with the Fighting/Lucario deck in mind.
"""
from dataclasses import dataclass

from cg.api import (
    to_observation_class, SelectType, SelectContext, OptionType,
    CardType, AreaType, EnergyType,
)
from src import engine

GUST_IDS = {1182, 1124, 1088, 1143}      # Boss's Orders / catchers / repel
DRAW_SUP_IDS = {1192, 1227, 1224, 1236}  # Carmine / Lillie / Cheren / Urbain
DMG_BOOST_IDS = {1141}                    # Premium Power Pro (+30 to Fighting)
HP_TOOL_IDS = {1159, 1158}               # Hero's Cape (+100) / Maximum Belt
MAX_ACTIONS_PER_TURN = 60
LOW_DECK = 10                            # below this, stop drawing/searching (avoid deck-out)


@dataclass
class Plan:
    attack_id: int = -1
    target_owner: int = -1     # opponent player index of the target
    target_is_active: bool = True
    target_bench_idx: int = -1
    needs_energy: bool = False
    needs_gust: bool = False
    ko: bool = False
    score: float = -1.0
    needs_switch: bool = False   # retreat to a benched attacker (e.g. to break a wall)
    switch_to: int = -1          # bench index of that attacker


def _cd(cid):
    return engine.card_db().get(cid)


def _prize(pokemon):
    c = _cd(pokemon.id)
    if c is None:
        return 1
    return 3 if c.megaEx else 2 if c.ex else 1


def _attacker_type(pokemon):
    c = _cd(pokemon.id)
    return None if c is None else c.energyType


_EX_IMMUNE_CACHE = {}


def _prevents_ex_damage(cd):
    """True if this Pokemon's ability blocks all damage from opponent ex attacks
    (e.g. Crustle 'Mysterious Rock Inn'). Cached per card id."""
    if cd is None:
        return False
    cid = cd.cardId
    if cid in _EX_IMMUNE_CACHE:
        return _EX_IMMUNE_CACHE[cid]
    val = False
    for s in (cd.skills or []):
        t = (s.text or "").lower()
        if "prevent all damage" in t and "{ex}" in t:
            val = True
            break
    _EX_IMMUNE_CACHE[cid] = val
    return val


def _dmg_vs(attack, atk_type, target, attacker_is_ex=False):
    """Weakness/resistance-adjusted damage of `attack` (by an atk_type pokemon) vs target.
    Returns 0 if the target is immune to ex damage and the attacker is an ex (the wall case)."""
    d = attack.damage
    if d > 0 and "flip a coin" in (attack.text or "").lower():
        d = int(d * 0.7)        # expected-value discount: don't rely on coin damage
    tc = _cd(target.id)
    if tc is None or atk_type is None:
        return d
    if attacker_is_ex and _prevents_ex_damage(tc):
        return 0
    if tc.weakness is not None and int(tc.weakness) == int(atk_type):
        d *= 2
    if tc.resistance is not None and int(tc.resistance) == int(atk_type):
        d -= 30
    return max(0, d)


class Ver5Agent:
    def __init__(self, deck=None, rng=None):
        self.deck = deck
        self._turn = -1
        self._acts = 0
        self._wall = False
        self._breaker_bi = -1

    def reset(self):
        self._turn = -1
        self._acts = 0
        self._wall = False
        self._breaker_bi = -1

    # ---- entry -------------------------------------------------------------
    def scores_for(self, obs):
        """Per-option heuristic scores for the current select (distillation signal)."""
        sel = obs.select
        if sel.type == SelectType.MAIN:
            t = obs.current.turn
            if t != self._turn:
                self._turn, self._acts = t, 0
            self._acts += 1
        plan = self._plan(obs) if obs.current is not None else Plan()
        return [self._score(obs, op, plan) for op in sel.option]

    def __call__(self, obs_dict):
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return list(self.deck)
        sel = obs.select
        if not sel.option:
            return list(range(sel.minCount))

        scores = self.scores_for(obs)
        order = sorted(range(len(sel.option)), key=lambda i: scores[i], reverse=True)
        k = max(sel.minCount, min(1 if sel.maxCount >= 1 else sel.maxCount, sel.maxCount))
        # multi-pick: take the top maxCount; single-pick: top 1 (>=minCount)
        k = sel.maxCount if sel.maxCount > 1 else max(sel.minCount, 1 if sel.maxCount == 1 else 0)
        k = min(k, len(sel.option))
        chosen = order[:k]
        if len(chosen) < sel.minCount:
            chosen = order[:sel.minCount]
        return chosen

    # ---- attack planning ---------------------------------------------------
    def _plan(self, obs):
        st = obs.current
        me = st.players[st.yourIndex]
        opp = st.players[1 - st.yourIndex]
        opp_idx = 1 - st.yourIndex
        best = Plan()
        self._wall = False        # our active can't damage the opp active (e.g. ex-immune wall)
        self._breaker_bi = -1     # benched non-ex attacker to power up / switch to
        if st.turn < 2:
            return best
        active = me.active[0] if me.active else None
        if active is None:
            return best
        active_cd = _cd(active.id)
        if active_cd is None:
            return best
        atk_type = active_cd.energyType
        attacker_is_ex = bool(active_cd.ex or active_cd.megaEx)
        adb = engine.attack_db()

        gust_ok = self._has_playable(obs, GUST_IDS) and not st.supporterPlayed
        targets = []
        if opp.active and opp.active[0] is not None:
            targets.append((opp.active[0], True, -1))
        if gust_ok:
            for bi, bp in enumerate(opp.bench or []):
                if bp is not None:
                    targets.append((bp, False, bi))

        cur_energy = len(active.energies)
        can_attach = (not st.energyAttached) and self._has_basic_energy_in_hand(obs)
        for aid in active_cd.attacks:
            a = adb.get(aid)
            if a is None:
                continue
            cost = len(a.energies)
            if cur_energy >= cost:
                needs_energy = False
            elif can_attach and cur_energy + 1 >= cost:
                needs_energy = True
            else:
                continue
            for target, is_active, bidx in targets:
                d = _dmg_vs(a, atk_type, target, attacker_is_ex)
                if d <= 0:
                    continue
                ko = target.hp <= d
                prize = _prize(target) if ko else 0
                score = prize * 10000 + (target.hp if ko else d)
                if is_active:
                    score += 300
                if len(opp.prize) <= prize and prize > 0:
                    score = 500000
                if score > best.score:
                    best = Plan(aid, opp_idx, is_active, bidx, needs_energy,
                                not is_active, ko, score)

        # ---- wall handling: active can't dent the opponent's active ----
        oa = opp.active[0] if opp.active else None
        if oa is not None:
            active_best = 0
            for aid in active_cd.attacks:
                a = adb.get(aid)
                if a:
                    active_best = max(active_best, _dmg_vs(a, atk_type, oa, attacker_is_ex))
            if active_best < 20:                     # effectively walled
                best_bi, powered_ko_bi = -1, -1
                for bi, bp in enumerate(me.bench or []):
                    bcd = _cd(bp.id)
                    if bcd is None:
                        continue
                    bex = bool(bcd.ex or bcd.megaEx)
                    for aid in bcd.attacks:
                        a = adb.get(aid)
                        if a is None:
                            continue
                        d = _dmg_vs(a, bcd.energyType, oa, bex)
                        if d <= 0:
                            continue
                        if best_bi < 0:
                            best_bi = bi
                        if len(bp.energies) >= len(a.energies) and oa.hp <= d:
                            powered_ko_bi = bi
                if best_bi >= 0:
                    self._wall = True
                    self._breaker_bi = powered_ko_bi if powered_ko_bi >= 0 else best_bi
                    if powered_ko_bi >= 0:           # switch in the powered breaker now
                        best = Plan(target_owner=opp_idx, target_is_active=True, ko=True,
                                    score=600000.0, needs_switch=True, switch_to=powered_ko_bi)

        return best

    def _has_playable(self, obs, ids):
        me = obs.current.players[obs.current.yourIndex]
        return any(c.id in ids for c in (me.hand or []))

    def _has_basic_energy_in_hand(self, obs):
        me = obs.current.players[obs.current.yourIndex]
        for c in (me.hand or []):
            cd = _cd(c.id)
            if cd and cd.cardType == int(CardType.BASIC_ENERGY):
                return True
        return False

    # ---- option scoring ----------------------------------------------------
    def _hand_card(self, obs, option):
        me = obs.current.players[obs.current.yourIndex]
        if option.index is None or me.hand is None or option.index >= len(me.hand):
            return None
        return me.hand[option.index]

    def _pokemon_at(self, obs, area, index, player_index):
        st = obs.current
        p = st.players[player_index]
        if area == AreaType.ACTIVE and p.active:
            return p.active[0]
        if area == AreaType.BENCH and index is not None and index < len(p.bench):
            return p.bench[index]
        if area == AreaType.HAND and index is not None and p.hand and index < len(p.hand):
            return p.hand[index]
        return None

    def _score(self, obs, op, plan):
        sel = obs.select
        ctx = sel.context
        if sel.type == SelectType.YES_NO:
            if op.type == OptionType.YES:
                return 0.0 if ctx == SelectContext.MULLIGAN else 100.0
            if op.type == OptionType.NO:
                return 100.0 if ctx == SelectContext.MULLIGAN else 0.0
            return 0.0
        if op.type == OptionType.NUMBER:
            return float(op.number or 0)

        if sel.type == SelectType.MAIN:
            return self._score_main(obs, op, plan)
        return self._score_sub(obs, op, plan, ctx)

    def _score_main(self, obs, op, plan):
        st = obs.current
        me = st.players[st.yourIndex]
        low_deck = me.deckCount <= LOW_DECK
        if self._acts > MAX_ACTIONS_PER_TURN:
            if op.type == OptionType.ATTACK:
                return 1000.0
            if op.type == OptionType.END:
                return 0.0
        t = op.type
        if t == OptionType.ABILITY:
            return 8000.0
        if t == OptionType.PLAY:
            c = self._hand_card(obs, op)
            cd = _cd(c.id) if c else None
            if cd is None:
                return 100.0
            if cd.cardType == int(CardType.POKEMON):
                if cd.ex or cd.megaEx:
                    ex_on_board = sum(
                        1 for p in (([me.active[0]] if me.active else []) + list(me.bench or []))
                        if p and (lambda c: bool(c and (c.ex or c.megaEx)))(_cd(p.id)))
                    if ex_on_board >= 1:
                        return 6000.0               # avoid stacking ex (2-3 prize liabilities)
                return 20000.0                      # develop board first
            if c.id in GUST_IDS:
                return 6500.0 if plan.needs_gust else -1.0
            if c.id in DMG_BOOST_IDS:
                return 5500.0 if plan.ko else -1.0
            if c.id in DRAW_SUP_IDS:
                if st.supporterPlayed or low_deck or me.handCount >= 6:
                    return -1.0                      # don't over-draw / deck out
                return 5000.0
            return -1.0 if low_deck else 4000.0      # other items (search etc.)
        if t == OptionType.EVOLVE:
            tgt = self._pokemon_at(obs, op.inPlayArea, op.inPlayIndex, st.yourIndex)
            bonus = len(tgt.energies) if tgt else 0
            return 9000.0 + bonus
        if t == OptionType.ATTACH:
            c = self._hand_card(obs, op)
            cd = _cd(c.id) if c else None
            tgt = self._pokemon_at(obs, op.inPlayArea, op.inPlayIndex, st.yourIndex)
            is_active = op.inPlayArea == AreaType.ACTIVE
            if c and c.id in HP_TOOL_IDS:
                return 7500.0 + (200.0 if is_active else 0.0)
            if cd and cd.cardType in (int(CardType.BASIC_ENERGY), int(CardType.SPECIAL_ENERGY)):
                # wall mode: power up the benched non-ex breaker instead of the stuck active
                if self._wall and op.inPlayArea == AreaType.BENCH and op.inPlayIndex == self._breaker_bi:
                    return 8500.0
                base = 7000.0
                if is_active and plan.needs_energy:
                    base += 1000.0
                if self._wall and is_active:
                    base -= 4000.0          # don't keep loading the walled active
                return base
            return 3000.0
        if t == OptionType.RETREAT:
            return 9500.0 if plan.needs_switch else -1.0
        if t == OptionType.ATTACK:
            return 1100.0 if op.attackId == plan.attack_id else 1000.0
        if t == OptionType.END:
            return 0.0
        return 50.0

    def _score_sub(self, obs, op, plan, ctx):
        st = obs.current
        me_idx = st.yourIndex
        # choosing which of our Pokemon is Active
        if ctx in (SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.TO_ACTIVE, SelectContext.SWITCH):
            # wall mode: switch in the planned breaker
            if (plan.needs_switch and op.area == AreaType.BENCH
                    and op.index == plan.switch_to):
                return 99999.0
            pk = self._pokemon_at(obs, op.area, op.index, op.playerIndex if op.playerIndex is not None else me_idx)
            if pk is None:
                return 0.0
            cd = _cd(pk.id)
            score = (cd.hp if cd else 0) + len(getattr(pk, "energies", [])) * 30
            if cd and (cd.megaEx or cd.ex):
                score += 5000        # put the big attacker forward
            elif cd and cd.evolvesFrom is None and not (cd.stage1 or cd.stage2):
                # fragile evolving basic: ok as setup but keep low-HP ones benched if alt exists
                pass
            return score
        # offensive target selection -> opponent
        if ctx in (SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY,
                   SelectContext.DEVOLVE):
            return 100.0 if op.playerIndex == (1 - me_idx) else 0.0
        # gust target -> the planned bench target
        if ctx == SelectContext.TO_ACTIVE and op.playerIndex == (1 - me_idx):
            return 100.0 if op.index == plan.target_bench_idx else 10.0
        # search to hand -> prefer pieces we lack
        if ctx == SelectContext.TO_HAND:
            return self._score_search(obs, op)
        # attach target -> the attacker / active
        if ctx == SelectContext.ATTACH_FROM:
            pk = self._pokemon_at(obs, op.area, op.index, me_idx)
            return self._energy_target_score(pk, op.area == AreaType.ACTIVE) if pk else 0.0
        # board development -> take it
        if ctx in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH, SelectContext.TO_FIELD):
            return 100.0
        # heal / draw beneficial -> take
        if ctx in (SelectContext.HEAL, SelectContext.REMOVE_DAMAGE_COUNTER, SelectContext.DRAW_COUNT):
            return 100.0
        return 0.0

    def _energy_target_score(self, pokemon, active):
        cd = _cd(pokemon.id)
        score = 1000.0 + (50.0 if active else 0.0)
        if cd and (cd.megaEx or cd.ex):
            score += 500.0
        score -= len(getattr(pokemon, "energies", [])) * 20.0   # spread / top up the bare one
        return score

    def _score_search(self, obs, op):
        st = obs.current
        me = st.players[st.yourIndex]
        pk = self._pokemon_at(obs, op.area, op.index, op.playerIndex if op.playerIndex is not None else st.yourIndex)
        # `deck` cards come via obs.select.deck for some searches
        card = pk
        if card is None and obs.select.deck and op.index is not None and op.index < len(obs.select.deck):
            card = obs.select.deck[op.index]
        if card is None:
            return 50.0
        cd = _cd(card.id)
        if cd is None:
            return 50.0
        in_play = sum(1 for s in (([me.active[0]] if me.active else []) + list(me.bench or [])) if s and s.id == card.id)
        in_hand = sum(1 for c in (me.hand or []) if c.id == card.id)
        score = 200.0 - (in_play + in_hand) * 60.0
        if cd.cardType == int(CardType.POKEMON):
            if cd.megaEx or cd.ex:
                score += 60.0       # the win condition
            elif cd.basic:
                score += 40.0
        return score
