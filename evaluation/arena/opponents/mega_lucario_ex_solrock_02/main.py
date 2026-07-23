"""Opponent: pilkwang_v2 - Lucario v2 strategic baseline with Crustle-aware policy"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time
import random

try:
    from cg.api import search_begin, search_step, search_release
    _SEARCH_AVAILABLE = True
except Exception:
    try:
        from cg.api import search_begin, search_step, search_end as search_release
        _SEARCH_AVAILABLE = True
    except Exception:
        _SEARCH_AVAILABLE = False

from cg.api import (
    AreaType,
    Card,
    CardType,
    EnergyType,
    Observation,
    OptionType,
    Pokemon,
    SelectContext,
    all_card_data,
    to_observation_class,
)


class C:
    DWEBBLE = 344
    CRUSTLE = 345

    KYOGRE = 721
    SNOVER = 722
    MEGA_ABOMASNOW_EX = 723

    MAKUHITA = 673
    HARIYAMA = 674
    LUNATONE = 675
    SOLROCK = 676
    RIOLU = 677
    MEGA_LUCARIO_EX = 678

    BASIC_FIGHTING_ENERGY = 6
    DUSK_BALL = 1102
    SWITCH = 1123
    PREMIUM_POWER_PRO = 1141
    FIGHTING_GONG = 1142
    POKE_PAD = 1152
    HERO_CAPE = 1159
    BOSS_ORDERS = 1182
    CARMINE = 1192
    LILLIE_DETERMINATION = 1227
    GRAVITY_MOUNTAIN = 1252

    LUMIOSE_CITY = 1267
    LILLIES_PEARL = 1172
    LEGACY_ENERGY = 12


MEGA_BRAVE = 983
LOW_DECK_COUNT = 8
CRUSTLE_AWARE = True

PACKAGE_ROOT = Path(__file__).resolve().parent


def read_deck_csv() -> list[int]:
    return [
        int(line.strip())
        for line in (PACKAGE_ROOT / "deck.csv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


DECK = read_deck_csv()
my_deck = DECK


all_card = all_card_data()
card_table = {card.cardId: card for card in all_card}


class AttackPlan:
    def __init__(
        self,
        attacker: int = -1,
        target: int = -1,
        attack_index: int = -1,
        remain_hp: int = -1,
        needs_energy: bool = False,
    ):
        self.attacker = attacker
        self.target = target
        self.attack_index = attack_index
        self.remain_hp = remain_hp
        self.needs_energy = needs_energy


plan = AttackPlan()
pre_turn = -1
ability_used = False


def _has_attack_plan() -> bool:
    return plan.attacker >= 0 and plan.target >= 0 and plan.attack_index >= 0


def _plan_kos() -> bool:
    return _has_attack_plan() and plan.remain_hp <= 0

# Optional search hook. The Phase 1 notebook keeps this disabled; it exists so later
# experiments can be connected without changing the submission contract.
USE_SEARCH = False

# --- Phase-1 diagnostics: distinguish "policy ran" from "fell back". ---------
# A submission that never crashes can still be silently broken: if the policy
# touches an SDK attribute that does not exist on the live engine, the exception
# is caught and we return _legal_fallback (the first minCount indices). "0 errors"
# then hides a near-random agent. _DIAG makes the fallback rate and the actual
# exception messages observable so the validation gate can catch this.
_DIAG = {"decisions": 0, "policy_ok": 0, "policy_fallback": 0,
         "obs_fallback": 0, "deck_returns": 0, "errors": {},
         "policy_errors": {}, "search_errors": {},
         "search_used": 0, "search_failed": 0, "search_disabled": 0,
         "chosen_types": {}, "attack_ids_chosen": {},
         "attack_opts_by_active": {}, "mega_brave_present": 0,
         "yes_no_contexts": {}, "crustle_seen": 0,
         "crustle_active_seen": 0, "crustle_wall_policy_seen": 0,
         "crustle_ex_attack_options": 0,
         "crustle_ex_attack_suppressed": 0,
         "crustle_ex_attack_chosen": 0,
         "crustle_non_ex_attack_chosen": 0,
         "crustle_final_plan_ex_into_wall": 0,
         "crustle_final_plan_non_ex_into_wall": 0}


def _diag_record_error(exc):
    key = type(exc).__name__ + ": " + str(exc)[:160]
    _DIAG["errors"][key] = _DIAG["errors"].get(key, 0) + 1
    _DIAG["policy_errors"][key] = _DIAG["policy_errors"].get(key, 0) + 1


def _diag_record_search_error(exc):
    key = type(exc).__name__ + ": " + str(exc)[:160]
    _DIAG["errors"][key] = _DIAG["errors"].get(key, 0) + 1
    _DIAG["search_errors"][key] = _DIAG["search_errors"].get(key, 0) + 1


def diag_reset():
    _DIAG.update({"decisions": 0, "policy_ok": 0, "policy_fallback": 0,
                  "obs_fallback": 0, "deck_returns": 0, "errors": {},
                  "policy_errors": {}, "search_errors": {},
                  "search_used": 0, "search_failed": 0, "search_disabled": 0,
                  "chosen_types": {}, "attack_ids_chosen": {},
                  "attack_opts_by_active": {}, "mega_brave_present": 0,
                  "yes_no_contexts": {}, "crustle_seen": 0,
                  "crustle_active_seen": 0, "crustle_wall_policy_seen": 0,
                  "crustle_ex_attack_options": 0,
                  "crustle_ex_attack_suppressed": 0,
                  "crustle_ex_attack_chosen": 0,
                  "crustle_non_ex_attack_chosen": 0,
                  "crustle_final_plan_ex_into_wall": 0,
                  "crustle_final_plan_non_ex_into_wall": 0})


def diag_snapshot():
    snap = {}
    for k, v in _DIAG.items():
        if k == "attack_opts_by_active":
            snap[k] = {kk: sorted(vv) for kk, vv in v.items()}
        elif isinstance(v, dict):
            snap[k] = dict(v)
        else:
            snap[k] = v
    dec = max(1, snap.get("decisions", 0))
    snap["fallback_rate"] = (snap.get("policy_fallback", 0) + snap.get("obs_fallback", 0)) / dec
    sd = snap.get("search_used", 0) + snap.get("search_failed", 0)
    snap["search_fail_rate"] = (snap.get("search_failed", 0) / sd) if sd else 0.0
    return snap


def _diag_observe(obs):
    """Record, on the REAL obs, which attacks are legal (reveals true attackIds)."""
    try:
        sel = obs.select
        st = obs.current
        if sel is None or st is None:
            return
        me = st.players[st.yourIndex]
        op = st.players[1 - st.yourIndex]
        active_id = me.active[0].id if (me.active and me.active[0] is not None) else -1
        op_active_id = op.active[0].id if (op.active and op.active[0] is not None) else -1
        opponent_ids = [p.id for p in (op.active + op.bench) if p is not None]
        crustle_seen = any(cid in {C.DWEBBLE, C.CRUSTLE} for cid in opponent_ids)
        if crustle_seen:
            _DIAG["crustle_seen"] += 1
        if op_active_id == C.CRUSTLE:
            _DIAG["crustle_active_seen"] += 1
        for opt in sel.option:
            if opt.type == OptionType.ATTACK:
                aid = getattr(opt, "attackId", None)
                if aid is not None:
                    _DIAG["attack_opts_by_active"].setdefault(active_id, set()).add(aid)
                    if active_id == C.MEGA_LUCARIO_EX and op_active_id == C.CRUSTLE:
                        _DIAG["crustle_ex_attack_options"] += 1
                    if aid == MEGA_BRAVE:
                        _DIAG["mega_brave_present"] += 1
            if opt.type in {OptionType.YES, OptionType.NO}:
                ctx = str(getattr(sel, "context", None))
                _DIAG["yes_no_contexts"][ctx] = _DIAG["yes_no_contexts"].get(ctx, 0) + 1
    except Exception:
        pass


def _diag_observe_choice(obs, selection):
    """Record which option TYPES (and attackIds) actually get chosen."""
    try:
        sel = obs.select
        st = obs.current
        if sel is None or st is None or not selection:
            return
        me = st.players[st.yourIndex]
        op = st.players[1 - st.yourIndex]
        active_id = me.active[0].id if (me.active and me.active[0] is not None) else -1
        op_active_id = op.active[0].id if (op.active and op.active[0] is not None) else -1
        for i in selection:
            if 0 <= i < len(sel.option):
                opt = sel.option[i]
                key = str(opt.type)
                _DIAG["chosen_types"][key] = _DIAG["chosen_types"].get(key, 0) + 1
                if opt.type == OptionType.ATTACK:
                    aid = getattr(opt, "attackId", None)
                    if aid is not None:
                        _DIAG["attack_ids_chosen"][aid] = _DIAG["attack_ids_chosen"].get(aid, 0) + 1
                    if op_active_id == C.CRUSTLE:
                        if active_id == C.MEGA_LUCARIO_EX:
                            _DIAG["crustle_ex_attack_chosen"] += 1
                        elif active_id in {C.HARIYAMA, C.MAKUHITA, C.SOLROCK}:
                            _DIAG["crustle_non_ex_attack_chosen"] += 1
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Forward-search over the engine model. This uses the live SDK convention:
# the observation dictionary carries search_begin_input, search_begin returns a
# wrapper with {error, state}, search_step advances a searchId, and search_release
# must be called for every started branch.
# ---------------------------------------------------------------------------
SEARCH_NODE_BUDGET = 120        # max forward-steps per decision
SEARCH_TIME_BUDGET_S = 1.5      # wall-clock cap per decision (respect 10-min/game)
SEARCH_ACTION_CAP = 20          # root MAIN candidates evaluated by search
SEARCH_FAILURE_DISABLE_AFTER = 8
SEARCH_MIN_ATTEMPTS_FOR_RATE = 10
SEARCH_FAIL_RATE_DISABLE = 0.75
SEARCH_OPPONENT_PLY = False     # opponent reply search is experimental, off by default


def _search_temporarily_disabled():
    attempts = _DIAG.get("search_used", 0) + _DIAG.get("search_failed", 0)
    failed = _DIAG.get("search_failed", 0)
    used = _DIAG.get("search_used", 0)
    if failed >= SEARCH_FAILURE_DISABLE_AFTER and used == 0:
        return True
    if attempts >= SEARCH_MIN_ATTEMPTS_FOR_RATE and failed / max(1, attempts) >= SEARCH_FAIL_RATE_DISABLE:
        return True
    return False


def _board_value(obs, my_index):
    """Win-aligned leaf score used only by optional forward search."""
    st = obs.current
    res = getattr(st, "result", -1)
    if res is not None and res >= 0:
        if res == my_index:
            return 1_000_000.0
        if res == (1 - my_index):
            return -1_000_000.0
        return 0.0

    me = st.players[my_index]
    op = st.players[1 - my_index]
    val = 10_000.0 * (len(op.prize) - len(me.prize))

    my_active = me.active[0] if me.active else None
    op_active = op.active[0] if op.active else None
    if op_active is not None:
        val += 2.0 * _damage_on(op_active)
        val += 800.0 * prize_count(op_active)
    if my_active is not None:
        val -= 1.5 * _damage_on(my_active)
        val -= 600.0 * prize_count(my_active)

    ready_attackers = 0
    for pkm in (me.active + me.bench):
        if pkm is None:
            continue
        energy_count = len(getattr(pkm, "energies", []))
        if pkm.id == C.MEGA_LUCARIO_EX and energy_count >= 2:
            ready_attackers += 1
        elif pkm.id == C.HARIYAMA and energy_count >= 3:
            ready_attackers += 1
        elif pkm.id == C.SOLROCK and energy_count >= 1:
            ready_attackers += 1
        val += 30.0 * energy_count

    val += 300.0 * ready_attackers
    val += 20.0 * getattr(me, "handCount", 0)
    if getattr(me, "deckCount", 0) <= 4:
        val -= 500.0 * (5 - getattr(me, "deckCount", 0))
    return val


def _should_search(obs, ranked, scores):
    sel = obs.select
    if sel is None or obs.current is None:
        return False
    if sel.context != SelectContext.MAIN:
        return False
    if len(sel.option) <= 1:
        return False
    option_types = {opt.type for opt in sel.option}
    tactical_types = {
        OptionType.ATTACK,
        OptionType.ATTACH,
        OptionType.EVOLVE,
        OptionType.RETREAT,
        OptionType.PLAY,
    }
    if not (option_types & tactical_types):
        return False
    if _plan_kos():
        return True
    if plan.target >= 1:
        return True
    if plan.attacker > 0:
        return True
    if plan.needs_energy:
        return True
    return False


def _result_state(result):
    if result is None:
        return None
    if isinstance(result, dict):
        error = result.get("error", 0)
        state = result.get("state")
    else:
        error = getattr(result, "error", 0)
        state = getattr(result, "state", None)
    if error not in (0, None):
        return None
    if state is not None:
        return state
    if hasattr(result, "observation") or hasattr(result, "searchId"):
        return result
    return None


def _state_id(state):
    if isinstance(state, dict):
        return state.get("searchId")
    return getattr(state, "searchId", None)


def _state_obs(state):
    if isinstance(state, dict):
        return state.get("observation")
    return getattr(state, "observation", None)


def _release_search(sid):
    try:
        search_release(sid)
    except TypeError:
        try:
            search_release()
        except Exception:
            pass
    except Exception:
        pass


def _search_step_state(sid, selection):
    if sid is None:
        return None
    return _result_state(search_step(sid, selection))


def _legal_sim_selection(obs, preferred_first=None):
    try:
        policy = LucarioPolicy(obs)
        ranked, scores = policy.rank()
        if preferred_first is not None:
            ranked = [preferred_first] + [i for i in ranked if i != preferred_first]
        selection = normalize_selection(ranked, scores, obs.select)
        return selection if selection else _legal_fallback(obs.select)
    except Exception:
        return _legal_fallback(obs.select)


def _rollout_search_candidate(sbi, first_idx, my_index, t0):
    sid = None
    steps = 0
    cur = None
    try:
        state = _result_state(search_begin(sbi))
        if state is None:
            return None, steps
        sid = _state_id(state)
        cur = _state_obs(state)
        if cur is None or cur.select is None:
            return None, steps

        selection = _legal_sim_selection(cur, preferred_first=first_idx)
        while steps < SEARCH_NODE_BUDGET and (time.time() - t0) <= SEARCH_TIME_BUDGET_S:
            state = _search_step_state(sid, selection)
            if state is None:
                return None, steps
            steps += 1
            cur = _state_obs(state)
            if cur is None or cur.current is None:
                break
            result = getattr(cur.current, "result", -1)
            if result is not None and result >= 0:
                break
            if cur.current.yourIndex != my_index:
                break
            if cur.select is None or len(cur.select.option) == 0:
                break
            selection = _legal_sim_selection(cur)
            if not selection:
                break
            if cur.select.context == SelectContext.MAIN:
                try:
                    if any(cur.select.option[i].type == OptionType.END for i in selection if 0 <= i < len(cur.select.option)):
                        state = _search_step_state(sid, selection)
                        if state is not None:
                            cur = _state_obs(state) or cur
                            steps += 1
                        break
                except Exception:
                    pass
        if cur is None:
            return None, steps
        return _board_value(cur, my_index), steps
    finally:
        if sid is not None:
            _release_search(sid)


def search_plan(obs_dict, obs, ranked, scores):
    if not _SEARCH_AVAILABLE:
        return None
    if _search_temporarily_disabled():
        _DIAG["search_disabled"] += 1
        return None
    try:
        if obs.select is None or obs.current is None:
            return None
        if obs.select.context != SelectContext.MAIN:
            return None
        if obs.select.maxCount == 0 or len(obs.select.option) <= 1:
            return None
        if not _should_search(obs, ranked, scores):
            return None
        sbi = getattr(obs, "search_begin_input", None)
        if sbi is None and isinstance(obs_dict, dict):
            sbi = obs_dict.get("search_begin_input")
        if sbi is None:
            return None

        my_index = obs.current.yourIndex
        candidates = []
        for idx in ranked:
            if not (0 <= idx < len(obs.select.option)) or idx in candidates:
                continue
            score = scores[idx] if idx < len(scores) else 0
            if score <= 0 and obs.select.minCount == 0:
                continue
            candidates.append(idx)
            if len(candidates) >= SEARCH_ACTION_CAP:
                break
        if not candidates:
            return None

        t0 = time.time()
        best_idx, best_val = None, None
        total_steps = 0
        for first_idx in candidates:
            if total_steps >= SEARCH_NODE_BUDGET or (time.time() - t0) > SEARCH_TIME_BUDGET_S:
                break
            value, used_steps = _rollout_search_candidate(sbi, first_idx, my_index, t0)
            total_steps += max(1, used_steps)
            if value is None:
                continue
            if best_val is None or value > best_val:
                best_idx, best_val = first_idx, value
        if best_idx is None:
            _DIAG["search_failed"] += 1
            return None
        _DIAG["search_used"] += 1
        return [best_idx] + [i for i in ranked if i != best_idx]
    except Exception as exc:
        _diag_record_search_error(exc)
        _DIAG["search_failed"] += 1
        return None

def normalize_selection(ranked, scores, select):
    n = len(select.option)
    minc = max(0, min(select.minCount, n))
    maxc = max(minc, min(select.maxCount, n))

    out = []
    seen = set()
    for i in ranked:
        if not (0 <= i < n) or i in seen:
            continue
        score = scores[i] if i < len(scores) else 0
        # Score 0 usually means an unmodelled neutral option. For optional
        # selections, do not take it unless minCount forces a choice.
        if score > 0 or len(out) < minc:
            out.append(i)
            seen.add(i)
        if len(out) >= maxc:
            break

    for i in range(n):
        if len(out) >= minc:
            break
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


def _legal_fallback(select):
    try:
        n = len(select.option)
        k = min(max(0, select.minCount), n)
        return list(range(k))
    except Exception:
        return []


def _legal_fallback_from_dict(obs_dict):
    try:
        sel = obs_dict.get("select") or {}
        opts = sel.get("option") or []
        minc = sel.get("minCount", 0)
        n = len(opts)
        k = min(max(0, minc), n)
        return list(range(k))
    except Exception:
        return []

def _safe_get(seq, index: int):
    try:
        if seq is None or index is None or index < 0 or index >= len(seq):
            return None
        return seq[index]
    except Exception:
        return None

def _ctx_is(context, *names) -> bool:
    for name in names:
        value = getattr(SelectContext, name, None)
        if value is not None and context == value:
            return True
    return False


def _pokemon_max_hp(pokemon: Pokemon) -> int:
    max_hp = getattr(pokemon, "maxHp", None)
    if max_hp is not None:
        return max_hp
    data = card_table.get(getattr(pokemon, "id", None))
    return getattr(data, "hp", getattr(pokemon, "hp", 0))


def _damage_on(pokemon: Pokemon) -> int:
    return max(0, _pokemon_max_hp(pokemon) - getattr(pokemon, "hp", 0))


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    try:
        player = obs.current.players[player_index]
        match area:
            case AreaType.DECK:
                return _safe_get(getattr(obs.select, "deck", None), index)
            case AreaType.HAND:
                return _safe_get(getattr(player, "hand", None), index)
            case AreaType.DISCARD:
                return _safe_get(getattr(player, "discard", None), index)
            case AreaType.ACTIVE:
                return _safe_get(getattr(player, "active", None), index)
            case AreaType.BENCH:
                return _safe_get(getattr(player, "bench", None), index)
            case AreaType.PRIZE:
                return _safe_get(getattr(player, "prize", None), index)
            case AreaType.STADIUM:
                return _safe_get(getattr(obs.current, "stadium", None), index)
            case AreaType.LOOKING:
                return _safe_get(getattr(obs.current, "looking", None), index)
            case _:
                return None
    except Exception:
        return None


def prize_count(pokemon: Pokemon) -> int:
    data = card_table.get(pokemon.id)
    if data is None:
        return 1
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == C.LEGACY_ENERGY:
            count -= 1
    for card in pokemon.tools:
        if card.id == C.LILLIES_PEARL and "Lillie" in data.name:
            count -= 1
    return max(0, count)


def target_score(pokemon: Pokemon) -> int:
    data = card_table.get(pokemon.id)
    if data is None:
        return prize_count(pokemon) * 1000 + getattr(pokemon, "hp", 0) + _damage_on(pokemon) * 2
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130

    if pokemon.id in {144, 322, 323, 337}:  # low-value support Pokemon
        score -= 200
    if pokemon.id == C.SNOVER:
        score += 950
    elif pokemon.id == C.MEGA_ABOMASNOW_EX:
        score += 250
    if pokemon.id == C.RIOLU:
        score += 800
    elif pokemon.id == C.MEGA_LUCARIO_EX:
        score += 100
    if pokemon.id == 112 and len(pokemon.energies) >= 1:  # Munkidori
        score += 300
    score += pokemon.hp
    return score

class LucarioPolicy:
    def __init__(self, obs: Observation):
        self.obs = obs
        self.state = obs.current
        self.select = obs.select
        self.context = self.select.context
        self.my_index = self.state.yourIndex
        self.op_index = 1 - self.my_index
        self.me = self.state.players[self.my_index]
        self.opponent = self.state.players[self.op_index]
        self.my_prizes_left = len(self.me.prize)

        self.field_counts = defaultdict(int)
        self.hand_counts = defaultdict(int)
        self.discard_counts = defaultdict(int)
        self.has_ready_lucario_line = False
        self.has_ready_hariyama_line = False
        self.can_switch = False
        self.can_gust = False
        self.can_attack = False
        self.can_use_mega_brave = False
        self.stadium_id = self.state.stadium[0].id if self.state.stadium else 0
        self.crustle_wall = self._opponent_is_crustle_wall()
        self.water_deck = self._opponent_is_water_deck()
        if self.crustle_wall:
            _DIAG["crustle_wall_policy_seen"] += 1

        self._count_cards()
        self._scan_main_options()

    def rank(self) -> tuple[list[int], list[float]]:
        if not self.select.option or self.select.maxCount == 0:
            return [], []

        if self.context == SelectContext.MAIN:
            self._plan_attack()

        scores = [self._score_option(option) for option in self.select.option]
        ranked = [i for i, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]
        return ranked, scores

    def choose(self) -> list[int]:
        ranked, scores = self.rank()
        selection = normalize_selection(ranked, scores, self.select)
        self.remember_chosen_side_effects(selection)
        return selection
    def _count_cards(self) -> None:
        for pokemon in self.me.active + self.me.bench:
            if pokemon is None:
                continue
            self.field_counts[pokemon.id] += 1
            if pokemon.id in {C.MAKUHITA, C.HARIYAMA} and len(pokemon.energies) >= 3:
                self.has_ready_hariyama_line = True
            if pokemon.id in {C.RIOLU, C.MEGA_LUCARIO_EX} and len(pokemon.energies) >= 2:
                self.has_ready_lucario_line = True

        for card in self.me.hand:
            self.hand_counts[card.id] += 1
        for card in self.me.discard:
            self.discard_counts[card.id] += 1

    def _scan_main_options(self) -> None:
        if self.context != SelectContext.MAIN:
            return
        for option in self.select.option:
            if option.type == OptionType.PLAY:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if card is None:
                    continue
                if card.id == C.SWITCH:
                    self.can_switch = True
                elif card.id == C.BOSS_ORDERS:
                    self.can_gust = True
            elif option.type == OptionType.EVOLVE:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if card is None:
                    continue
                if card.id == C.HARIYAMA:
                    self.can_gust = True
            elif option.type == OptionType.RETREAT:
                self.can_switch = True
            elif option.type == OptionType.ATTACK:
                self.can_attack = True
                if option.attackId == MEGA_BRAVE:
                    self.can_use_mega_brave = True

    def _my_board(self) -> list[Pokemon | None]:
        return self.me.active + self.me.bench

    def _opponent_board(self) -> list[Pokemon | None]:
        return self.opponent.active + self.opponent.bench

    def _opponent_has(self, ids: set[int]) -> bool:
        return any(pokemon is not None and pokemon.id in ids for pokemon in self._opponent_board())

    def _opponent_is_crustle_wall(self) -> bool:
        return CRUSTLE_AWARE and self._opponent_has({C.DWEBBLE, C.CRUSTLE})

    def _opponent_is_water_deck(self) -> bool:
        return self._opponent_has({C.KYOGRE, C.SNOVER, C.MEGA_ABOMASNOW_EX})

    def _is_ex_attacker(self, pokemon: Pokemon) -> bool:
        data = card_table.get(pokemon.id)
        return bool(data is not None and (getattr(data, "ex", False) or getattr(data, "megaEx", False)))

    def _can_evolve_board_index(self, board_index: int) -> bool:
        for option in self.select.option:
            if option.type != OptionType.EVOLVE:
                continue
            target_index = option.inPlayIndex
            if option.inPlayArea == AreaType.BENCH:
                target_index += 1
            if target_index == board_index:
                return True
        return False

    def _base_attack(self, pokemon: Pokemon, attack_index: int) -> tuple[int, int, int] | None:
        energy_required = 0
        base_damage = 0
        base_score = 0

        if pokemon.id == C.MEGA_LUCARIO_EX:
            if attack_index == 0:
                energy_required = 1
                base_damage = 130
                base_score += 60 * min(3, self.discard_counts[C.BASIC_FIGHTING_ENERGY])
            else:
                energy_required = 2
                base_damage = 270
            if self.my_prizes_left in {2, 3}:
                base_score -= 500
            if self.water_deck and len(self.opponent.prize) <= 3:
                base_score -= 500
        elif attack_index == 1:
            return None
        elif pokemon.id == C.HARIYAMA:
            energy_required = 3
            base_damage = 210
        elif pokemon.id == C.MAKUHITA:
            return None
        elif pokemon.id == C.SOLROCK and self.field_counts[C.LUNATONE] >= 1:
            energy_required = 1
            base_damage = 70

        if base_damage <= 0:
            return None
        return energy_required, base_damage, base_score

    def _base_attack_after_evolution(self, pokemon: Pokemon, board_index: int, attack_index: int):
        if pokemon.id == C.MAKUHITA and attack_index == 0 and self._can_evolve_board_index(board_index):
            return 3, 210, -100
        return self._base_attack(pokemon, attack_index)

    def _plan_attack(self) -> None:
        global plan
        best_score = -1
        plan = AttackPlan()

        if self.state.turn < 2:
            return

        for attacker_index, my_pokemon in enumerate(self._my_board()):
            if my_pokemon is None:
                continue
            if attacker_index != 0 and not self.can_switch:
                break

            for attack_index in range(2):
                attack = self._base_attack_after_evolution(my_pokemon, attacker_index, attack_index)
                if attack is None:
                    continue
                energy_required, base_damage, base_score = attack

                energy_count = len(my_pokemon.energies)
                if attack_index == 1 and attacker_index == 0 and energy_count >= 2 and not self.can_use_mega_brave:
                    break

                needs_energy = False
                if energy_count < energy_required:
                    if self.hand_counts[C.BASIC_FIGHTING_ENERGY] >= 1 and not self.state.energyAttached:
                        energy_count += 1
                        needs_energy = energy_count >= energy_required
                    if not needs_energy:
                        continue

                for target_index, op_pokemon in enumerate(self._opponent_board()):
                    if op_pokemon is None:
                        continue
                    if target_index != 0 and not self.can_gust:
                        break

                    damage = base_damage
                    op_data = card_table.get(op_pokemon.id)
                    if op_data is None:
                        continue
                    if op_data.weakness == EnergyType.FIGHTING:
                        damage *= 2
                    elif op_data.resistance == EnergyType.FIGHTING:
                        damage -= 30

                    crustle_immune = (
                        self.crustle_wall
                        and op_pokemon.id == C.CRUSTLE
                        and self._is_ex_attacker(my_pokemon)
                    )
                    if crustle_immune:
                        damage = 0

                    score = target_score(op_pokemon)
                    prize = prize_count(op_pokemon) if op_pokemon.hp <= damage else 0
                    if prize == 0:
                        score *= damage / max(1, op_pokemon.hp)
                    if len(self.opponent.prize) <= prize:
                        score = 50000

                    if crustle_immune:
                        score = -10000
                    elif self.crustle_wall:
                        if op_pokemon.id == C.CRUSTLE and my_pokemon.id in {C.HARIYAMA, C.MAKUHITA}:
                            score += 2600
                        elif op_pokemon.id == C.CRUSTLE and my_pokemon.id == C.SOLROCK:
                            score += 450
                        active_wall = bool(self.opponent.active and self.opponent.active[0] is not None and self.opponent.active[0].id == C.CRUSTLE)
                        if active_wall and target_index >= 1:
                            score += 700

                    score += base_score
                    score += 220 if attacker_index == 0 else 0
                    score += 300 if target_index == 0 else 0
                    score += energy_count

                    if score > best_score:
                        best_score = score
                        plan = AttackPlan(
                            attacker=attacker_index,
                            target=target_index,
                            attack_index=attack_index,
                            remain_hp=op_pokemon.hp - damage,
                            needs_energy=needs_energy,
                        )

        if self.crustle_wall and _has_attack_plan():
            board = self._my_board()
            targets = self._opponent_board()
            attacker = board[plan.attacker] if 0 <= plan.attacker < len(board) else None
            target = targets[plan.target] if 0 <= plan.target < len(targets) else None
            if isinstance(attacker, Pokemon) and isinstance(target, Pokemon) and target.id == C.CRUSTLE:
                if self._is_ex_attacker(attacker):
                    _DIAG["crustle_final_plan_ex_into_wall"] += 1
                else:
                    _DIAG["crustle_final_plan_non_ex_into_wall"] += 1

    def _energy_target_score(self, pokemon: Pokemon, active: bool) -> int:
        energy_count = len(pokemon.energies)
        score = 8000 + (10 if active else 0)

        if self.crustle_wall:
            if pokemon.id in {C.MAKUHITA, C.HARIYAMA}:
                score += 450
                if len(pokemon.energies) < 3:
                    score += 220
            elif pokemon.id == C.MEGA_LUCARIO_EX:
                score -= 120

        if pokemon.id in {C.MAKUHITA, C.HARIYAMA}:
            score += 1 if pokemon.id == C.HARIYAMA else 0
            score += 100 if energy_count < 3 else 0
            score -= 50 if self.has_ready_hariyama_line else 0
        elif pokemon.id == C.LUNATONE:
            score -= 100
        elif pokemon.id == C.SOLROCK:
            score += 20 if energy_count < 1 else -100
        elif pokemon.id in {C.RIOLU, C.MEGA_LUCARIO_EX}:
            score += 1 if pokemon.id == C.MEGA_LUCARIO_EX else 0
            score += 100 if energy_count < 2 else 0
            score -= 50 if self.has_ready_lucario_line else 0
        return score

    def _score_option(self, option) -> float:
        if option.type == OptionType.NUMBER:
            return option.number
        if option.type == OptionType.YES:
            return 100 if self.context == SelectContext.IS_FIRST else 1
        if option.type == OptionType.NO:
            return 0
        if option.type == OptionType.CARD:
            return self._score_card_choice(option)
        if option.type == OptionType.PLAY:
            return self._score_play(option)
        if option.type == OptionType.ATTACH:
            return self._score_attach(option)
        if option.type == OptionType.EVOLVE:
            return self._score_evolve(option)
        if option.type == OptionType.ABILITY:
            return self._score_ability(option)
        if option.type == OptionType.RETREAT:
            if self.crustle_wall and plan.attacker >= 1:
                return 3200 if _plan_kos() else 2400
            return 2000 if plan.attacker >= 1 else -1
        if option.type == OptionType.ATTACK:
            active = self.me.active[0] if self.me.active else None
            op_active = self.opponent.active[0] if self.opponent.active else None
            if (
                self.crustle_wall
                and isinstance(active, Pokemon)
                and isinstance(op_active, Pokemon)
                and self._is_ex_attacker(active)
                and op_active.id == C.CRUSTLE
                and plan.target < 0
            ):
                _DIAG["crustle_ex_attack_suppressed"] += 1
                return -1
            return 1100 if (option.attackId == MEGA_BRAVE) == (plan.attack_index == 1) else 1000
        return 0

    def _score_card_choice(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, option.playerIndex)
        if card is None:
            return 0

        if self.context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
            return self._score_active_choice(option, card)
        if self.context == SelectContext.SETUP_ACTIVE_POKEMON:
            return self._score_setup_active(card)
        if self.context == SelectContext.SETUP_BENCH_POKEMON:
            return self._score_setup_bench(card)
        if self.context == SelectContext.TO_HAND:
            return self._score_to_hand(card)
        if self.context == SelectContext.TO_BENCH:
            return self._score_to_bench(card)
        if _ctx_is(self.context, "TO_FIELD"):
            return self._score_to_field(option, card)
        if self.context == SelectContext.DISCARD:
            return self._score_discard(card)
        if self.context in {SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY}:
            return self._score_damage_counter(option, card)
        if _ctx_is(self.context, "DAMAGE", "EFFECT_TARGET"):
            return self._score_effect_target(option, card)
        if _ctx_is(self.context, "HEAL", "REMOVE_DAMAGE_COUNTER"):
            return self._score_heal_target(option, card)
        if _ctx_is(self.context, "EVOLVES_FROM", "EVOLVES_TO"):
            return self._score_evolution_context(option, card)
        if _ctx_is(self.context, "MULLIGAN"):
            return self._score_mulligan(card)
        if self.context == SelectContext.ATTACH_FROM and isinstance(card, Pokemon):
            return self._energy_target_score(card, option.area == AreaType.ACTIVE)
        return 0

    def _score_active_choice(self, option, card: Pokemon | Card) -> float:
        if not isinstance(card, Pokemon):
            return 0

        if option.playerIndex != self.my_index:
            return 100 if option.index == plan.target - 1 else 0

        score = len(card.energies) * 2
        if option.index == plan.attacker - 1:
            score += 100
        if card.id == C.MEGA_LUCARIO_EX:
            late_ex_liability = self.my_prizes_left in {2, 3} or (self.water_deck and len(self.opponent.prize) <= 3)
            score += 8 if late_ex_liability else 20
            if self.crustle_wall:
                score -= 35
        elif card.id == C.HARIYAMA and len(card.energies) >= 2:
            score += 45 if self.crustle_wall else 15
        elif card.id == C.MAKUHITA and len(card.energies) >= 2:
            score += 35 if self.crustle_wall else 10
        elif card.id == C.SOLROCK:
            score += 5
        elif card.id == C.RIOLU:
            score += 4
        return score

    def _score_setup_active(self, card: Pokemon | Card) -> int:
        if card is None:
            return 0
        if card.id == C.SOLROCK:
            return 2 if self.state.firstPlayer == self.my_index else 4
        if card.id == C.RIOLU:
            return 3
        if card.id == C.MAKUHITA:
            return 1
        return 0

    def _score_setup_bench(self, card: Pokemon | Card) -> int:
        if card is None:
            return 0
        if card.id == C.RIOLU:
            return 120 - 25 * self.field_counts[C.RIOLU]
        if card.id == C.SOLROCK:
            return 90 if self.field_counts[C.SOLROCK] == 0 else -1
        if card.id == C.LUNATONE:
            return 80 if self.field_counts[C.LUNATONE] == 0 else -1
        if card.id == C.MAKUHITA:
            score = 65 if self.field_counts[C.MAKUHITA] == 0 else 10
            if self.crustle_wall:
                score += 120 if self.field_counts[C.MAKUHITA] == 0 else 45
            return score
        return 0

    def _score_to_bench(self, card: Pokemon | Card) -> float:
        if card is None:
            return 0
        data = card_table.get(card.id)
        if data is None or data.cardType != CardType.POKEMON:
            return 0
        return self._score_setup_bench(card)

    def _score_discard(self, card: Pokemon | Card) -> float:
        if card is None:
            return 0
        cid = card.id
        # Positive means "safe/desirable to discard". Required discard contexts will
        # still pick the highest values; optional discard contexts will skip <= 0.
        if self.crustle_wall:
            if cid == C.BASIC_FIGHTING_ENERGY and self.hand_counts[cid] <= 2:
                return -120
            if cid in {C.MAKUHITA, C.HARIYAMA, C.SWITCH, C.GRAVITY_MOUNTAIN}:
                return -90
        if cid == C.BASIC_FIGHTING_ENERGY:
            score = 45 if self.hand_counts[cid] >= 2 else 5
            if plan.needs_energy and not self.state.energyAttached:
                score -= 200
            return score
        if self.hand_counts[cid] >= 2:
            return 70
        if cid in {C.LUNATONE, C.SOLROCK} and self.field_counts[cid] >= 1:
            return 55
        if cid == C.GRAVITY_MOUNTAIN and self.stadium_id == C.GRAVITY_MOUNTAIN:
            return 50
        if cid in {C.CARMINE, C.LILLIE_DETERMINATION} and self.state.supporterPlayed:
            return 30
        if cid == C.MEGA_LUCARIO_EX and self.field_counts[C.RIOLU] == 0:
            return -80
        if cid == C.HARIYAMA and self.field_counts[C.MAKUHITA] == 0:
            return -50
        if cid in {C.RIOLU, C.MAKUHITA, C.BOSS_ORDERS, C.HERO_CAPE}:
            return -40
        return 0

    def _is_opponent_option(self, option) -> bool:
        return getattr(option, "playerIndex", self.my_index) == self.op_index

    def _score_damage_counter(self, option, card: Pokemon | Card) -> float:
        if not isinstance(card, Pokemon):
            return 0
        if self._is_opponent_option(option):
            return 10000 + prize_count(card) * 1000 - getattr(card, "hp", 0) + _damage_on(card) * 5
        return -target_score(card)

    def _score_effect_target(self, option, card: Pokemon | Card) -> float:
        if not isinstance(card, Pokemon):
            return self._score_to_hand(card)
        if self._is_opponent_option(option):
            return 2000 + target_score(card) + _damage_on(card) * 8
        score = 300 + len(card.energies) * 50 + len(card.tools) * 40 + _damage_on(card) * 8
        if card.id == C.MEGA_LUCARIO_EX:
            score += 250
        elif card.id in {C.RIOLU, C.HARIYAMA}:
            score += 120
        elif card.id in {C.SOLROCK, C.LUNATONE, C.MAKUHITA}:
            score += 70
        return score

    def _score_heal_target(self, option, card: Pokemon | Card) -> float:
        if not isinstance(card, Pokemon):
            return 0
        damage = _damage_on(card)
        if self._is_opponent_option(option):
            return -1000 - damage
        score = damage * 20 + len(card.energies) * 30 + len(card.tools) * 25
        if card.id == C.MEGA_LUCARIO_EX:
            score += 300
        elif card.id in {C.RIOLU, C.HARIYAMA}:
            score += 120
        return score if damage > 0 else max(0, score // 10)

    def _score_evolution_context(self, option, card: Pokemon | Card) -> float:
        if not isinstance(card, Pokemon):
            return 0
        if self._is_opponent_option(option):
            return target_score(card)
        if card.id in {C.RIOLU, C.MEGA_LUCARIO_EX}:
            return 900 + len(card.energies) * 30
        if card.id in {C.MAKUHITA, C.HARIYAMA}:
            return 650 + len(card.energies) * 30
        return 100 + len(card.energies) * 20

    def _score_to_field(self, option, card: Pokemon | Card) -> float:
        if isinstance(card, Pokemon):
            return self._score_to_bench(card) + self._score_setup_active(card)
        return self._score_to_hand(card)

    def _score_mulligan(self, card: Pokemon | Card) -> float:
        if not isinstance(card, Pokemon):
            return 0
        data = card_table.get(card.id)
        if data is None:
            return 0
        is_basic = (
            getattr(data, "basic", False)
            or (
                data.cardType == CardType.POKEMON
                and not getattr(data, "stage1", False)
                and not getattr(data, "stage2", False)
                and not getattr(data, "megaEx", False)
            )
        )
        if not is_basic:
            return 0
        return self._score_setup_active(card) + self._score_setup_bench(card)

    def _score_to_hand(self, card: Pokemon | Card) -> float:
        if card is None:
            return 0
        score = 200 - self.hand_counts[card.id] * 100
        if self.crustle_wall:
            if card.id == C.MAKUHITA:
                score += 120 if self.field_counts[C.MAKUHITA] == 0 else 30
            elif card.id == C.HARIYAMA:
                score += 140 if self.field_counts[C.MAKUHITA] >= 1 else 25
            elif card.id == C.SWITCH:
                score += 80
            elif card.id in {C.POKE_PAD, C.FIGHTING_GONG}:
                score += 55
            elif card.id == C.GRAVITY_MOUNTAIN:
                score += 80
            elif card.id == C.BASIC_FIGHTING_ENERGY:
                score += 45
            elif card.id == C.BOSS_ORDERS:
                score += 60
        if card.id == C.MAKUHITA:
            score += -10 if self.field_counts[card.id] >= 1 else 10
        elif card.id == C.HARIYAMA:
            score += 20 if self.field_counts[C.MAKUHITA] >= 1 else -20
        elif card.id == C.LUNATONE:
            score += -250 if self.field_counts[card.id] >= 1 else 60
        elif card.id == C.SOLROCK:
            score += -250 if self.field_counts[card.id] >= 1 else 50
        elif card.id == C.RIOLU:
            lucario_line = self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX]
            score += -150 if lucario_line >= 2 else -3 if lucario_line >= 1 else 40
        elif card.id == C.MEGA_LUCARIO_EX:
            score += 40 if self.field_counts[C.RIOLU] >= 1 else -15
        elif card.id == C.BASIC_FIGHTING_ENERGY:
            score += 30 if not ability_used or not self.state.energyAttached else -1
        return score

    def _score_play(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        if card is None:
            return 0
        data = card_table.get(card.id)
        if data is None:
            return 0
        if data.cardType == CardType.POKEMON:
            return self._score_play_pokemon(card)
        return self._score_play_trainer(card)

    def _score_play_pokemon(self, card: Card) -> float:
        score = 20000
        if card.id in {C.LUNATONE, C.SOLROCK} and self.field_counts[card.id] >= 1:
            return -1
        if card.id == C.RIOLU and self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX] >= 2:
            return -1
        return score

    def _score_play_trainer(self, card: Card) -> float:
        if _plan_kos() and card.id in {
            C.DUSK_BALL,
            C.FIGHTING_GONG,
            C.POKE_PAD,
            C.CARMINE,
            C.LILLIE_DETERMINATION,
            C.GRAVITY_MOUNTAIN,
        }:
            return -1
        if card.id == C.SWITCH:
            if plan.attacker > 0 and _plan_kos():
                return 14000
            return 6500 if plan.attacker > 0 else -1
        if card.id == C.PREMIUM_POWER_PRO:
            if self.state.supporterPlayed and _plan_kos():
                return -1
            if not self.can_attack:
                can_bridge_draw = (
                    not self.state.supporterPlayed
                    and self.hand_counts[C.CARMINE] > 0
                    and self.hand_counts[C.LILLIE_DETERMINATION] == 0
                    and not self._low_deck()
                )
                return 3050 if can_bridge_draw else -1
            return 5000
        if card.id == C.BOSS_ORDERS:
            if plan.target >= 1 and _plan_kos():
                return 15000
            if self.crustle_wall and plan.target >= 1:
                return 7600
            return 4200 if plan.target >= 1 else -1
        if card.id == C.CARMINE:
            return -1 if self._low_deck() else 3000
        if card.id == C.LILLIE_DETERMINATION:
            return -1 if self._low_deck() else 3100
        if card.id == C.GRAVITY_MOUNTAIN:
            return self._score_gravity_mountain()
        return 10000

    def _score_gravity_mountain(self) -> float:
        if self.crustle_wall and self.stadium_id != C.GRAVITY_MOUNTAIN:
            return 3600
        opponent_has_stage2 = any(
            pokemon is not None and card_table.get(pokemon.id) is not None and card_table[pokemon.id].stage2
            for pokemon in self._opponent_board()
        )
        if opponent_has_stage2:
            return 3500
        return 1200 if self.stadium_id else -1

    def _low_deck(self) -> bool:
        return self.me.deckCount <= LOW_DECK_COUNT

    def _score_attach(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if card is None or not isinstance(pokemon, Pokemon):
            return 0

        if card.id == C.HERO_CAPE:
            if self.water_deck and not self.crustle_wall:
                if pokemon.id == C.RIOLU:
                    return 12200
                if pokemon.id == C.MEGA_LUCARIO_EX:
                    return 12800
            score = 7000
            if self.crustle_wall:
                if pokemon.id in {C.MAKUHITA, C.HARIYAMA}:
                    score += 350
                elif pokemon.id == C.MEGA_LUCARIO_EX:
                    score -= 200
            if pokemon.id == C.RIOLU:
                score += 100
            elif pokemon.id == C.MEGA_LUCARIO_EX:
                score += 200
            return score

        score = self._energy_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE)
        board_index = option.inPlayIndex if option.inPlayArea == AreaType.ACTIVE else option.inPlayIndex + 1
        if board_index == plan.attacker and plan.needs_energy:
            score += 200
        return score

    def _score_evolve(self, option) -> float:
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return 0
        if pokemon.id == C.MAKUHITA and plan.target == 0 and not self.crustle_wall:
            return -1
        evolve_card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        score = 9000 + len(pokemon.energies)
        if evolve_card is not None:
            if evolve_card.id == C.MEGA_LUCARIO_EX and pokemon.id == C.RIOLU:
                score += 350
            elif evolve_card.id == C.HARIYAMA and pokemon.id == C.MAKUHITA:
                score += 180
                if self.crustle_wall:
                    score += 650
        return score

    def _score_ability(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card is None:
            return 0
        if card.id == C.LUMIOSE_CITY:
            return 1
        if card.id == C.LUNATONE and self._low_deck():
            return -1
        return 30000

    def remember_chosen_side_effects(self, selection: list[int]) -> None:
        global ability_used
        if self.context != SelectContext.MAIN:
            return
        for idx in selection:
            if not (0 <= idx < len(self.select.option)):
                continue
            option = self.select.option[idx]
            if option.type != OptionType.ABILITY:
                continue
            card = get_card(self.obs, option.area, option.index, self.my_index)
            if card is not None and card.id == C.LUNATONE:
                ability_used = True

def agent(obs_dict: dict) -> list[int]:
    global pre_turn, ability_used, plan

    # Deck-selection phase: the engine asks for the deck with select == None.
    # Return the 60-card id list (different return semantics from a normal turn).
    try:
        select_is_none = isinstance(obs_dict, dict) and obs_dict.get("select") is None
    except Exception:
        select_is_none = False
    if select_is_none:
        _DIAG["deck_returns"] += 1
        return list(my_deck)

    _DIAG["decisions"] += 1
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            _DIAG["deck_returns"] += 1
            _DIAG["decisions"] -= 1
            return list(my_deck)

        if obs.current is not None and pre_turn != obs.current.turn:
            pre_turn = obs.current.turn
            ability_used = False
            plan = AttackPlan()

        try:
            policy = LucarioPolicy(obs)
            _diag_observe(obs)
            ranked, scores = policy.rank()
            planned = search_plan(obs_dict, obs, ranked, scores) if USE_SEARCH else None
            ranked_use = ranked if planned is None else planned
            selection = normalize_selection(ranked_use, scores, obs.select)
            policy.remember_chosen_side_effects(selection)
            _diag_observe_choice(obs, selection)
            _DIAG["policy_ok"] += 1
            return selection
        except Exception as exc:
            _diag_record_error(exc)
            _DIAG["policy_fallback"] += 1
            return _legal_fallback(obs.select)
    except Exception as exc:
        _diag_record_error(exc)
        _DIAG["obs_fallback"] += 1
        return _legal_fallback_from_dict(obs_dict if isinstance(obs_dict, dict) else {})
