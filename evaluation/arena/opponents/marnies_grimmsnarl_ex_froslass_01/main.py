"""Marnie's Grimmsnarl ex rule proxy inspired by kazuki0123 public replays.

The policy is a compact imitation shell:
  - establish Impidimp -> Grimmsnarl ex quickly,
  - use Darkness Energy/Punk Up to power Shadow Bullet,
  - use Dudunsparce/Lillie/Dawn/Poke Pad/Spikemuth as the resource engine,
  - use Xerosic only when the opponent's hand is worth punishing.

All final actions are normalized against the legal option list.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    ROOT = __file__
except NameError:
    ROOT = None

PACKAGE_ROOT = Path(__file__).resolve().parent

CG_PATH = "/kaggle_simulations/agent"
for path in ([os.path.dirname(os.path.abspath(ROOT))] if ROOT else []) + [CG_PATH]:
    if path and path not in sys.path and os.path.isdir(path):
        sys.path.insert(0, path)

from cg.api import AreaType, OptionType, SelectContext, all_card_data, to_observation_class

try:
    from cg.api import all_attack

    ALL_ATTACKS = {int(a.attackId): a for a in all_attack()}
except Exception:
    ALL_ATTACKS = {}


CARD_DB = {int(c.cardId): c for c in all_card_data()}

DARK_ENERGY = 7

MUNKIDORI = 112
DUNSPARCE = 305
DUDUNSPARCE = 66
IMPIDIMP = 646
MORGREM = 647
GRIMMSNARL_EX = 648
MORPEKO = 649

RARE_CANDY = 1079
BUDDY_POFFIN = 1086
NIGHT_STRETCHER = 1097
ENERGY_SEARCH = 1119
ENERGY_RECYCLER = 1139
POKE_PAD = 1152
HERO_CAPE = 1159
XEROSIC = 1197
LILLIE = 1227
DAWN = 1231
SPIKEMUTH_GYM = 1259

FILCH = 934
IMPIDIMP_PUNCH = 935
MORGREM_PUNCH = 936
SHADOW_BULLET = 937
SPIKY_WHEEL = 938
MIND_BEND = 141
TRADING_PLACES = 423
LAND_CRUSH = 76

MARNIE_LINE = {IMPIDIMP, MORGREM, GRIMMSNARL_EX, MORPEKO}
GRIMMSNARL_LINE = {IMPIDIMP, MORGREM, GRIMMSNARL_EX}
ENGINE_BASICS = {IMPIDIMP, DUNSPARCE, MUNKIDORI, MORPEKO}
DRAW_SUPPORTERS = {LILLIE, DAWN, XEROSIC}

ARCHALUDON_LINE = {169, 190}
DURALUDON = 169
ARCHALUDON_EX = 190
CINDERACE = 666
RELICANTH = 57
ALAKAZAM_LINE = {741, 742, 743}
DUDUNSPARCE_LINE = {65, 66, 305}
STARMIE_LINE = {1030, 1031}
DRAGAPULT_LINE = {119, 120, 121}
LUCARIO_LINE = {677, 678}
CRUSTLE_LINE = {344, 345, 532}
HOP_LINE = {288, 289, 299, 304, 307, 308, 309, 310, 878, 879}

KEY_BENCH_TARGETS = (
    ALAKAZAM_LINE
    | DUDUNSPARCE_LINE
    | STARMIE_LINE
    | DRAGAPULT_LINE
    | LUCARIO_LINE
    | HOP_LINE
    | {MUNKIDORI}
)


def read_deck_csv() -> list[int]:
    for filename in (PACKAGE_ROOT / "deck.csv", Path("/kaggle_simulations/agent/deck.csv")):
        if os.path.exists(filename):
            with open(filename, encoding="utf-8") as f:
                return [int(line) for line in f.read().splitlines() if line.strip()]
    return []


def _card_id(card) -> int | None:
    if card is None:
        return None
    for attr in ("id", "cardId"):
        value = getattr(card, attr, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                return None
    return None


def _get_cards(cards) -> list:
    return [card for card in (cards or []) if card is not None]


def my_state(obs):
    return obs.current.players[obs.current.yourIndex]


def opp_state(obs):
    return obs.current.players[1 - obs.current.yourIndex]


def get_card(obs, area, index, player_index):
    if area is None or index is None:
        return None
    try:
        ps = obs.current.players[player_index]
        if area == AreaType.DECK and obs.select and obs.select.deck is not None:
            return obs.select.deck[index] if index < len(obs.select.deck) else None
        if area == AreaType.HAND and ps.hand is not None:
            return ps.hand[index] if index < len(ps.hand) else None
        if area == AreaType.DISCARD:
            return ps.discard[index] if index < len(ps.discard) else None
        if area == AreaType.ACTIVE:
            return ps.active[index] if index < len(ps.active) else None
        if area == AreaType.BENCH:
            return ps.bench[index] if index < len(ps.bench) else None
        if area == AreaType.PRIZE:
            return ps.prize[index] if index < len(ps.prize) else None
        if area == AreaType.STADIUM:
            return obs.current.stadium[index] if index < len(obs.current.stadium) else None
        if area == AreaType.LOOKING and obs.current.looking is not None:
            return obs.current.looking[index] if index < len(obs.current.looking) else None
    except Exception:
        return None
    return None


def option_card(obs, opt):
    yi = obs.current.yourIndex
    pi = getattr(opt, "playerIndex", None)
    if pi is None:
        pi = yi
    if opt.type == OptionType.PLAY:
        return get_card(obs, AreaType.HAND, getattr(opt, "index", None), pi)
    return get_card(obs, getattr(opt, "area", None), getattr(opt, "index", None), pi)


def option_card_id(obs, opt) -> int | None:
    card = option_card(obs, opt)
    cid = _card_id(card)
    if cid is not None:
        return cid
    raw = getattr(opt, "cardId", None)
    try:
        return int(raw) if raw is not None else None
    except Exception:
        return None


def option_target(obs, opt):
    area = getattr(opt, "inPlayArea", None)
    index = getattr(opt, "inPlayIndex", None)
    if area is None or index is None:
        return None
    pi = getattr(opt, "playerIndex", None)
    if pi is None:
        pi = obs.current.yourIndex
    return get_card(obs, area, index, pi)


def all_my_pokemon(obs) -> list:
    ps = my_state(obs)
    return _get_cards(ps.active) + _get_cards(ps.bench)


def all_opp_visible_pokemon(obs) -> list:
    ps = opp_state(obs)
    return _get_cards(ps.active) + _get_cards(ps.bench)


def my_active(obs):
    active = _get_cards(my_state(obs).active)
    return active[0] if active else None


def opp_active(obs):
    active = _get_cards(opp_state(obs).active)
    return active[0] if active else None


def my_bench(obs) -> list:
    return _get_cards(my_state(obs).bench)


def opp_bench(obs) -> list:
    return _get_cards(opp_state(obs).bench)


def hand_ids(obs) -> list[int]:
    return [_card_id(c) for c in _get_cards(my_state(obs).hand) if _card_id(c) is not None]


def discard_ids(obs) -> list[int]:
    return [_card_id(c) for c in _get_cards(my_state(obs).discard) if _card_id(c) is not None]


def energy_count(pokemon) -> int:
    if pokemon is None:
        return 0
    for attr in ("energyCards", "energies"):
        cards = getattr(pokemon, attr, None)
        if cards is not None:
            return len(cards)
    return 0


def dark_energy_count(pokemon) -> int:
    if pokemon is None:
        return 0
    total = 0
    for attr in ("energyCards", "energies"):
        cards = getattr(pokemon, attr, None)
        if cards is not None:
            for card in cards:
                if _card_id(card) == DARK_ENERGY:
                    total += 1
            return total
    return 0


def damage_on(pokemon) -> int:
    if pokemon is None:
        return 0
    max_hp = getattr(pokemon, "maxHp", None)
    if max_hp is None:
        data = CARD_DB.get(_card_id(pokemon))
        max_hp = getattr(data, "hp", None) if data else None
    hp = getattr(pokemon, "hp", None)
    if max_hp is None or hp is None:
        return 0
    return max(0, int(max_hp) - int(hp))


def has_tool(pokemon) -> bool:
    return bool(getattr(pokemon, "tools", []) or [])


def count_in_play(obs, ids) -> int:
    if isinstance(ids, int):
        ids = {ids}
    return sum(1 for p in all_my_pokemon(obs) if _card_id(p) in ids)


def has_in_play(obs, ids) -> bool:
    return count_in_play(obs, ids) > 0


def ready_grimmsnarls(obs) -> list:
    return [p for p in all_my_pokemon(obs) if _card_id(p) == GRIMMSNARL_EX and dark_energy_count(p) >= 2]


def any_grimmsnarl_pressure(obs) -> bool:
    if ready_grimmsnarls(obs):
        return True
    return any(_card_id(p) in {MORGREM, GRIMMSNARL_EX} for p in all_my_pokemon(obs))


def need_impidimp(obs) -> bool:
    return count_in_play(obs, GRIMMSNARL_LINE) < 2


def need_dunsparce(obs) -> bool:
    return count_in_play(obs, {DUNSPARCE, DUDUNSPARCE}) < 2


def need_munkidori(obs) -> bool:
    if detect_matchup(obs) == "crustle":
        return count_in_play(obs, MUNKIDORI) < 2
    return count_in_play(obs, MUNKIDORI) < 1


def need_morpeko(obs) -> bool:
    return count_in_play(obs, MORPEKO) < 1 and len(my_bench(obs)) >= 2


def own_dark_energy_in_hand(obs) -> int:
    return hand_ids(obs).count(DARK_ENERGY)


def dark_energy_in_discard(obs) -> int:
    return discard_ids(obs).count(DARK_ENERGY)


def visible_opponent_ids(obs) -> set[int]:
    ids = set()
    opp = opp_state(obs)
    for zone in (opp.active, opp.bench, opp.discard):
        for card in _get_cards(zone):
            cid = _card_id(card)
            if cid is not None:
                ids.add(cid)
    return ids


def detect_matchup(obs) -> str:
    ids = visible_opponent_ids(obs)
    if ids & CRUSTLE_LINE:
        return "crustle"
    if ids & ARCHALUDON_LINE:
        return "archaludon"
    if ids & STARMIE_LINE:
        return "starmie"
    if ids & ALAKAZAM_LINE:
        return "alakazam"
    if ids & DRAGAPULT_LINE:
        return "dragapult"
    if ids & LUCARIO_LINE:
        return "lucario"
    if ids & HOP_LINE:
        return "hop"
    if ids & DUDUNSPARCE_LINE:
        return "dudunsparce"
    return "generic"


def opponent_hand_count(obs) -> int:
    value = getattr(opp_state(obs), "handCount", None)
    return int(value or 0)


def effect_id(obs) -> int | None:
    effect = getattr(obs.select, "effect", None)
    cid = _card_id(effect)
    if cid is not None:
        return cid
    raw = getattr(effect, "id", None)
    try:
        return int(raw) if raw is not None else None
    except Exception:
        return None


def effect_name(obs) -> str:
    effect = getattr(obs.select, "effect", None)
    name = getattr(effect, "name", "") if effect is not None else ""
    return str(name or "").lower()


def prize_value(pokemon) -> int:
    cid = _card_id(pokemon)
    data = CARD_DB.get(cid)
    if data and getattr(data, "ex", False):
        return 2
    return 1


def hp_left(pokemon) -> int:
    hp = getattr(pokemon, "hp", None)
    if hp is None:
        data = CARD_DB.get(_card_id(pokemon))
        hp = getattr(data, "hp", 999) if data else 999
    return int(hp)


def effective_damage(base_damage: int, target) -> int:
    if target is None:
        return base_damage
    weakness = getattr(CARD_DB.get(_card_id(target)), "weakness", None)
    if weakness == 7:
        return base_damage * 2
    return base_damage


def setup_score(obs, opt):
    ctx = obs.select.context
    cid = option_card_id(obs, opt)

    if ctx == SelectContext.MULLIGAN:
        return (10000, "keep opening hand") if opt.type == OptionType.NO else (0, "mulligan")

    if ctx == SelectContext.IS_FIRST:
        return (12000, "choose first for evolution tempo") if opt.type == OptionType.YES else (8000, "choose second")

    if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
        priority = {
            IMPIDIMP: 30000,
            MORPEKO: 24000,
            DUNSPARCE: 22000,
            MUNKIDORI: 12000,
        }
        return priority.get(cid, 0), "setup active"

    if ctx == SelectContext.SETUP_BENCH_POKEMON:
        priority = {
            IMPIDIMP: 28000,
            DUNSPARCE: 23000,
            MUNKIDORI: 18000,
            MORPEKO: 14000,
        }
        return priority.get(cid, -1000), "setup bench"

    return 0, "setup fallback"


def score_play(obs, opt):
    cid = option_card_id(obs, opt)
    ids = hand_ids(obs)
    deck_count = int(getattr(my_state(obs), "deckCount", 0) or 0)
    supporter_played = bool(getattr(obs.current, "supporterPlayed", False))

    if cid == IMPIDIMP:
        return (30000 if need_impidimp(obs) else 8000), "bench Impidimp"
    if cid == DUNSPARCE:
        return (25000 if need_dunsparce(obs) else 6000), "bench Dunsparce"
    if cid == MUNKIDORI:
        return (21000 if need_munkidori(obs) else 4000), "bench Munkidori"
    if cid == MORPEKO:
        return (12000 if need_morpeko(obs) else 2000), "bench Morpeko backup"

    if cid == SPIKEMUTH_GYM:
        if deck_count <= 1:
            return -500, "skip stadium with empty deck"
        return 27000, "play Spikemuth Gym"

    if cid == BUDDY_POFFIN:
        if deck_count <= 1:
            return -1000, "skip Poffin empty deck"
        need = int(need_impidimp(obs)) + int(need_dunsparce(obs)) + int(need_munkidori(obs))
        return (28000 + need * 1000 if need else 6000), "play Poffin for setup"

    if cid == POKE_PAD:
        if deck_count <= 1:
            return -1000, "skip Poke Pad empty deck"
        if need_impidimp(obs) or MORGREM not in ids:
            return 24000, "Poke Pad for evolution line"
        if need_dunsparce(obs) or need_munkidori(obs):
            return 20000, "Poke Pad for engine"
        return 7000, "Poke Pad value"

    if cid == RARE_CANDY:
        if GRIMMSNARL_EX in ids and has_in_play(obs, IMPIDIMP):
            return 31000, "Rare Candy to Grimmsnarl"
        if has_in_play(obs, IMPIDIMP):
            return 17000, "Rare Candy setup"
        return -200, "save Rare Candy"

    if cid == ENERGY_SEARCH:
        if own_dark_energy_in_hand(obs) == 0 and not bool(getattr(obs.current, "energyAttached", False)):
            return 23000, "Energy Search for attach"
        return 3000, "Energy Search optional"

    if cid == HERO_CAPE:
        targets = [p for p in all_my_pokemon(obs) if _card_id(p) == GRIMMSNARL_EX and not has_tool(p)]
        if targets:
            return 26000, "Hero Cape on Grimmsnarl"
        morpeko = [p for p in all_my_pokemon(obs) if _card_id(p) == MORPEKO and dark_energy_count(p) >= 3 and not has_tool(p)]
        if morpeko:
            return 9000, "Hero Cape backup Morpeko"
        return -1000, "save Hero Cape"

    if cid == NIGHT_STRETCHER:
        disc = discard_ids(obs)
        urgent = (
            GRIMMSNARL_EX in disc
            or (IMPIDIMP in disc and need_impidimp(obs))
            or (DARK_ENERGY in disc and own_dark_energy_in_hand(obs) == 0 and any_grimmsnarl_pressure(obs))
            or (MUNKIDORI in disc and need_munkidori(obs))
        )
        return (19000 if urgent else 1500), "Night Stretcher resources"

    if cid == ENERGY_RECYCLER:
        darks = dark_energy_in_discard(obs)
        if darks >= 3:
            return 21000, "Energy Recycler long game"
        if darks >= 2 and deck_count <= 8:
            return 15000, "Energy Recycler low deck"
        return -500, "save Energy Recycler"

    if cid == DAWN:
        if supporter_played:
            return -1000, "supporter already played"
        if deck_count <= 2:
            return -2000, "Dawn empty deck risk"
        if not has_in_play(obs, GRIMMSNARL_EX):
            return 30000, "Dawn assembles Grimmsnarl line"
        if need_dunsparce(obs) or need_munkidori(obs):
            return 17000, "Dawn finds engine"
        return 9000, "Dawn optional"

    if cid == LILLIE:
        if supporter_played:
            return -1000, "supporter already played"
        hand_count = int(getattr(my_state(obs), "handCount", 0) or 0)
        if deck_count <= 2:
            return -4000, "avoid Lillie deckout"
        if deck_count <= 5 and hand_count < 7:
            return -2000, "avoid thin-deck Lillie"
        if not has_in_play(obs, GRIMMSNARL_EX) or hand_count <= 4:
            return 25000, "Lillie dig"
        return 12000, "Lillie refresh"

    if cid == XEROSIC:
        if supporter_played:
            return -1000, "supporter already played"
        opp_hand = opponent_hand_count(obs)
        pressure = any_grimmsnarl_pressure(obs)
        if opp_hand >= 8 and pressure:
            return 28000, "Xerosic punish big hand"
        if opp_hand >= 7 and ready_grimmsnarls(obs):
            return 23000, "Xerosic with pressure"
        if detect_matchup(obs) in {"alakazam", "archaludon", "starmie"} and opp_hand >= 6 and pressure:
            return 18000, "Xerosic meta pressure"
        return -1500, "save Xerosic"

    return 1000, "generic play"


def score_evolve(obs, opt):
    cid = option_card_id(obs, opt)
    target = option_target(obs, opt)
    tid = _card_id(target)

    if cid == GRIMMSNARL_EX:
        bonus = 7000 if tid == IMPIDIMP else 0
        return 42000 + bonus, "evolve Grimmsnarl ex"
    if cid == MORGREM:
        if GRIMMSNARL_EX in hand_ids(obs):
            return 31000, "evolve Morgrem with ex in hand"
        return 26000, "evolve Morgrem"
    if cid == DUDUNSPARCE:
        return 21000, "evolve Dudunsparce engine"
    return 1000, "generic evolve"


def attach_target_score(obs, target, area=None) -> int:
    tid = _card_id(target)
    if target is None:
        return 0
    if tid == GRIMMSNARL_EX:
        darks = dark_energy_count(target)
        if darks < 2:
            return 36000 - darks * 1000
        return 16000
    if tid == MORGREM:
        return 24000 if dark_energy_count(target) < 2 else 6000
    if tid == IMPIDIMP:
        return 19000 if dark_energy_count(target) < 1 else 3000
    if tid == MORPEKO:
        darks = dark_energy_count(target)
        return 15000 + darks * 800 if darks < 4 else 5000
    if tid == MUNKIDORI:
        if dark_energy_count(target) < 1:
            if any(damage_on(p) for p in all_my_pokemon(obs)):
                return 30000
            if ready_grimmsnarls(obs):
                return 18000
            return 9000
        return 3000
    if tid == DUDUNSPARCE:
        return 2500
    if tid == DUNSPARCE:
        return 1500
    return 500


def punk_up_attach_target_score(obs, target) -> int:
    tid = _card_id(target)
    if target is None:
        return 0
    darks = dark_energy_count(target)
    if tid == GRIMMSNARL_EX:
        if darks < 2:
            return 50000 - darks * 2000
        return 7000
    if tid == MORGREM:
        if darks < 2:
            return 36000 - darks * 1500
        return 5000
    if tid == IMPIDIMP:
        if darks < 1:
            return 26000
        return 4000
    if tid == MORPEKO:
        if darks < 3:
            return 34000 - darks * 1000
        return 8000
    if tid == MUNKIDORI:
        if darks < 1:
            return 30000 if any(damage_on(p) > 0 for p in all_my_pokemon(obs)) else 24000
        return 5000
    return attach_target_score(obs, target)


def score_attach(obs, opt):
    source_cid = option_card_id(obs, opt)
    target = option_target(obs, opt)
    if source_cid is not None and source_cid != DARK_ENERGY:
        return 1000, "attach non-dark"
    if bool(getattr(obs.current, "energyAttached", False)):
        return -1000, "energy already attached"
    return attach_target_score(obs, target), "manual Darkness attach"


def attack_score(obs, attack_id):
    try:
        aid = int(attack_id)
    except Exception:
        return 0

    active = my_active(obs)
    active_id = _card_id(active)
    opp = opp_active(obs)

    if aid == SHADOW_BULLET:
        base = 180
        score = 1000 + min(base, hp_left(opp)) + 200
        if opp and effective_damage(base, opp) >= hp_left(opp):
            score += 500
        if opp_bench(obs):
            score += 160
        return score

    if aid == SPIKY_WHEEL:
        dmg = 20 + 40 * dark_energy_count(active)
        score = 400 + dmg
        if opp and effective_damage(dmg, opp) >= hp_left(opp):
            score += 400
        if active_id == MORPEKO and dark_energy_count(active) >= 3:
            score += 150
        return score

    if aid == MORGREM_PUNCH:
        return 450
    if aid == MIND_BEND:
        return 420
    if aid == LAND_CRUSH:
        return 380
    if aid == TRADING_PLACES:
        return 360 if ready_grimmsnarls(obs) else 80
    if aid == FILCH:
        deck_count = int(getattr(my_state(obs), "deckCount", 0) or 0)
        return 260 if deck_count > 6 else -100
    if aid == IMPIDIMP_PUNCH:
        return 120

    attack = ALL_ATTACKS.get(aid)
    return int(getattr(attack, "damage", 0) or 0)


def score_retreat(obs, opt):
    active = my_active(obs)
    active_id = _card_id(active)
    if active_id == GRIMMSNARL_EX and dark_energy_count(active) >= 2:
        return -2000, "keep attacking Grimmsnarl active"
    if ready_grimmsnarls(obs):
        return 18000, "retreat into ready Grimmsnarl"
    if active_id == MORPEKO:
        return 6000, "free pivot retreat"
    if active_id in {DUNSPARCE, MUNKIDORI, IMPIDIMP} and any_grimmsnarl_pressure(obs):
        return 7000, "retreat support Pokemon"
    return -100, "avoid retreat"


def score_to_hand(obs, opt):
    cid = option_card_id(obs, opt)
    eid = effect_id(obs)
    ids = hand_ids(obs)

    if eid == DAWN:
        if cid == GRIMMSNARL_EX:
            return 30000, "Dawn take Grimmsnarl ex"
        if cid == MORGREM:
            return 26000, "Dawn take Morgrem"
        if cid == IMPIDIMP and need_impidimp(obs):
            return 25000, "Dawn take Impidimp"
        if cid == DUNSPARCE and need_dunsparce(obs):
            return 20000, "Dawn take Dunsparce"
        if cid == MUNKIDORI and need_munkidori(obs):
            return 18000, "Dawn take Munkidori"
        if cid == DUDUNSPARCE:
            return 17000, "Dawn take Dudunsparce"
        return 1000, "Dawn fallback"

    if eid == POKE_PAD:
        if cid == MORGREM and has_in_play(obs, IMPIDIMP):
            return 27000, "Poke Pad take Morgrem"
        if cid == IMPIDIMP and need_impidimp(obs):
            return 25000, "Poke Pad take Impidimp"
        if cid == DUDUNSPARCE and has_in_play(obs, DUNSPARCE):
            return 22000, "Poke Pad take Dudunsparce"
        if cid == MUNKIDORI and need_munkidori(obs):
            return 20000, "Poke Pad take Munkidori"
        if cid == DUNSPARCE and need_dunsparce(obs):
            return 18000, "Poke Pad take Dunsparce"
        if cid == MORPEKO and need_morpeko(obs):
            return 12000, "Poke Pad take Morpeko"
        return 1000, "Poke Pad fallback"

    if eid == SPIKEMUTH_GYM:
        if cid == GRIMMSNARL_EX and (MORGREM in ids or has_in_play(obs, MORGREM) or has_in_play(obs, IMPIDIMP)):
            return 30000, "Spikemuth take Grimmsnarl ex"
        if cid == MORGREM and has_in_play(obs, IMPIDIMP):
            return 26000, "Spikemuth take Morgrem"
        if cid == IMPIDIMP and need_impidimp(obs):
            return 22000, "Spikemuth take Impidimp"
        if cid == MORPEKO and need_morpeko(obs):
            return 12000, "Spikemuth take Morpeko"
        return 1000, "Spikemuth fallback"

    if eid == ENERGY_SEARCH:
        return (30000, "take Darkness Energy") if cid == DARK_ENERGY else (0, "skip non-energy")

    if eid == NIGHT_STRETCHER:
        if cid == GRIMMSNARL_EX:
            return 28000, "recover Grimmsnarl"
        if cid == IMPIDIMP and need_impidimp(obs):
            return 24000, "recover Impidimp"
        if cid == DARK_ENERGY and own_dark_energy_in_hand(obs) == 0:
            return 23000, "recover Darkness Energy"
        if cid in {MORGREM, MUNKIDORI, DUNSPARCE, DUDUNSPARCE}:
            return 14000, "recover engine"
        return 1000, "recover fallback"

    if cid == GRIMMSNARL_EX:
        return 24000, "take Grimmsnarl"
    if cid == MORGREM:
        return 22000, "take Morgrem"
    if cid == IMPIDIMP and need_impidimp(obs):
        return 21000, "take Impidimp"
    if cid == DARK_ENERGY:
        return 17000, "take Darkness Energy"
    if cid in {DUNSPARCE, DUDUNSPARCE, MUNKIDORI}:
        return 15000, "take engine"
    if cid == RARE_CANDY:
        return 14000, "take Rare Candy"
    if cid in DRAW_SUPPORTERS and not bool(getattr(obs.current, "supporterPlayed", False)):
        return 10000, "take supporter"
    if cid == HERO_CAPE:
        return 9000, "take Hero Cape"
    return 1000, "generic take"


def score_discard(obs, opt):
    cid = option_card_id(obs, opt)
    pi = getattr(opt, "playerIndex", None)
    if pi is not None and pi != obs.current.yourIndex:
        card = option_card(obs, opt)
        value = 10000
        if cid in {GRIMMSNARL_EX, MORGREM, IMPIDIMP}:
            value += 7000
        if cid in DRAW_SUPPORTERS:
            value += 5000
        if cid == DARK_ENERGY:
            value += 2500
        if card is not None:
            value += max(0, 300 - hp_left(card))
        return value, "opponent discard high value"

    ids = hand_ids(obs)
    if cid == DARK_ENERGY:
        if dark_energy_in_discard(obs) < 2 and ids.count(DARK_ENERGY) > 1:
            return 12000, "discard spare Darkness"
        return -3000, "keep Darkness"
    if cid in {GRIMMSNARL_EX, MORGREM, IMPIDIMP}:
        if ids.count(cid) > 1:
            return 3000, "discard duplicate line"
        return -5000, "keep evolution line"
    if cid in {MUNKIDORI, DUNSPARCE, DUDUNSPARCE, MORPEKO}:
        if has_in_play(obs, cid) or ids.count(cid) > 1:
            return 5000, "discard extra engine"
        return -2000, "keep engine"
    if cid in {SPIKEMUTH_GYM, POKE_PAD, BUDDY_POFFIN} and ids.count(cid) > 1:
        return 9000, "discard duplicate utility"
    if cid in DRAW_SUPPORTERS and ids.count(cid) > 1:
        return 7000, "discard duplicate supporter"
    if cid == XEROSIC and opponent_hand_count(obs) < 6:
        return 8000, "discard inactive Xerosic"
    return 1000, "generic discard"


def own_target_score(obs, card, ctx) -> int:
    cid = _card_id(card)
    if ctx in {SelectContext.HEAL, SelectContext.REMOVE_DAMAGE_COUNTER}:
        if cid == GRIMMSNARL_EX:
            return 26000 + damage_on(card)
        if cid == MUNKIDORI:
            return 16000 + damage_on(card)
        return damage_on(card)

    if ctx in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
        if cid == GRIMMSNARL_EX and dark_energy_count(card) >= 2:
            return 30000, "promote ready Grimmsnarl"
        if cid == MORPEKO:
            return 16000 + dark_energy_count(card) * 500, "promote Morpeko pivot"
        if cid == GRIMMSNARL_EX:
            return 15000, "promote Grimmsnarl"
        if cid == MORGREM:
            return 8000, "promote Morgrem"
        if cid == IMPIDIMP:
            return 5000, "promote Impidimp"
        if cid == DUNSPARCE:
            return 4500, "promote Dunsparce"
        if cid == MUNKIDORI:
            return 3500, "promote Munkidori"
        return 1000, "promote fallback"

    if ctx in {SelectContext.ATTACH_TO, SelectContext.ATTACH_FROM}:
        if (
            ctx == SelectContext.ATTACH_FROM
            and effect_id(obs) == GRIMMSNARL_EX
            and cid == MUNKIDORI
            and dark_energy_count(card) < 1
        ):
            return 28000
        return attach_target_score(obs, card)

    if ctx in {SelectContext.TO_FIELD, SelectContext.TO_BENCH}:
        if cid == GRIMMSNARL_EX:
            return 30000
        if cid == MORGREM:
            return 24000
        if cid == IMPIDIMP:
            return 22000
        if cid == DUNSPARCE:
            return 18000
        if cid == MUNKIDORI:
            return 17000
        if cid == MORPEKO:
            return 9000

    return 1000


def opponent_damage_target_score(obs, card) -> int:
    cid = _card_id(card)
    if card is None:
        return 0
    hp = hp_left(card)
    score = 10000 - min(hp, 400)
    if hp <= 30:
        score += 12000
    if hp <= 60:
        score += 4000
    if detect_matchup(obs) == "archaludon":
        if cid == DURALUDON:
            score += 9000
        elif cid == ARCHALUDON_EX:
            score += 3500
        elif cid == CINDERACE:
            score += 2500
        elif cid == RELICANTH:
            score -= 2500
    if cid in KEY_BENCH_TARGETS:
        score += 3000
    if cid in {GRIMMSNARL_EX, 190, 743, 1031, 121, 678}:
        score += 2000
    score += prize_value(card) * 1000
    score += energy_count(card) * 300
    return score


def score_target(obs, opt):
    ctx = obs.select.context
    card = option_card(obs, opt)
    cid = _card_id(card)
    pi = getattr(opt, "playerIndex", None)
    if pi is None:
        pi = obs.current.yourIndex

    if ctx == SelectContext.ATTACH_TO and cid == DARK_ENERGY:
        return 30000, "select Darkness Energy"

    damage_contexts = {
        SelectContext.DAMAGE,
        SelectContext.DAMAGE_COUNTER,
        SelectContext.DAMAGE_COUNTER_ANY,
        SelectContext.EFFECT_TARGET,
        SelectContext.SWITCH,
        SelectContext.TO_ACTIVE,
    }

    if pi != obs.current.yourIndex:
        if ctx in damage_contexts:
            return opponent_damage_target_score(obs, card), "opponent target"
        return 1000, "opponent target fallback"

    result = own_target_score(obs, card, ctx)
    if isinstance(result, tuple):
        return result
    return result, "own target"


def score_number(obs, opt):
    number = int(getattr(opt, "number", 0) or 0)
    if effect_id(obs) == ENERGY_RECYCLER:
        return number, "recycle as many as offered"
    if "adrena" in effect_name(obs) or obs.select.context in {
        SelectContext.DAMAGE_COUNTER_COUNT,
        SelectContext.REMOVE_DAMAGE_COUNTER_COUNT,
    }:
        return number, "move max damage counters"
    return number, "number"


def score_yes_no(obs, opt):
    ctx = obs.select.context
    if ctx == SelectContext.IS_FIRST:
        return setup_score(obs, opt)
    if ctx == SelectContext.ACTIVATE:
        name = effect_name(obs)
        deck_count = int(getattr(my_state(obs), "deckCount", 0) or 0)
        if "punk up" in name:
            return (100000, "activate Punk Up") if opt.type == OptionType.YES else (-100000, "do not skip Punk Up")
        if "adrena" in name:
            has_damage = any(damage_on(p) > 0 for p in all_my_pokemon(obs))
            return (60000 if has_damage and opt.type == OptionType.YES else -1000), "Adrena-Brain"
        if "run away draw" in name:
            safe = deck_count > 6 and len(all_my_pokemon(obs)) >= 2
            if opt.type == OptionType.YES and safe:
                return 26000, "Run Away Draw"
            return (-3000 if opt.type == OptionType.YES else 500), "skip unsafe Run Away Draw"
        if "spikemuth" in name:
            return (30000, "use Spikemuth") if opt.type == OptionType.YES else (-1000, "skip Spikemuth")
        return (1000, "yes") if opt.type == OptionType.YES else (0, "no")
    return (1, "yes") if opt.type == OptionType.YES else (0, "no")


def apply_overrides(obs, opt, score, reason):
    ctx = obs.select.context
    cid = option_card_id(obs, opt)
    matchup = detect_matchup(obs)
    deck_count = int(getattr(my_state(obs), "deckCount", 0) or 0)

    if deck_count <= 2 and opt.type == OptionType.PLAY and cid in {LILLIE, DAWN, BUDDY_POFFIN, POKE_PAD}:
        return -5000, "hard low-deck draw/search guard"

    if matchup == "crustle":
        if opt.type == OptionType.PLAY and cid == XEROSIC and opponent_hand_count(obs) < 9:
            return -3000, "Crustle: save Xerosic"
        if opt.type == OptionType.ATTACK and getattr(opt, "attackId", None) == SHADOW_BULLET:
            return score + 100, "Crustle: still pressure with Shadow Bullet"
        if opt.type == OptionType.PLAY and cid in {ENERGY_RECYCLER, NIGHT_STRETCHER}:
            return score + 5000, "Crustle: resource loop"
        if opt.type == OptionType.PLAY and cid == MUNKIDORI:
            return score + 4000, "Crustle: Munkidori utility"

    return score, reason


MAIN_DISPATCH = {
    OptionType.PLAY: score_play,
    OptionType.EVOLVE: score_evolve,
    OptionType.ATTACH: score_attach,
    OptionType.RETREAT: score_retreat,
}


def score_option(obs, opt):
    ctx = obs.select.context

    if ctx in {
        SelectContext.IS_FIRST,
        SelectContext.MULLIGAN,
        SelectContext.SETUP_ACTIVE_POKEMON,
        SelectContext.SETUP_BENCH_POKEMON,
    }:
        return setup_score(obs, opt)

    if opt.type in {OptionType.YES, OptionType.NO}:
        return score_yes_no(obs, opt)

    if opt.type == OptionType.NUMBER:
        return score_number(obs, opt)

    if ctx == SelectContext.MAIN:
        fn = MAIN_DISPATCH.get(opt.type)
        if fn is not None:
            score, reason = fn(obs, opt)
        elif opt.type == OptionType.ABILITY:
            name = effect_name(obs)
            cid = option_card_id(obs, opt)
            if cid == SPIKEMUTH_GYM or "spikemuth" in name:
                if getattr(my_state(obs), "deckCount", 0) and (
                    need_impidimp(obs) or not has_in_play(obs, GRIMMSNARL_EX) or need_morpeko(obs)
                ):
                    score, reason = 30000, "use Spikemuth search"
                else:
                    score, reason = 9000, "use Spikemuth value"
            elif cid == MUNKIDORI or "adrena" in name:
                has_damage = any(damage_on(p) > 0 for p in all_my_pokemon(obs))
                score, reason = (24000 if has_damage else 6000), "use Munkidori"
            elif cid == DUDUNSPARCE or "run away draw" in name:
                deck_count = int(getattr(my_state(obs), "deckCount", 0) or 0)
                hand_count = int(getattr(my_state(obs), "handCount", 0) or 0)
                if deck_count > 7 and len(all_my_pokemon(obs)) >= 2 and hand_count <= 7:
                    score, reason = 21000, "use Dudunsparce draw"
                else:
                    score, reason = 2000, "save Dudunsparce draw"
            else:
                score, reason = 1000, "generic ability"
        elif opt.type == OptionType.ATTACK:
            score, reason = attack_score(obs, getattr(opt, "attackId", None)), "attack"
        elif opt.type == OptionType.END:
            score, reason = 0, "end"
        else:
            score, reason = 500, "main fallback"
    elif ctx == SelectContext.TO_HAND:
        score, reason = score_to_hand(obs, opt)
    elif ctx in {SelectContext.DISCARD, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD}:
        score, reason = score_discard(obs, opt)
    elif ctx in {
        SelectContext.ATTACH_TO,
        SelectContext.TO_FIELD,
        SelectContext.TO_BENCH,
        SelectContext.ATTACH_FROM,
        SelectContext.SWITCH,
        SelectContext.TO_ACTIVE,
        SelectContext.HEAL,
        SelectContext.DAMAGE,
        SelectContext.DAMAGE_COUNTER,
        SelectContext.DAMAGE_COUNTER_ANY,
        SelectContext.EFFECT_TARGET,
        SelectContext.REMOVE_DAMAGE_COUNTER,
        SelectContext.EVOLVES_FROM,
        SelectContext.EVOLVES_TO,
        SelectContext.TO_DECK,
        SelectContext.TO_DECK_BOTTOM,
        SelectContext.TO_DECK_ENERGY,
        SelectContext.TO_HAND_ENERGY,
        SelectContext.DISCARD_ENERGY,
        SelectContext.DISCARD_ENERGY_CARD,
    }:
        score, reason = score_target(obs, opt)
    elif ctx == SelectContext.ATTACK:
        score, reason = attack_score(obs, getattr(opt, "attackId", None)), "attack"
    elif opt.type == OptionType.CARD:
        score, reason = score_to_hand(obs, opt)
    elif opt.type == OptionType.ENERGY:
        score, reason = 1000, "energy"
    elif opt.type == OptionType.END:
        score, reason = 0, "end"
    else:
        score, reason = 100, "fallback"

    return apply_overrides(obs, opt, score, reason)


def choose_options(obs) -> list[int]:
    scored = []
    for i, opt in enumerate(obs.select.option):
        try:
            score, reason = score_option(obs, opt)
        except Exception as exc:
            score, reason = -999999, f"error {type(exc).__name__}: {exc}"
        scored.append((score, i, reason))

    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    if obs.select.context == SelectContext.SETUP_BENCH_POKEMON:
        if obs.select.minCount <= 0:
            return [scored[0][1]] if scored and scored[0][0] > 0 else []
        return [i for _, i, _ in scored[: obs.select.minCount]]

    selected = []
    for score, i, reason in scored:
        if len(selected) >= obs.select.maxCount:
            break
        if score < 0 and len(selected) >= obs.select.minCount:
            continue
        selected.append(i)

    if len(selected) < obs.select.minCount:
        for _, i, _ in scored:
            if i not in selected:
                selected.append(i)
                if len(selected) >= obs.select.minCount:
                    break

    return selected


def _legal_fallback(select) -> list[int]:
    n = len(select.option)
    if n <= 0 or select.maxCount <= 0:
        return []
    k = min(n, max(select.minCount, 1), select.maxCount)
    return list(range(k))


def _normalize_ordered_indices(raw, select) -> list[int]:
    n = len(select.option)
    if n <= 0 or select.maxCount <= 0:
        return []
    if not isinstance(raw, list):
        raw = []
    selected = []
    for item in raw:
        if isinstance(item, int) and 0 <= item < n and item not in selected:
            selected.append(item)
            if len(selected) >= select.maxCount:
                break
    if len(selected) < select.minCount:
        for idx in range(n):
            if idx not in selected:
                selected.append(idx)
                if len(selected) >= select.minCount:
                    break
    return selected[: select.maxCount]


def _policy_agent(obs):
    if obs.select is None:
        return read_deck_csv()
    if not obs.select.option:
        return []
    try:
        return choose_options(obs)
    except Exception:
        return _legal_fallback(obs.select)


def agent(obs_dict):
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        if isinstance(obs_dict, dict) and obs_dict.get("select") is None:
            return read_deck_csv()
        return [0]

    if obs.select is None:
        return read_deck_csv()

    try:
        return _normalize_ordered_indices(_policy_agent(obs), obs.select)
    except Exception:
        return _legal_fallback(obs.select)
