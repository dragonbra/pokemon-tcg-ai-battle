import os
import random
from collections import defaultdict

from cg.api import (
    AreaType, CardType, EnergyType, Log, LogType, Observation, SelectContext,
    OptionType, Card, Pokemon, State, all_card_data, to_observation_class,
)


my_deck = [
    119, 119, 119, 119, 120, 120, 120, 120, 121, 121, 
    121, 121, 140, 235, 235, 1071, 1079, 1079, 1080, 1086, 
    1086, 1086, 1086, 1097, 1097, 1120, 1120, 1120, 1120, 1121, 
    1121, 1121, 1121, 1152, 1152, 1152, 1156, 1182, 1182, 1182, 
    1198, 1198, 1198, 1198, 1210, 1210, 1227, 1227, 1227, 1227, 
    1256, 1256, 2, 2, 2, 2, 5, 5, 5, 5
]

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

# ── Card IDs (this deck) 
Dreepy        = 119
Drakloak      = 120
Dragapult_ex  = 121
Fezandipiti_ex = 140
Latias_ex     = 184
Budew         = 235
Meowth_ex     = 1071
Rare_Candy    = 1079
Unfair_Stamp  = 1080
Buddy_Buddy_Poffin = 1086
Night_Stretcher = 1097
Crushing_Hammer = 1120
Ultra_Ball    = 1121
Poke_Pad      = 1152
Lucky_Helmet  = 1156
Boss_Orders   = 1182
Crispin       = 1198
Brock_Scouting = 1210
Lillie_Determination = 1227
Team_Rocket_Watchtower = 1256
Basic_Fire_Energy    = 2
Basic_Psychic_Energy = 5

FIRE = int(EnergyType.FIRE)        # 2
PSYCHIC = int(EnergyType.PSYCHIC)  # 5

PHANTOM_DIVE = 154
ITCHY_POLLEN = 323

UNNECESSARY = -10000000


class AttackPlan:
    attack: int = 0
    counter: list = []


# ── Cross-turn state 
can_switch = False
can_attack = False
can_main_attack = False
bench_attacker = False
use_support = 0
pre_turn_log: list = []
current_turn_log: list = []
prize: list = []
card_counts = defaultdict(int)
serial_set = set()
plan_a = AttackPlan()
plan_b = AttackPlan()


# ── Energy helpers (type-aware) 
def count_type(pokemon, etype):
    return sum(1 for e in pokemon.energies if int(e) == etype)


def needs_RP(pokemon):
    """(needs_fire, needs_psychic) for a {R}{P} attacker."""
    return (count_type(pokemon, FIRE) < 1, count_type(pokemon, PSYCHIC) < 1)


# ── Deck / prize inference ───────────────────────────────────────────────────
def add_card_count(card, my_index):
    if card is None:
        return
    if isinstance(card, Pokemon) or card.playerIndex == my_index:
        if card.serial not in serial_set:
            card_counts[card.id] -= 1
            serial_set.add(card.serial)
    if isinstance(card, Pokemon):
        for c in card.energyCards:
            add_card_count(c, my_index)
        for c in card.tools:
            add_card_count(c, my_index)
        for c in card.preEvolution:
            add_card_count(c, my_index)


def set_card_counts(obs, my_index):
    card_counts.clear()
    serial_set.clear()
    for cid in my_deck:
        card_counts[cid] += 1
    state = obs.current
    ms = state.players[my_index]
    for card in ms.hand:
        add_card_count(card, my_index)
    for card in ms.discard:
        add_card_count(card, my_index)
    for card in ms.bench:
        add_card_count(card, my_index)
    for card in ms.active:
        add_card_count(card, my_index)
    for card in state.stadium:
        add_card_count(card, my_index)
    if state.looking is not None:
        for card in state.looking:
            add_card_count(card, my_index)
    add_card_count(obs.select.effect, my_index)


def get_card(obs, area, index, player_index):
    ps = obs.current.players[player_index]
    if area == AreaType.DECK:
        return obs.select.deck[index]
    if area == AreaType.HAND:
        return ps.hand[index]
    if area == AreaType.DISCARD:
        return ps.discard[index]
    if area == AreaType.ACTIVE:
        return ps.active[index]
    if area == AreaType.BENCH:
        return ps.bench[index]
    if area == AreaType.PRIZE:
        return ps.prize[index]
    if area == AreaType.STADIUM:
        return obs.current.stadium[index]
    if area == AreaType.LOOKING:
        return obs.current.looking[index]
    return None


# ── Target heuristics 
def no_damage_dex(cid):
    # Drednaw, Milotic ex, Sylveon, Crustle — immune to Dragapult ex's attack
    return cid in (158, 207, 330, 345)


def no_damage_counter(pokemon):
    if pokemon.id in (28, 199, 203, 207, 362, 1136):
        return True
    for card in pokemon.energyCards:
        if card.id in (11, 20):  # Mist Energy, Rock Fighting Energy
            return True
    return False


def prize_count(pokemon, is_attack_damage):
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    if is_attack_damage:
        for card in pokemon.energyCards:
            if card.id == 12:  # Legacy Energy
                count -= 1
        for card in pokemon.tools:
            if card.id == 1172 and "Lillie" in data.name:  # Lillie's Pearl
                count -= 1
    return max(0, count)


def pokemon_score(pokemon, is_attack_damage):
    data = card_table[pokemon.id]
    score = prize_count(pokemon, is_attack_damage) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130
    if pokemon.id in (173, 174, 190, 1071):  # passive bench sitters
        score -= 200
    score += pokemon.hp
    return score


# ── Phantom Dive counter-spread planner (1-turn lookahead) 
def main_option_proc(obs, damage):
    state = obs.current
    select = obs.select
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    global can_switch, can_attack, can_main_attack
    can_switch = can_attack = can_main_attack = False
    for o in select.option:
        if o.type == OptionType.RETREAT:
            can_switch = True
        elif o.type == OptionType.ATTACK:
            can_attack = True
            if o.attackId == PHANTOM_DIVE:
                can_main_attack = True

    plan_a.attack = plan_b.attack = -1
    if not can_main_attack and not (bench_attacker and can_switch):
        return

    cards = [op_state.active[0]] + list(op_state.bench)

    counter_indices = []
    ci = [0]
    remain_damage = 60
    while ci:
        index = ci[-1]
        hp = cards[index].hp
        if remain_damage >= hp:
            counter_indices.append(ci.copy())
            if index < len(cards) - 1:
                remain_damage -= hp
                ci.append(index + 1)
                continue
        if index == len(cards) - 1:
            ci.pop()
            if ci:
                remain_damage += cards[ci[-1]].hp
        if ci:
            ci[-1] += 1
    counter_indices.append([])

    remain_prize = len(my_state.prize)
    plan_score = 0
    for i, pokemon in enumerate(cards):
        base_prize = 0
        base_score = pokemon_score(pokemon, True)
        active_damage = 0 if no_damage_dex(pokemon.id) else damage
        if pokemon.hp <= active_damage:
            base_prize += prize_count(pokemon, True)
        else:
            base_score *= active_damage / pokemon.hp
        ci = []
        max_score = base_score
        if remain_prize <= base_prize:
            max_score = 50000
        else:
            for indices in counter_indices:
                if i in indices:
                    continue
                p = base_prize
                s = base_score
                for idx in indices:
                    p += prize_count(cards[idx], False)
                    s += pokemon_score(cards[idx], False)
                if remain_prize <= p:
                    s = 50000
                else:
                    if p >= 2:
                        s -= 1200 if remain_prize <= 4 else 0
                    elif p == 1:
                        s -= 300
                    else:
                        s += 1200
                if max_score < s:
                    max_score = s
                    ci = indices
        if plan_score < max_score:
            plan_score = max_score
            plan_a.attack = i
            plan_a.counter = ci
        if i == 0:
            plan_b.attack = plan_a.attack
            plan_b.counter = plan_a.counter


# ── Robust fallback 
def safe_default(obs):
    try:
        if obs.select is None:
            return my_deck
        n = len(obs.select.option)
        if n == 0:
            return []
        k = max(obs.select.minCount, min(obs.select.maxCount, n))
        return random.sample(range(n), k)
    except Exception:
        return my_deck


_DEBUG = bool(os.environ.get("AGENT_DEBUG"))


def agent(obs_dict):
    try:
        return _agent_impl(obs_dict)
    except Exception:
        if _DEBUG:
            raise
        try:
            return safe_default(to_observation_class(obs_dict))
        except Exception:
            return my_deck


def _agent_impl(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    global pre_turn_log, current_turn_log

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    if state.turn == 0:
        prize.clear()
        pre_turn_log.clear()
        current_turn_log.clear()
    else:
        for log in obs.logs:
            current_turn_log.append(log)
            if log.type == LogType.TURN_END:
                pre_turn_log = current_turn_log
                current_turn_log = []

    pre_ko = False
    no_item = False
    for log in pre_turn_log:
        if log.type == LogType.ATTACK and log.attackId == ITCHY_POLLEN:
            no_item = True
        elif log.type == LogType.MOVE_CARD:
            if (log.playerIndex == my_index
                    and log.fromArea in (AreaType.BENCH, AreaType.ACTIVE)
                    and log.toArea == AreaType.DISCARD):
                pre_ko = True

    if select.deck is not None:
        set_card_counts(obs, my_index)
        for card in select.deck:
            card_counts[card.id] -= 1
        prize.clear()
        for cid in card_counts:
            for _ in range(card_counts[cid]):
                prize.append(cid)

    set_card_counts(obs, my_index)
    for cid in prize:
        card_counts[cid] -= 1
    deck_counts = card_counts

    prize_diff = len(my_state.prize) - len(op_state.prize)

    global bench_attacker
    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    discard_counts = defaultdict(int)

    active_id = 0
    bench_attacker = False
    can_evolve_dreepy = False
    evolve_dreepy_count = 0
    can_evolve_drakloak = False
    damage = 200

    for card in my_state.active:
        if card is None:
            continue
        active_id = card.id
        field_counts[card.id] += 1
        if not card.appearThisTurn:
            if card.id == Dreepy:
                can_evolve_dreepy = True
                evolve_dreepy_count += 1
            elif card.id == Drakloak:
                can_evolve_drakloak = True
    for card in my_state.bench:
        field_counts[card.id] += 1
        if not card.appearThisTurn:
            if card.id == Dreepy:
                can_evolve_dreepy = True
                evolve_dreepy_count += 1
            elif card.id == Drakloak:
                can_evolve_drakloak = True
        if card.id == Dragapult_ex and count_type(card, FIRE) >= 1 and count_type(card, PSYCHIC) >= 1:
            bench_attacker = True

    main_pokemon_count = (field_counts[Dreepy] + field_counts[Drakloak]
                          + field_counts[Dragapult_ex])
    no_more_dex = (field_counts[Dragapult_ex] * 2 >= len(op_state.prize))

    stadium_id = 0
    for card in state.stadium:
        stadium_id = card.id

    for card in my_state.discard:
        discard_counts[card.id] += 1

    # ── attach scoring (TYPE-AWARE energy + tools) ───────────────────────────
    def attach_score(attach_id, pokemon, active):
        # Tools (Lucky Helmet)
        if card_table[attach_id].cardType == CardType.TOOL:
            if pokemon.tools:
                return -1  # already has a tool
            score = 30000
            # Lucky Helmet wants to be on the Pokémon taking hits (the attacker/active)
            if pokemon.id == Dragapult_ex:
                score += 2000
            if active:
                score += 1000
            return score

        # Energy
        if pokemon.id == Budew:
            return -1
        if pokemon.id in (Meowth_ex, Fezandipiti_ex, Latias_ex):
            # Utility ex: only fuel if stranded active with no escape and no real attacker
            if active and not can_switch and not my_state.asleep and not my_state.paralyzed:
                return 18000 if not bench_attacker else -1
            return -1

        if pokemon.id in (Dreepy, Drakloak, Dragapult_ex):
            # Primary attack is {R}{P}: want exactly one Fire + one Psychic.
            need_fire, need_psy = needs_RP(pokemon)
            attaching_fire = (attach_id == Basic_Fire_Energy)
            attaching_psy = (attach_id == Basic_Psychic_Energy)
            useful = (attaching_fire and need_fire) or (attaching_psy and need_psy)
            if not useful:
                return -1  # never waste an attachment on a redundant/irrelevant type
            if active and can_main_attack:
                return -1  # already able to Phantom Dive
            score = 20000
            # Prefer completing the pair on the most-evolved attacker.
            if pokemon.id == Dragapult_ex:
                score += 600
            elif pokemon.id == Drakloak:
                score += 200
            else:
                score += 50
            # Psychic is the broader-need type; small nudge when starting from empty.
            if attaching_psy:
                score += 30
            if active:
                score += 200
            if no_more_dex and pokemon.id in (Dreepy, Drakloak):
                score -= 500
            return score
        return -1

    # ── hand-card value ──────────────────────────────────────────────────────
    def hand_score(cid, ignore_count):
        score = 0
        if cid == Dreepy:
            score = 1000 if main_pokemon_count >= 3 else 18000
        elif cid == Drakloak:
            score = 20000 if can_evolve_dreepy else 3000
        elif cid == Dragapult_ex:
            if no_more_dex:
                score = UNNECESSARY
            elif can_evolve_dreepy and hand_counts[Rare_Candy] >= 1 and not no_item:
                score = 40000  # Rare Candy line: Dreepy -> Dragapult ex
            elif can_evolve_drakloak:
                score = 30000 if field_counts[cid] == 0 else 10000 if field_counts[cid] == 1 else 50
            else:
                score = 2000 if field_counts[cid] < 2 else 50
        elif cid == Budew:
            if field_counts[Budew] >= 1:
                score = UNNECESSARY
            elif (field_counts[Drakloak] + field_counts[Dragapult_ex] >= 1 and state.turn >= 2):
                score = 30000
            else:
                score = 5000
        elif cid == Fezandipiti_ex:
            if pre_ko:
                score = 50000
            elif prize_diff <= -2:
                score = 5
            elif len(op_state.prize) == 1:
                score = UNNECESSARY
            else:
                score = 3000
        elif cid == Latias_ex:
            # Skyliner (free retreat for basics) is passive value; bench early.
            if field_counts[Latias_ex] >= 1:
                score = UNNECESSARY
            elif field_counts[Drakloak] + field_counts[Dragapult_ex] == 0:
                score = 12000
            else:
                score = 4000
        elif cid == Meowth_ex:
            if support_count > hand_counts[Boss_Orders] or stadium_id == Team_Rocket_Watchtower:
                score = 5  # ability blocked by our own Watchtower, or supporters already in hand
            elif state.supporterPlayed:
                score = 40
            else:
                score = 35000
        elif cid == Rare_Candy:
            if no_more_dex:
                score = UNNECESSARY
            elif can_evolve_dreepy and hand_counts[Dragapult_ex] >= 1 and not no_item:
                score = 40000
            else:
                score = 1500
        elif cid == Lillie_Determination:
            if my_state.deckCount <= 6:
                score = 100  # low-deck guard: don't shuffle a tiny deck
            elif not ignore_count or support_count == 0:
                score = 45000
        elif cid == Crispin:
            if not ignore_count or support_count == 0:
                if deck_counts[Basic_Fire_Energy] == 0 and deck_counts[Basic_Psychic_Energy] == 0:
                    score = 10
                elif not can_main_attack and not bench_attacker and field_counts[Dragapult_ex] >= 1:
                    score = 55000
                else:
                    score = 28000
        elif cid == Brock_Scouting:
            if not ignore_count or support_count == 0:
                if main_pokemon_count <= 2:
                    score = 30000
                else:
                    score = 12000
        elif cid == Boss_Orders:
            if not ignore_count or support_count == 0:
                if plan_a.attack > 0:
                    score = 60000
        elif cid == Unfair_Stamp:
            score = 80000 if pre_ko else (UNNECESSARY if len(op_state.prize) == 1 else 80)
        elif cid == Buddy_Buddy_Poffin:
            count = deck_counts[Dreepy]
            if state.turn <= 2 and field_counts[Budew] == 0 and deck_counts[Budew] >= 1:
                count += 1
            if deck_counts[Dreepy] == 0:
                score = UNNECESSARY
            else:
                score = 35000 if count >= 2 else 20000
        elif cid == Night_Stretcher:
            for did in discard_counts:
                if discard_counts[did] >= 1:
                    ct = card_table[did].cardType
                    if ct in (CardType.POKEMON, CardType.BASIC_ENERGY):
                        score = max(score, hand_score(did, ignore_count))
        elif cid == Crushing_Hammer:
            score = 20
        elif cid == Ultra_Ball:
            if main_pokemon_count <= 2 or field_counts[Dreepy] >= 1:
                score = 70
            else:
                score = 5
        elif cid == Poke_Pad:
            score = max(hand_score(Dreepy, ignore_count), hand_score(Drakloak, ignore_count))
        elif cid == Lucky_Helmet:
            score = 15
        elif cid in (Basic_Fire_Energy, Basic_Psychic_Energy):
            if can_main_attack and (len(op_state.prize) <= 2
                    or (bench_attacker and len(op_state.prize) <= 4)):
                score = UNNECESSARY
            else:
                max_s = -10000
                for pk in my_state.active:
                    if pk is not None:
                        max_s = max(max_s, attach_score(cid, pk, True))
                for pk in my_state.bench:
                    max_s = max(max_s, attach_score(cid, pk, False))
                score = max_s - 5000
                if can_main_attack or bench_attacker:
                    score //= 10
        elif cid == Team_Rocket_Watchtower:
            if stadium_id != 0 and stadium_id != Team_Rocket_Watchtower:
                score = 4000
            elif stadium_id == 0:
                score = 2000

        if not ignore_count and hand_counts[cid] > 0:
            if cid == Drakloak and hand_counts[cid] < evolve_dreepy_count:
                score -= 10
            elif cid == Dreepy:
                score -= 100
            else:
                score -= 100000
        return score

    # ── MAIN: attack plan + best supporter ───────────────────────────────────
    support_count = 0
    global use_support
    if context == SelectContext.MAIN:
        main_option_proc(obs, damage)
        use_support = 0
        if not state.supporterPlayed:
            support_score = 0
            for o in select.option:
                if o.type == OptionType.PLAY:
                    card = get_card(obs, AreaType.HAND, o.index, my_index)
                    if card_table[card.id].cardType == CardType.SUPPORTER:
                        s = hand_score(card.id, True)
                        if support_score < s:
                            support_score = s
                            use_support = card.id

    hand_scores = []
    negative_hand_count = 0
    for card in my_state.hand:
        s = hand_score(card.id, False)
        hand_scores.append(s)
        if s < 0:
            negative_hand_count += 1
        hand_counts[card.id] += 1
        if card_table[card.id].cardType == CardType.SUPPORTER and card.id != Boss_Orders:
            support_count += 1

    no_draw = (my_state.deckCount <= 8)
    do_switch = (not can_main_attack
                 and (bench_attacker
                      or (active_id != Budew and field_counts[Budew] >= 1 and state.turn >= 2)))
    effect_card_id = 0 if select.effect is None else select.effect.id
    context_card_id = 0 if select.contextCard is None else select.contextCard.id

    scores = []
    for o in select.option:
        score = 0

        if o.type == OptionType.NUMBER:
            score = o.number  # damage-counter / draw counts: take the max offered

        elif o.type == OptionType.YES:
            score = -1 if context == SelectContext.IS_FIRST else 1

        elif o.type == OptionType.NO:
            score = 1 if context == SelectContext.IS_FIRST else -1

        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is not None:
                energy_count = len(card.energies) if isinstance(card, Pokemon) else 0
                hp = card.hp if isinstance(card, Pokemon) else 0

                if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE,
                               SelectContext.SETUP_ACTIVE_POKEMON):
                    if o.playerIndex == my_index:
                        if card.id == Dreepy:
                            score += 10000
                        elif card.id == Drakloak:
                            score += 20000 if energy_count >= 1 else -10000
                        elif card.id == Dragapult_ex:
                            score += 50000
                        elif card.id == Budew:
                            score += (100000 if context != SelectContext.SWITCH
                                      else 30000 if not bench_attacker else -5000)
                        elif card.id in (Fezandipiti_ex, Meowth_ex, Latias_ex):
                            score -= 2000
                    else:
                        if plan_a.attack == o.index + 1:
                            score += 100000
                    score += energy_count * 1000 + hp

                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    if my_index == state.firstPlayer or card.id != Dreepy:
                        score = -1

                elif context in (SelectContext.TO_BENCH, SelectContext.TO_HAND):
                    score = hand_score(card.id, False)
                    hand_counts[card.id] += 1
                    if effect_card_id == Crispin:
                        score = 100000 - hand_score(card.id, True)

                elif context == SelectContext.DISCARD:
                    hand_counts[card.id] -= 1
                    if card_table[card.id].cardType == CardType.SUPPORTER:
                        support_count -= 1
                    score = -hand_score(card.id, False)

                elif context in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
                    if hp > 0:
                        score = 100000 - 10 * hp + pokemon_score(card, False)
                        if context == SelectContext.DAMAGE_COUNTER:
                            if 210 <= hp <= 230:
                                score += 20000 + hp * 20
                                if o.area == AreaType.ACTIVE:
                                    score += 10000
                            elif 40 <= hp <= 90:
                                score += 10000 + hp * 20
                            elif hp <= 30:
                                score += -10000 + hp * 20
                        else:
                            index = o.index + 1
                            if index in plan_b.counter:
                                score += 100000
                            else:
                                remain = select.remainDamageCounter * 10
                                if 210 <= hp <= 200 + remain:
                                    score += 30000
                                elif 20 <= hp <= 60 + remain:
                                    score += 10000
                                elif hp == 10:
                                    score -= 100000
                            if no_damage_counter(card):
                                score = -1

                elif context in (SelectContext.REMOVE_DAMAGE_COUNTER, SelectContext.HEAL):
                    if o.playerIndex == my_index and isinstance(card, Pokemon):
                        score = (card.maxHp - card.hp) + (300 if card.id == Dragapult_ex else 0)
                    else:
                        score = -1

                elif context == SelectContext.ATTACH_FROM:
                    score = attach_score(context_card_id, card, o.area == AreaType.ACTIVE)
                    if card.id == Dragapult_ex:
                        score += 200

                else:
                    score = hand_score(card.id, False) if card.id in card_table else 0

        elif o.type in (OptionType.ENERGY_CARD, OptionType.ENERGY):
            # which energy to discard (opponent: Crushing Hammer target; ours: retreat)
            if o.playerIndex != my_index:
                score = 20 if o.area == AreaType.BENCH else 10
                card = get_card(obs, o.area, o.index, o.playerIndex)
                if card is not None and card_table[card.id].cardType == CardType.SPECIAL_ENERGY:
                    score += 1
            else:
                score = 5

        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            card_score = hand_scores[o.index]
            cid = card.id
            if cid == Dreepy:
                score = 51000
            elif cid == Fezandipiti_ex:
                score = 53000 if card_score > 0 else -1
            elif cid == Latias_ex:
                score = 51000 if (active_id not in (Drakloak, Dragapult_ex) and card_score > 0) else -1
            elif cid == Budew:
                score = 52000 if (field_counts[Budew] == 0 and field_counts[Dragapult_ex] >= 1) else -1
            elif cid == Meowth_ex:
                if state.supporterPlayed or stadium_id == Team_Rocket_Watchtower:
                    score = -1
                elif support_count == 0 or (support_count == hand_counts[Boss_Orders] and plan_a.attack > 0):
                    score = 50000
                else:
                    score = -1
            elif cid == Rare_Candy:
                score = 75000 if (not no_more_dex and can_evolve_dreepy
                                  and hand_counts[Dragapult_ex] >= 1 and not no_item) else -1
            elif cid == Boss_Orders:
                score = 35000 if cid == use_support else -1
            elif cid in (Lillie_Determination, Crispin, Brock_Scouting):
                score = 14000 if cid == use_support else -1
            elif cid == Unfair_Stamp:
                score = 15000 if pre_ko else -1
            elif cid == Crushing_Hammer:
                score = 40000
            elif cid == Night_Stretcher:
                score = 42000 if card_score >= 18000 else -1
            elif cid == Buddy_Buddy_Poffin:
                score = 46000 if deck_counts[Dreepy] > 0 else -1
            elif cid == Ultra_Ball:
                score = 44000 if negative_hand_count >= 2 else -1
            elif cid == Poke_Pad:
                score = 45000 if (deck_counts[Dreepy] + deck_counts[Drakloak] > 0) else -1
            elif cid == Lucky_Helmet:
                score = 10000
            elif cid == Team_Rocket_Watchtower:
                score = 80000 if (stadium_id > 0 or state.turn == 1) else -1
            elif no_draw:
                score = -1

        elif o.type == OptionType.ATTACH:
            card = get_card(obs, o.area, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = attach_score(card.id, pokemon, o.inPlayArea == AreaType.ACTIVE)

        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score += len(pokemon.energies)
            if pokemon.id == Dreepy:
                score += 30000
            elif pokemon.id == Drakloak:
                if field_counts[Dragapult_ex] >= 2 or (
                        field_counts[Dragapult_ex] == 1 and len(op_state.prize) <= 2):
                    score = -1
                else:
                    score += 70000

        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if no_draw:
                score = -1
            else:
                score = 40000  # Recon Directive / Flip the Script / Last-Ditch Catch

        elif o.type == OptionType.RETREAT:
            score = 10000 if do_switch else -1

        elif o.type == OptionType.ATTACK:
            score = o.attackId
            if o.attackId == PHANTOM_DIVE:
                score += 1000

        scores.append(score)

    output = []
    if scores:
        ordered = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        for i in range(select.maxCount):
            if (ordered[i][1] >= 0
                    or select.minCount > i
                    or context not in (SelectContext.TO_BENCH, SelectContext.SETUP_BENCH_POKEMON)):
                output.append(ordered[i][0])
        if len(output) < select.minCount:
            for idx, _ in ordered:
                if idx not in output:
                    output.append(idx)
                if len(output) >= select.minCount:
                    break
    return output
