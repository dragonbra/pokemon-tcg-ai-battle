"""Opponent: kiyotah_abomasnow - Mega Abomasnow ex with Hammer-lanche strategy"""
from collections import defaultdict
from pathlib import Path

from cg.api import AreaType, CardType, Observation, SelectContext, OptionType, Card, Pokemon, all_card_data, to_observation_class

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

Kyogre = 721
Snover = 722
Mega_Abomasnow_ex = 723
Ultra_Ball = 1121
Precious_Trolley = 1126
Carmine = 1192
Lillie_Determination = 1227
Surfing_Beach = 1262
Basic_Water_Energy = 3

PACKAGE_ROOT = Path(__file__).resolve().parent
DECK = [int(line) for line in (PACKAGE_ROOT / "deck.csv").read_text(encoding="utf-8").splitlines()]


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


def agent(obs_dict: dict) -> list[int]:
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return list(DECK)

        state = obs.current
        select = obs.select
        context = select.context
        my_index = state.yourIndex
        my_state = state.players[my_index]

        field_counts = defaultdict(int)
        hand_counts = defaultdict(int)
        discard_counts = defaultdict(int)

        bench_attacker_index0 = -1
        bench_attacker_index1 = -1
        for i, card in enumerate(my_state.bench):
            field_counts[card.id] += 1
            if card.id == Mega_Abomasnow_ex and len(card.energies) >= 2:
                bench_attacker_index0 = i
            elif card.id == Kyogre and len(card.energies) >= 1:
                bench_attacker_index1 = i

        for card in my_state.hand:
            hand_counts[card.id] += 1

        for card in my_state.discard:
            discard_counts[card.id] += 1

        op_active_hp = 0
        for card in state.players[1 - my_index].active:
            if card is None:
                continue
            op_active_hp = card.hp

        prefer_ky = op_active_hp <= 20 * discard_counts[Basic_Water_Energy]
        switch_index = -1
        for card in my_state.active:
            if card is None:
                continue
            field_counts[card.id] += 1
            if card.id == Mega_Abomasnow_ex and len(card.energies) >= 2:
                if prefer_ky and bench_attacker_index1 >= 0:
                    switch_index = bench_attacker_index1
            elif card.id == Kyogre and len(card.energies) >= 1:
                if not prefer_ky and bench_attacker_index0 >= 0:
                    switch_index = bench_attacker_index0
            elif bench_attacker_index0 >= 0:
                switch_index = bench_attacker_index0

        scores = []
        for o in select.option:
            score = 0
            if o.type == OptionType.NUMBER:
                score = o.number
            elif o.type == OptionType.YES:
                score = 1
            elif o.type == OptionType.CARD:
                card = get_card(obs, o.area, o.index, o.playerIndex)
                if card is not None:
                    energy_count = 0
                    if isinstance(card, Pokemon):
                        energy_count = len(card.energies)
                    if (context == SelectContext.SWITCH
                        or context == SelectContext.TO_ACTIVE
                        or context == SelectContext.SETUP_ACTIVE_POKEMON):
                        score += energy_count * 2
                        if o.index == switch_index:
                            score += 100
                        if card.id == Mega_Abomasnow_ex:
                            score += 20
                        elif card.id == Kyogre:
                            score += 10
                    elif context == SelectContext.TO_BENCH or context == SelectContext.TO_HAND:
                        if card.id == Snover:
                            if field_counts[card.id] >= 1:
                                score += 5
                            elif field_counts[Mega_Abomasnow_ex] >= 1:
                                score += 15
                            else:
                                score += 30
                        elif card.id == Mega_Abomasnow_ex:
                            if field_counts[Snover] >= 1 and field_counts[card.id] + hand_counts[card.id] == 0:
                                score += 100
                            else:
                                score += 10
                        elif card.id == Kyogre:
                            if field_counts[card.id] >= 1:
                                score += 1
                            else:
                                score += 20
                    elif context == SelectContext.DISCARD:
                        if card.id == Basic_Water_Energy:
                            score += 100
                        elif card.id == Mega_Abomasnow_ex:
                            score += 10
                        elif card.id == Carmine:
                            if hand_counts[Lillie_Determination] >= 1:
                                score += 30
                        elif card.id == Lillie_Determination:
                            score -= 20

                        if hand_counts[card.id] >= 2:
                            score += 500
                        hand_counts[card.id] -= 1
            elif o.type == OptionType.PLAY:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                score = 10000
                if card.id == Ultra_Ball:
                    if hand_counts[Basic_Water_Energy] >= 3 or (my_state.handCount >= 4 and (field_counts[Mega_Abomasnow_ex] + hand_counts[Mega_Abomasnow_ex] == 0 or field_counts[Mega_Abomasnow_ex] + field_counts[Snover] == 0 or field_counts[Kyogre] == 0)):
                        score = 4000
                    else:
                        score = -1
                elif card.id == Carmine:
                    if field_counts[Snover] >= 1 and hand_counts[Mega_Abomasnow_ex] >= 1:
                        score = -1
                    else:
                        score = 3000
                elif card.id == Lillie_Determination:
                    if field_counts[Snover] >= 1 and field_counts[Mega_Abomasnow_ex] == 0 and hand_counts[Mega_Abomasnow_ex] >= 1:
                        score = -1
                    else:
                        score = 3100
            elif o.type == OptionType.ATTACH:
                pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
                score = 5000
                energy_count = len(pokemon.energies)
                if energy_count == 0:
                    if o.inPlayArea == AreaType.BENCH:
                        score += 1
                if pokemon.id == Snover:
                    score += 1
                    if energy_count == 1:
                        score -= 100
                    elif energy_count >= 2:
                        score -= 400
                    if bench_attacker_index0 >= 0:
                        score -= 300
                elif pokemon.id == Mega_Abomasnow_ex:
                    score += 10
                    if energy_count == 1:
                        score += 30
                    elif energy_count >= 2:
                        score -= 300
                    if bench_attacker_index0 >= 0:
                        score -= 200
                elif pokemon.id == Kyogre:
                    score += 5
                    if len(pokemon.energies) >= 1:
                        score -= 200
                    if bench_attacker_index1 >= 0:
                        score -= 200
                if o.inPlayArea == AreaType.ACTIVE:
                    if bench_attacker_index0 >= 0 and bench_attacker_index1 >= 0 and energy_count <= 2:
                        score += 200
            elif o.type == OptionType.EVOLVE:
                pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
                score = 10000 + len(pokemon.energies)
            elif o.type == OptionType.ABILITY:
                card = get_card(obs, o.area, o.index, my_index)
                if card.id == Surfing_Beach and switch_index >= 0:
                    score = 2000
                else:
                    score = -1
            elif o.type == OptionType.RETREAT:
                if switch_index >= 0:
                    score = 1500
                else:
                    score = -1
            elif o.type == OptionType.ATTACK:
                score = 1000
                if o.attackId == 1042:  # Riptide
                    score += discard_counts[Basic_Water_Energy] * 20 - 90
                elif o.attackId == 1046:  # Hammer-lanche
                    if op_active_hp <= 200:
                        score -= 100
                    else:
                        score += 100

            scores.append(score)

        desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
        return desc_indices[:select.maxCount]
    except Exception:
        if obs_dict.get("select") is None:
            return list(DECK)
        return [0]
