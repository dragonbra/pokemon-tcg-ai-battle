"""Opponent: pixiux_crustle - Advanced Crustle wall with detailed scoring"""
from collections import Counter
from pathlib import Path

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


DWEBBLE = 344
CRUSTLE = 345
JUMBO_ICE_CREAM = 1147
HERO_CAPE = 1159
BATTLE_CAGE = 1264
COOK = 1212
CHEREN = 1224
GROW_GRASS_ENERGY = 18
MIST_ENERGY = 11
BUDDY_POFFIN = 1086
SPIKY_ENERGY = 14
BASIC_GRASS_ENERGY = 1

ASCENSION = 478
SUPERB_SCISSORS = 479

PACKAGE_ROOT = Path(__file__).resolve().parent


def read_deck_csv() -> list[int]:
    return [
        int(line.strip())
        for line in (PACKAGE_ROOT / "deck.csv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


DECK = read_deck_csv()
my_deck = DECK

CARD_TABLE = {c.cardId: c for c in all_card_data()}


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    state = obs.current
    if state is None:
        return None
    player = state.players[player_index]
    if area == AreaType.DECK and obs.select.deck is not None:
        return obs.select.deck[index]
    if area == AreaType.HAND:
        return player.hand[index]
    if area == AreaType.DISCARD:
        return player.discard[index]
    if area == AreaType.ACTIVE:
        return player.active[index]
    if area == AreaType.BENCH:
        return player.bench[index]
    if area == AreaType.PRIZE:
        return player.prize[index]
    if area == AreaType.STADIUM:
        return state.stadium[index]
    if area == AreaType.LOOKING and state.looking is not None:
        return state.looking[index]
    return None


def all_my_pokemon(obs: Observation, my_index: int) -> list[tuple[Pokemon, bool, int]]:
    player = obs.current.players[my_index]
    result = []
    for idx, pokemon in enumerate(player.active):
        if pokemon is not None:
            result.append((pokemon, True, idx))
    for idx, pokemon in enumerate(player.bench):
        result.append((pokemon, False, idx))
    return result


def damage_on(pokemon: Pokemon | None) -> int:
    if pokemon is None:
        return 0
    return max(0, pokemon.maxHp - pokemon.hp)


def has_grass_energy(pokemon: Pokemon) -> bool:
    if EnergyType.GRASS in pokemon.energies:
        return True
    return any(card.id in (BASIC_GRASS_ENERGY, GROW_GRASS_ENERGY) for card in pokemon.energyCards)


def has_tool(pokemon: Pokemon, card_id: int) -> bool:
    return any(card.id == card_id for card in pokemon.tools)


def crustle_ready(pokemon: Pokemon) -> bool:
    return pokemon.id == CRUSTLE and len(pokemon.energies) >= 3 and has_grass_energy(pokemon)


def prize_value(pokemon: Pokemon) -> int:
    data = CARD_TABLE[pokemon.id]
    if data.megaEx:
        return 3
    if data.ex:
        return 2
    return 1


def target_score(pokemon: Pokemon, active: bool, remaining_prize: int) -> int:
    score = prize_value(pokemon) * 1000 + len(pokemon.energies) * 120 + pokemon.maxHp - pokemon.hp
    data = CARD_TABLE[pokemon.id]
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 120
    if active:
        score += 500
    damage = 120
    if data.weakness == EnergyType.GRASS:
        damage *= 2
    elif data.resistance == EnergyType.GRASS:
        damage -= 30
    if pokemon.hp <= damage:
        score += 2500 + prize_value(pokemon) * 800
        if remaining_prize <= prize_value(pokemon):
            score += 50000
    return score


def board_counts(obs: Observation, my_index: int) -> tuple[Counter, Counter, Counter]:
    state = obs.current
    player = state.players[my_index]
    field = Counter()
    hand = Counter()
    discard = Counter()
    for pokemon, _, _ in all_my_pokemon(obs, my_index):
        field[pokemon.id] += 1
        for card in pokemon.preEvolution:
            field[card.id] += 1
        for card in pokemon.energyCards:
            field[card.id] += 1
        for card in pokemon.tools:
            field[card.id] += 1
    for card in player.hand or []:
        field[card.id] += 0
        hand[card.id] += 1
    for card in player.discard:
        discard[card.id] += 1
    return field, hand, discard


def best_attach_target_score(card_id: int, pokemon: Pokemon, active: bool) -> int:
    if pokemon.id not in (DWEBBLE, CRUSTLE):
        return -10000

    if card_id == HERO_CAPE:
        if has_tool(pokemon, HERO_CAPE):
            return -10000
        score = 45000
        if pokemon.id == CRUSTLE:
            score += 5000
        if active:
            score += 6000
        score += damage_on(pokemon)
        return score

    energy_count = len(pokemon.energies)
    if energy_count >= 3:
        return -5000

    is_grass = card_id in (BASIC_GRASS_ENERGY, GROW_GRASS_ENERGY)
    score = 30000
    if active:
        score += 5000
    if pokemon.id == CRUSTLE:
        score += 5000
    if energy_count == 0:
        score += 2000
    elif energy_count == 1:
        score += 3500
    elif energy_count == 2:
        score += 6000

    if not has_grass_energy(pokemon):
        score += 9000 if is_grass else -4000
    elif is_grass and card_id == BASIC_GRASS_ENERGY:
        score += 300

    if card_id == SPIKY_ENERGY and active:
        score += 1200
    if card_id == MIST_ENERGY and active:
        score += 900
    return score


def wanted_card_score(card_id: int, obs: Observation, my_index: int, ignore_hand: bool = False) -> int:
    state = obs.current
    player = state.players[my_index]
    field, hand, discard = board_counts(obs, my_index)
    active = player.active[0] if player.active else None
    active_damage = damage_on(active)
    dwebble_total = field[DWEBBLE] + field[CRUSTLE]
    ready_count = sum(1 for pokemon, _, _ in all_my_pokemon(obs, my_index) if crustle_ready(pokemon))
    can_evolve = any(p.id == DWEBBLE and not p.appearThisTurn for p, _, _ in all_my_pokemon(obs, my_index))
    stadium_id = state.stadium[0].id if state.stadium else 0

    score = 0
    if card_id == DWEBBLE:
        score = 22000 if dwebble_total < 3 else 2000
        if player.benchMax <= len(player.bench) and active is not None:
            score = -10000
    elif card_id == CRUSTLE:
        score = 30000 if can_evolve else 3000
        if field[CRUSTLE] >= 2 and ready_count >= 1:
            score -= 15000
    elif card_id == BUDDY_POFFIN:
        score = 26000 if dwebble_total < 3 and len(player.bench) < player.benchMax else -10000
    elif card_id == HERO_CAPE:
        has_any_cape = any(has_tool(p, HERO_CAPE) for p, _, _ in all_my_pokemon(obs, my_index))
        score = -10000 if has_any_cape else 24000
    elif card_id == BATTLE_CAGE:
        score = 18000 if stadium_id != BATTLE_CAGE else -10000
    elif card_id == JUMBO_ICE_CREAM:
        if active is not None and active.id == CRUSTLE and len(active.energies) >= 3 and active_damage >= 50:
            score = 26000 + active_damage
        else:
            score = 500
    elif card_id == COOK:
        if not state.supporterPlayed and active_damage >= 50:
            score = 21000 + active_damage
        else:
            score = 800
    elif card_id == CHEREN:
        score = 12000 if not state.supporterPlayed else -5000
    elif card_id in (BASIC_GRASS_ENERGY, GROW_GRASS_ENERGY, MIST_ENERGY, SPIKY_ENERGY):
        best = -10000
        for pokemon, is_active, _ in all_my_pokemon(obs, my_index):
            best = max(best, best_attach_target_score(card_id, pokemon, is_active))
        score = best - 3000

    if not ignore_hand and hand[card_id] > 0:
        score -= 12000
    if discard[card_id] >= 3 and card_id in (JUMBO_ICE_CREAM, COOK, BATTLE_CAGE):
        score -= 2000
    return score


def discard_score(card_id: int, obs: Observation, my_index: int) -> int:
    field, hand, _ = board_counts(obs, my_index)
    if card_id == BASIC_GRASS_ENERGY and hand[card_id] >= 3:
        return 100
    if card_id == BATTLE_CAGE and hand[card_id] >= 2:
        return 90
    if card_id == JUMBO_ICE_CREAM and hand[card_id] >= 2:
        return 80
    if card_id == COOK and hand[card_id] >= 2:
        return 60
    if card_id == CHEREN and hand[card_id] >= 2:
        return 50
    if card_id == DWEBBLE and field[DWEBBLE] + field[CRUSTLE] >= 3:
        return 40
    return -wanted_card_score(card_id, obs, my_index, ignore_hand=True)


def score_option(obs: Observation, option) -> int:
    state = obs.current
    select = obs.select
    my_index = state.yourIndex
    player = state.players[my_index]
    opponent = state.players[1 - my_index]
    context = select.context
    active = player.active[0] if player.active else None

    if option.type == OptionType.NUMBER:
        return option.number or 0

    if option.type == OptionType.YES:
        if context == SelectContext.IS_FIRST:
            return -10
        return 100
    if option.type == OptionType.NO:
        if context == SelectContext.IS_FIRST:
            return 100
        if context in (SelectContext.MULLIGAN,):
            return 20
        return 0

    if option.type == OptionType.CARD:
        card = get_card(obs, option.area, option.index, option.playerIndex)
        if card is None:
            return -10000
        card_id = card.id

        if context == SelectContext.SETUP_ACTIVE_POKEMON:
            return 10000 if card_id == DWEBBLE else 0
        if context in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH, SelectContext.TO_FIELD):
            if card_id == DWEBBLE:
                field, _, _ = board_counts(obs, my_index)
                return 9000 - field[DWEBBLE] * 300
            return wanted_card_score(card_id, obs, my_index)
        if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
            if option.playerIndex != my_index or not isinstance(card, Pokemon):
                return target_score(card, option.area == AreaType.ACTIVE, len(player.prize))
            score = 1000 + len(card.energies) * 500 + card.hp
            if card.id == CRUSTLE:
                score += 6000
                if crustle_ready(card):
                    score += 6000
            elif card.id == DWEBBLE:
                score += 1500
            return score
        if context == SelectContext.TO_HAND:
            return wanted_card_score(card_id, obs, my_index)
        if context == SelectContext.DISCARD:
            return discard_score(card_id, obs, my_index)
        if context in (SelectContext.HEAL, SelectContext.REMOVE_DAMAGE_COUNTER):
            if isinstance(card, Pokemon) and option.playerIndex == my_index:
                return damage_on(card) + (5000 if card.id == CRUSTLE else 0)
        if context in (SelectContext.EVOLVES_FROM, SelectContext.ATTACH_FROM):
            if isinstance(card, Pokemon):
                return best_attach_target_score(select.effect.id if select.effect else BASIC_GRASS_ENERGY, card, option.area == AreaType.ACTIVE)
        if context == SelectContext.EVOLVES_TO:
            return 10000 if card_id == CRUSTLE else wanted_card_score(card_id, obs, my_index)
        if context in (SelectContext.DAMAGE, SelectContext.EFFECT_TARGET, SelectContext.ATTACK):
            if isinstance(card, Pokemon):
                return target_score(card, option.area == AreaType.ACTIVE, len(player.prize))
        return wanted_card_score(card_id, obs, my_index)

    if option.type == OptionType.PLAY:
        card = get_card(obs, AreaType.HAND, option.index, my_index)
        if card is None:
            return -10000
        card_id = card.id
        data = CARD_TABLE[card_id]
        if data.cardType == CardType.POKEMON:
            return 25000 if card_id == DWEBBLE and len(player.bench) < player.benchMax else -5000
        if card_id == BUDDY_POFFIN:
            return wanted_card_score(card_id, obs, my_index, ignore_hand=True)
        if card_id == BATTLE_CAGE:
            stadium_id = state.stadium[0].id if state.stadium else 0
            return 23000 if stadium_id != BATTLE_CAGE else -10000
        if card_id == JUMBO_ICE_CREAM:
            if active is not None and active.id == CRUSTLE and len(active.energies) >= 3 and damage_on(active) >= 50:
                return 36000 + damage_on(active)
            return -2000
        if card_id == COOK:
            if not state.supporterPlayed and active is not None and damage_on(active) >= 50:
                return 31000 + damage_on(active)
            return -2000
        if card_id == CHEREN:
            return 12000 if not state.supporterPlayed else -5000
        return wanted_card_score(card_id, obs, my_index, ignore_hand=True)

    if option.type == OptionType.ATTACH:
        card = get_card(obs, option.area, option.index, my_index)
        pokemon = get_card(obs, option.inPlayArea, option.inPlayIndex, my_index)
        if card is None or pokemon is None or not isinstance(pokemon, Pokemon):
            return -10000
        return best_attach_target_score(card.id, pokemon, option.inPlayArea == AreaType.ACTIVE)

    if option.type == OptionType.EVOLVE:
        pokemon = get_card(obs, option.inPlayArea, option.inPlayIndex, my_index)
        if pokemon is None or not isinstance(pokemon, Pokemon):
            return -10000
        score = 52000 + len(pokemon.energies) * 1000
        if option.inPlayArea == AreaType.ACTIVE:
            score += 5000
        return score

    if option.type == OptionType.ABILITY:
        return 20000

    if option.type == OptionType.RETREAT:
        if active is None:
            return -10000
        best_bench = 0
        for pokemon in player.bench:
            score = 1000
            if pokemon.id == CRUSTLE:
                score += 5000
                if crustle_ready(pokemon):
                    score += 8000
            best_bench = max(best_bench, score)
        if active.id != CRUSTLE and best_bench > 5000:
            return 15000 + best_bench
        return -10000

    if option.type == OptionType.ATTACK:
        if option.attackId == SUPERB_SCISSORS:
            score = 18000
            if opponent.active and opponent.active[0] is not None:
                score += target_score(opponent.active[0], True, len(player.prize))
            return score
        if option.attackId == ASCENSION:
            field, _, _ = board_counts(obs, my_index)
            return 28000 if field[CRUSTLE] == 0 else 15000
        return 1000

    if option.type == OptionType.END:
        return -100000

    return 0


def agent(obs_dict: dict) -> list[int]:
    if obs_dict.get("select") is None:
        return list(DECK)
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return list(DECK)

        scores = [score_option(obs, option) for option in obs.select.option]
        order = [idx for idx, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]

        if obs.select.minCount == 0:
            selected = [idx for idx in order if scores[idx] > 0][: obs.select.maxCount]
            return selected
        return order[: obs.select.maxCount]
    except Exception:
        return [0]
