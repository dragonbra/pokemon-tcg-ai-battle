"""Opponent: yakitori_raging_bolt - Raging Bolt ex deck with multi-energy acceleration"""
from pathlib import Path
from collections import defaultdict

from cg.api import (
    AreaType,
    CardType,
    Observation,
    SelectContext,
    SelectType,
    OptionType,
    Card,
    Pokemon,
    to_observation_class,
)

CARD_NAME = {
    1: "Basic {G} Energy",
    4: "Basic {L} Energy",
    5: "Basic {P} Energy",
    6: "Basic {F} Energy",
    63: "Raging Bolt ex",
    96: "Teal Mask Ogerpon ex",
    140: "Fezandipiti ex",
    176: "Terapagos ex",
    184: "Latias ex",
    756: "Mega Kangaskhan ex",
    1094: "Bug Catching Set",
    1098: "Glass Trumpet",
    1102: "Dusk Ball",
    1116: "Energy Switch",
    1118: "Energy Retrieval",
    1121: "Ultra Ball",
    1125: "Master Ball",
    1182: "Boss's Orders",
    1198: "Crispin",
    1205: "Cyrano",
    1210: "Brock's Scouting",
    1227: "Lillie's Determination",
    1250: "Area Zero Underdepths",
}

PACKAGE_ROOT = Path(__file__).resolve().parent
DECK = [
    int(line)
    for line in (PACKAGE_ROOT / "deck.csv").read_text(encoding="utf-8").splitlines()
    if line.strip()
]

ENERGY_NAMES = {
    "Basic {G} Energy",
    "Basic {L} Energy",
    "Basic {F} Energy",
    "Basic {P} Energy",
}

MAIN_ATTACKERS = [
    "Raging Bolt ex",
    "Teal Mask Ogerpon ex",
    "Mega Kangaskhan ex",
]

BENCH_PRIORITY = [
    "Raging Bolt ex",
    "Teal Mask Ogerpon ex",
    "Mega Kangaskhan ex",
    "Latias ex",
    "Fezandipiti ex",
    "Terapagos ex",
]

ABILITY_PRIORITY = [
    "Teal Mask Ogerpon ex",
    "Fezandipiti ex",
    "Latias ex",
]


def card_name_from_game_id(card_id):
    return CARD_NAME.get(int(card_id), f"UNKNOWN_{card_id}")


def get_my_player(obs):
    return obs.current.players[obs.current.yourIndex]


def card_value_by_name(name):
    if name == "Basic {G} Energy":
        return 25
    if name == "Basic {L} Energy":
        return 20
    if name == "Basic {F} Energy":
        return 15
    if name == "Basic {P} Energy":
        return 10

    if name == "Boss's Orders":
        return 20

    if name in {
        "Raging Bolt ex",
        "Teal Mask Ogerpon ex",
        "Fezandipiti ex",
        "Mega Kangaskhan ex",
    }:
        return 80

    if name == "Latias ex":
        return 70

    if name == "Terapagos ex":
        return 65

    if name in {
        "Crispin",
        "Lillie's Determination",
        "Boss's Orders",
        "Ultra Ball",
    }:
        return 50

    if name == "Energy Switch":
        return 5

    return 30


def play_priority(obs, name):
    if name == "Teal Mask Ogerpon ex":
        return 100

    if name == "Raging Bolt ex":
        return 95

    if name == "Fezandipiti ex":
        return 80

    if name == "Latias ex":
        return 70

    if name == "Terapagos ex":
        return 65

    if name == "Mega Kangaskhan ex":
        return 60

    if name == "Lillie's Determination":
        return 90

    if name == "Brock's Scouting":
        return 85

    if name == "Bug Catching Set":
        return 80

    if name == "Crispin":
        if count_energy_in_hand(obs) >= 2:
            return 20
        return 75

    if name == "Master Ball":
        return 70

    if name == "Dusk Ball":
        return 65

    if name == "Area Zero Underdepths":
        return 50

    if name == "Energy Retrieval":
        return 25

    if name == "Energy Switch":
        return 0

    if name == "Glass Trumpet":
        return 60

    if name == "Boss's Orders":
        return 5

    return 40


def option_hand_card_name(obs, op):
    my = obs.current.players[obs.current.yourIndex]
    if op.index is None or my.hand is None:
        return None
    return card_name_from_game_id(my.hand[op.index].id)


def get_target_pokemon(obs, op):
    my = obs.current.players[obs.current.yourIndex]

    if op.inPlayArea == AreaType.ACTIVE:
        return my.active[op.inPlayIndex]
    if op.inPlayArea == AreaType.BENCH:
        return my.bench[op.inPlayIndex]
    return None


def count_energy_in_hand(obs):
    my = obs.current.players[obs.current.yourIndex]
    hand = my.hand or []

    cnt = 0

    for card in hand:
        name = card_name_from_game_id(card.id)
        if name in ENERGY_NAMES:
            cnt += 1

    return cnt


def get_attach_target(option):
    if option.inPlayArea is None or option.inPlayIndex is None:
        return None
    return option.inPlayArea, option.inPlayIndex


def target_is_active(option):
    target = get_attach_target(option)
    if target is None:
        return False

    area, index = target
    return area == 0


def score_basic_to_bench_card(name):
    for i, key in enumerate(BENCH_PRIORITY):
        if key in name:
            return 100 - i * 10

    if "ex" in name:
        return 30

    return 0


def ability_score(obs, op):
    target = get_target_pokemon(obs, op)
    if target is None:
        return 0

    name = card_name_from_game_id(target.id)

    score = 0

    if name == "Teal Mask Ogerpon ex":
        score += 100
    elif name == "Fezandipiti ex":
        score += 80
    elif name == "Latias ex":
        score += 40
    else:
        score += 10

    if op.inPlayArea == AreaType.ACTIVE:
        score += 5

    return score


def choose_first(options, option_type):
    for i, op in enumerate(options):
        if op.type == option_type:
            return [i]
    return None


def should_play_ultra_ball(obs):
    my = obs.current.players[obs.current.yourIndex]
    hand = my.hand or []

    if len(hand) < 7:
        return False

    low = 0

    for card in hand:
        name = card_name_from_game_id(card.id)
        if card_value_by_name(name) <= 30:
            low += 1

    return low >= 2


def choose_discard_low_value(obs):
    hand = obs.current.players[obs.current.yourIndex].hand
    values = []

    for option_i, op in enumerate(obs.select.option):
        name = card_name_from_game_id(hand[op.index].id)
        v = card_value_by_name(name)
        values.append((v, option_i, op.index, name))

    values.sort()

    return [values[0][1], values[1][1]]


def choose_good_play(obs, options):
    best_i = None
    best_score = -10**9

    for i, op in enumerate(options):
        if op.type != OptionType.PLAY:
            continue

        name = option_hand_card_name(obs, op)

        if name == "Ultra Ball" and not should_play_ultra_ball(obs):
            continue

        if name == "Area Zero Underdepths" and obs.current.stadium:
            continue

        if name == "Energy Retrieval" and count_energy_in_hand(obs) >= 2:
            continue

        if name == "Crispin" and count_energy_in_hand(obs) >= 3:
            continue

        if name == "Energy Switch":
            continue

        score = play_priority(obs, name)

        if score > best_score:
            best_score = score
            best_i = i

    if best_i is None:
        return None
    return [best_i]


def choose_effect_choice(obs):
    context = obs.select.context
    options = obs.select.option

    if context == SelectContext.IS_FIRST:
        return [0]

    if context == SelectContext.MULLIGAN:
        return [0]

    if context in [
        SelectContext.SETUP_ACTIVE_POKEMON,
        SelectContext.SETUP_BENCH_POKEMON,
        SelectContext.TO_ACTIVE,
        SelectContext.TO_BENCH,
        SelectContext.TO_FIELD,
    ]:
        return [0]

    if context in [
        SelectContext.TO_HAND,
        SelectContext.TO_DECK,
        SelectContext.TO_DECK_BOTTOM,
        SelectContext.NOT_MOVE,
    ]:
        return choose_required_count(obs.select)

    if context in [
        SelectContext.DISCARD,
    ]:
        return choose_discard_low_value(obs)

    if context in [
        SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
    ]:
        return [0]

    if context in [
        SelectContext.ATTACH_FROM,
        SelectContext.ATTACH_TO,
        SelectContext.DETACH_FROM,
        SelectContext.DISCARD_ENERGY_CARD,
        SelectContext.DISCARD_TOOL_CARD,
        SelectContext.SWITCH_ENERGY_CARD,
        SelectContext.DISCARD_ENERGY,
        SelectContext.TO_HAND_ENERGY,
        SelectContext.TO_DECK_ENERGY,
        SelectContext.SWITCH_ENERGY,
    ]:
        return [0]

    if context == SelectContext.ATTACK:
        return [len(options) - 1]

    if context in [
        SelectContext.ACTIVATE,
        SelectContext.FIRST_EFFECT,
        SelectContext.MORE_DEVOLVE,
        SelectContext.COIN_HEAD,
    ]:
        return [0]

    return [0]


def choose_required_count(select):
    n = select.maxCount
    return list(range(min(n, len(select.option))))


def safe_first_action(obs):
    n = len(obs.select.option)
    min_c = obs.select.minCount
    max_c = obs.select.maxCount

    if max_c == 0:
        return []

    if n == 0:
        return []

    k = min_c

    if k == 0 and max_c > 0:
        k = 1

    k = min(k, max_c, n)

    return list(range(k))


def attach_score(obs, op):
    energy_name = option_hand_card_name(obs, op)
    if energy_name is None:
        return -10**9
    target = get_target_pokemon(obs, op)
    if target is None:
        return -10**9

    target_name = card_name_from_game_id(target.id)
    score = 0

    if energy_name == "Basic {G} Energy":
        if target_name == "Teal Mask Ogerpon ex":
            score += 100
        elif target_name == "Raging Bolt ex":
            score += 60

    elif energy_name == "Basic {L} Energy":
        if target_name == "Raging Bolt ex":
            score += 100
        elif target_name == "Teal Mask Ogerpon ex":
            score += 40

    elif energy_name == "Basic {F} Energy":
        if target_name == "Mega Kangaskhan ex":
            score += 100
        elif target_name == "Raging Bolt ex":
            score += 40

    hp = getattr(target, "hp", 999)
    if hp <= 50:
        score -= 80

    if op.inPlayArea == AreaType.ACTIVE:
        score += 20

    return score


def choose_best_attach(obs, options):
    my_player = get_my_player(obs)

    best_i = None
    best_score = -1

    for i, op in enumerate(options):
        if op.type != OptionType.ATTACH:
            continue

        if op.index is None:
            continue

        if op.index >= len(my_player.hand):
            continue

        score = attach_score(obs, op)

        if score > best_score:
            best_score = score
            best_i = i

    if best_i is not None and best_score > 0:
        return [best_i]

    return None


def choose_attack(options):
    attacks = [i for i, op in enumerate(options) if op.type == OptionType.ATTACK]
    if attacks:
        return [attacks[-1]]
    return None


def choose_best_play_basic_to_bench(obs, options):
    my_player = get_my_player(obs)

    best_i = None
    best_score = -1

    for i, op in enumerate(options):
        if op.type != OptionType.PLAY:
            continue

        if op.index is None:
            continue

        if op.index >= len(my_player.hand):
            continue

        card = my_player.hand[op.index]
        name = card_name_from_game_id(card.id)

        score = score_basic_to_bench_card(name)

        if score > best_score:
            best_score = score
            best_i = i

    if best_i is not None and best_score > 0:
        return [best_i]

    return None


def choose_best_ability(obs, options):
    best_i = None
    best_score = -10**9

    for i, op in enumerate(options):
        if op.type != OptionType.ABILITY:
            continue

        score = ability_score(obs, op)

        if score > best_score:
            best_score = score
            best_i = i

    if best_i is not None:
        return [best_i]

    return None


def _agent_impl(obs_dict: dict) -> list[int]:
    obs: Observation = to_observation_class(obs_dict)

    if obs.select is None:
        return list(DECK)

    options = obs.select.option

    if obs.select.context == SelectContext.MAIN:
        res = choose_best_play_basic_to_bench(obs, options)
        if res is not None:
            return res

        res = choose_best_attach(obs, options)
        if res is not None:
            return res

        res = choose_good_play(obs, options)
        if res is not None:
            return res

        res = choose_best_ability(obs, options)
        if res is not None:
            return res

        res = choose_attack(options)
        if res is not None:
            return res

        res = choose_first(options, OptionType.END)
        if res is not None:
            return res

    else:
        return choose_effect_choice(obs)

    return safe_first_action(obs)


def agent(obs_dict: dict) -> list[int]:
    if obs_dict.get("select") is None:
        return list(DECK)
    try:
        return _agent_impl(obs_dict)
    except Exception:
        return [0]
