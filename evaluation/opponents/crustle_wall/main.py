"""Opponent: dashimaki_anti_crustle - Simple rule-based Crustle wall deck"""
from pathlib import Path

from cg.api import Observation, to_observation_class, OptionType, SelectContext, AreaType, Pokemon, Card

PACKAGE_ROOT = Path(__file__).resolve().parent


def read_deck_csv() -> list[int]:
    return [
        int(line.strip())
        for line in (PACKAGE_ROOT / "deck.csv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


DECK = read_deck_csv()
my_deck = DECK


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    ps = obs.current.players[player_index]
    if area == AreaType.DECK:
        return obs.select.deck[index]
    elif area == AreaType.HAND:
        return ps.hand[index]
    elif area == AreaType.DISCARD:
        return ps.discard[index]
    elif area == AreaType.ACTIVE:
        return ps.active[index]
    elif area == AreaType.BENCH:
        return ps.bench[index]
    elif area == AreaType.PRIZE:
        return ps.prize[index]
    elif area == AreaType.STADIUM:
        return obs.current.stadium[index]
    elif area == AreaType.LOOKING:
        return obs.current.looking[index]
    else:
        return None


def agent(obs_dict: dict) -> list[int]:
    if obs_dict.get("select") is None:
        return list(DECK)
    try:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return list(DECK)

        select = obs.select
        options = select.option
        context = select.context

        scores = []
        for o in options:
            score = 0

            if context == SelectContext.MAIN:
                if o.type == OptionType.ATTACH:
                    score = 1000
                    card = get_card(obs, o.area, o.index, obs.current.yourIndex)
                    if card is not None and card.id == 1159:
                        if o.inPlayArea == AreaType.ACTIVE:
                            score = 2100
                        else:
                            score = 0
                elif o.type == OptionType.EVOLVE:
                    score = 800
                elif o.type == OptionType.PLAY:
                    score = 600
                    card = get_card(obs, AreaType.HAND, o.index, obs.current.yourIndex)
                    if card is not None:
                        if card.id == 1147:
                            active = obs.current.players[obs.current.yourIndex].active
                            if len(active) > 0 and active[0] is not None:
                                pokemon = active[0]
                                if pokemon.hp < pokemon.maxHp and len(pokemon.energies) >= 3:
                                    score = 2000
                                else:
                                    score = 0
                        elif card.id == 1212:
                            active = obs.current.players[obs.current.yourIndex].active
                            if len(active) > 0 and active[0] is not None:
                                pokemon = active[0]
                                if pokemon.hp < pokemon.maxHp:
                                    score = 1500
                                else:
                                    score = 0
                        elif card.id == 1224:
                            score = 1400
                        elif card.id == 1264:
                            score = 1300
                elif o.type == OptionType.ABILITY:
                    score = 400
                elif o.type == OptionType.ATTACK:
                    score = 100
                elif o.type == OptionType.RETREAT:
                    score = -1
            else:
                score = 2000

                if o.type == OptionType.CARD:
                    card = get_card(obs, o.area, o.index, o.playerIndex)
                    if card is not None:
                        if context == SelectContext.EVOLVE or context == SelectContext.TO_BENCH:
                            score += 500

                        if isinstance(card, Pokemon):
                            if o.playerIndex != obs.current.yourIndex:
                                score += 500 if o.area == AreaType.ACTIVE else 100
                                score += len(card.energies) * 50
                            else:
                                score += card.hp

                elif o.type == OptionType.YES:
                    score += 100
                elif o.type == OptionType.NUMBER:
                    score += o.number

            scores.append(score)

        sorted_options = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        output = []
        for i in range(min(len(sorted_options), select.maxCount)):
            idx = sorted_options[i]
            if scores[idx] >= 0 or len(output) < select.minCount:
                output.append(idx)

        return output
    except Exception:
        return [0]
