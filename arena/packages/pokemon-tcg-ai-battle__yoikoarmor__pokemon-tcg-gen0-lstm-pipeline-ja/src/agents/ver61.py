"""Ver6.1: rank-2 public Mega Lucario + Solrock/Lunatone candidate."""

from cg.api import AreaType, CardType, OptionType, SelectContext

from .ver5 import GUST_IDS, _cd, _prize
from .ver58 import (
    AURA_JAB_ID,
    BASIC_FIGHTING_ID,
    FIGHTING_GONG_ID,
    GRAVITY_MOUNTAIN_ID,
    HARIYAMA_ID,
    JUDGE_ID,
    LILLIE_ID,
    LUNATONE_ID,
    MAKUHITA_ID,
    MEGA_LUCARIO_ID,
    POKE_PAD_ID,
    POWER_PROTEIN_ID,
    RIOLU_ID,
    SOLROCK_ID,
    Ver58Agent,
)


DUSK_BALL_ID = 1102
SWITCH_ID = 1123
HERO_CAPE_ID = 1159
XEROSIC_ID = 1197
WALLY_ID = 1229

RANK2_SEARCH_ITEMS = {DUSK_BALL_ID, FIGHTING_GONG_ID, POKE_PAD_ID}


class Ver61Agent(Ver58Agent):
    """V58 rules adjusted for the observed rank-2 Lucario list.

    The deck has a thicker 4-4 Lucario line and 3-3 Lunar engine, so the
    important change is not to blindly fill every bench slot. It searches the
    line first, keeps one Lunar pair, and treats Wally/Xerosic as situational
    tempo cards instead of generic supporters.
    """

    def _score_main(self, obs, op, plan):
        score = super()._score_main(obs, op, plan)
        st = obs.current
        me = st.players[st.yourIndex]

        if op.type == OptionType.PLAY:
            card = self._hand_card(obs, op)
            data = _cd(card.id) if card else None

            if card and card.id == DUSK_BALL_ID:
                return self._score_dusk_ball(obs, score)

            if card and card.id == WALLY_ID:
                return self._score_wally(obs, plan)

            if card and card.id == XEROSIC_ID:
                return self._score_xerosic(obs, plan)

            if card and card.id == GRAVITY_MOUNTAIN_ID:
                if st.stadiumPlayed:
                    return -1.0
                return 5200.0 if self._opponent_has_stage2(obs) else 2600.0

            if card and card.id == SWITCH_ID and plan.needs_switch:
                return 9900.0

            if data and data.cardType == int(CardType.POKEMON):
                return self._score_rank2_pokemon_play(obs, data, score)

        if op.type == OptionType.ATTACH:
            card = self._hand_card(obs, op)
            target = self._pokemon_at(obs, op.inPlayArea, op.inPlayIndex, st.yourIndex)
            if card and card.id == HERO_CAPE_ID and target is not None:
                target_cd = _cd(target.id)
                if target_cd and target_cd.megaEx:
                    damage = max(0, target_cd.hp - target.hp)
                    return 8900.0 + min(900.0, damage * 3.0)
                return 3500.0

        return score

    def _score_dusk_ball(self, obs, score):
        me = obs.current.players[obs.current.yourIndex]
        if me.deckCount <= 3:
            return -1.0
        if self._needs_line_search(obs):
            return max(score, 6600.0)
        if self._needs_lunar_pair(obs):
            return max(score, 5700.0)
        if self._opponent_wall_seen(obs) and not self._has_in_play(obs, MAKUHITA_ID):
            return max(score, 6100.0)
        return max(score, 4300.0)

    def _score_wally(self, obs, plan):
        st = obs.current
        if st.supporterPlayed:
            return -1.0
        best = self._best_damaged_mega(obs)
        if best is None:
            return -1.0
        _pokemon, damage, energy_count, active = best
        opp = st.players[1 - st.yourIndex]
        if plan.ko and active and damage < 180 and len(opp.prize) > 3:
            return -1.0
        score = 5000.0 + min(3200.0, damage * 18.0)
        if len(opp.prize) <= 3:
            score += 2200.0
        if active:
            score += 500.0
            if energy_count >= 2 and not plan.ko:
                score -= 800.0
        else:
            score += 900.0
        return score

    def _score_xerosic(self, obs, plan):
        st = obs.current
        if st.supporterPlayed:
            return -1.0
        if plan.needs_gust:
            return -1.0
        opp = st.players[1 - st.yourIndex]
        over = opp.handCount - 3
        if over <= 0:
            return -1.0
        score = 2600.0 + over * 720.0
        if opp.handCount >= 7:
            score += 900.0
        if plan.ko:
            score += 450.0
        return score

    def _score_rank2_pokemon_play(self, obs, data, score):
        st = obs.current
        me = st.players[st.yourIndex]
        bench_count = len(me.bench or [])
        line = self._line_state(obs)

        if data.cardId == RIOLU_ID:
            riolu_count = self._count_in_play(obs, RIOLU_ID)
            if not line["parent_in_play"]:
                return 25200.0 if line["mega_in_hand"] else 23500.0
            if not self._has_mega_in_play(obs) and riolu_count < 2 and bench_count <= 2:
                return 11200.0
            if self._has_mega_in_play(obs) and riolu_count < 1 and bench_count <= 3:
                return 9400.0
            return min(score, 2200.0)

        if data.cardId in (LUNATONE_ID, SOLROCK_ID):
            own_count = self._count_in_play(obs, data.cardId)
            if own_count >= 1:
                return min(score, 700.0)
            if bench_count >= 4:
                return min(score, 900.0)
            pair_bonus = 4300.0 if self._lunar_pair_progress(obs, data.cardId) else 0.0
            if not line["parent_in_play"] and bench_count >= 2:
                return 4200.0 + pair_bonus * 0.35
            return max(score, 9300.0 + pair_bonus)

        if data.cardId == MAKUHITA_ID:
            if self._opponent_wall_seen(obs):
                return 21600.0
            if bench_count >= 4:
                return min(score, 1000.0)
            return max(score, 7800.0 if line["parent_in_play"] else 6200.0)

        return score

    def _score_sub(self, obs, op, plan, ctx):
        effect = getattr(obs.select, "effect", None)
        effect_id = getattr(effect, "id", None)

        if effect_id == WALLY_ID and ctx in (
            SelectContext.HEAL,
            SelectContext.REMOVE_DAMAGE_COUNTER,
            SelectContext.TO_HAND,
        ):
            owner = op.playerIndex if op.playerIndex is not None else obs.current.yourIndex
            pokemon = self._pokemon_at(obs, op.area, op.index, owner)
            if owner == obs.current.yourIndex and pokemon is not None:
                data = _cd(pokemon.id)
                if data and data.megaEx:
                    damage = max(0, data.hp - pokemon.hp)
                    return 10000.0 + damage
            return 0.0

        if ctx == SelectContext.ACTIVATE:
            if getattr(effect, "id", None) == LUNATONE_ID:
                return self._score_lunar_activate(obs, op)

        if ctx in (SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.TO_ACTIVE, SelectContext.SWITCH):
            owner = op.playerIndex if op.playerIndex is not None else obs.current.yourIndex
            pokemon = self._pokemon_at(obs, op.area, op.index, owner)
            data = _cd(pokemon.id) if pokemon and owner == obs.current.yourIndex else None
            if data and data.cardId == RIOLU_ID and self._line_state(obs)["mega_in_hand"]:
                return 8200.0

        return super()._score_sub(obs, op, plan, ctx)

    def _score_lunar_activate(self, obs, op):
        if op.type != OptionType.YES:
            return 0.0
        st = obs.current
        me = st.players[st.yourIndex]
        energy_count = self._basic_energy_in_hand_count(obs)
        if energy_count <= 0 or me.deckCount <= 9:
            return 0.0
        if me.handCount <= 6 or energy_count >= 2:
            return 120.0
        if self._active_can_aura_jab(obs):
            return 110.0
        return 0.0

    def _score_search(self, obs, op):
        score = super()._score_search(obs, op)
        st = obs.current
        card = self._pokemon_at(
            obs,
            op.area,
            op.index,
            op.playerIndex if op.playerIndex is not None else st.yourIndex,
        )
        if card is None and obs.select.deck and op.index is not None and op.index < len(obs.select.deck):
            card = obs.select.deck[op.index]
        if card is None:
            return score

        line = self._line_state(obs)
        if card.id == RIOLU_ID:
            if not line["parent_in_play"]:
                return 34500.0
            if self._has_mega_in_play(obs) and self._count_in_play(obs, RIOLU_ID) == 0:
                return 13000.0
            return min(score, 500.0)
        if card.id == MEGA_LUCARIO_ID:
            if line["parent_in_play"]:
                return 34000.0
            if line["parent_in_hand"]:
                return 17000.0
            return min(score, 700.0)
        if card.id in (LUNATONE_ID, SOLROCK_ID):
            if self._count_in_play(obs, card.id) >= 1:
                return min(score, 600.0)
            return max(score, 17200.0 if self._needs_lunar_pair(obs) else 9400.0)
        if card.id == MAKUHITA_ID:
            if self._opponent_wall_seen(obs):
                return 31800.0
            return max(score, 8400.0)
        if card.id == HARIYAMA_ID:
            if self._has_in_play(obs, MAKUHITA_ID):
                return 28000.0 if self._opponent_wall_seen(obs) else 16800.0
            return max(score, 5200.0)
        if card.id == BASIC_FIGHTING_ID:
            return 14200.0 if self._needs_basic_energy(obs) else 2200.0
        if card.id in (LILLIE_ID, JUDGE_ID, WALLY_ID, XEROSIC_ID):
            return self._score_supporter_search(obs, card.id, score)
        if card.id == POWER_PROTEIN_ID:
            return 12800.0 if self._damage_boost_ko_gain(obs, 30) else score
        return score

    def _score_supporter_search(self, obs, card_id, score):
        st = obs.current
        me = st.players[st.yourIndex]
        opp = st.players[1 - st.yourIndex]
        if card_id in (LILLIE_ID, JUDGE_ID):
            return max(score, 9000.0 if me.handCount <= 3 and me.deckCount > 8 else score)
        if card_id == WALLY_ID:
            return max(score, 7400.0 if self._best_damaged_mega(obs) else score)
        if card_id == XEROSIC_ID:
            return max(score, 7600.0 if opp.handCount >= 6 else score)
        return score

    def _needs_lunar_pair(self, obs):
        return not (
            self._has_in_play(obs, LUNATONE_ID)
            and self._has_in_play(obs, SOLROCK_ID)
        )

    def _count_in_play(self, obs, card_id):
        me = obs.current.players[obs.current.yourIndex]
        return sum(
            1
            for pokemon in list(me.active or []) + list(me.bench or [])
            if pokemon is not None and pokemon.id == card_id
        )

    def _best_damaged_mega(self, obs):
        st = obs.current
        me = st.players[st.yourIndex]
        best = None
        for area, pokemon in [(AreaType.ACTIVE, me.active[0] if me.active else None)]:
            if pokemon is None:
                continue
            best = self._consider_wally_target(best, pokemon, area == AreaType.ACTIVE)
        for pokemon in me.bench or []:
            if pokemon is not None:
                best = self._consider_wally_target(best, pokemon, False)
        return best

    def _consider_wally_target(self, best, pokemon, active):
        data = _cd(pokemon.id)
        if data is None or not data.megaEx:
            return best
        damage = max(0, data.hp - pokemon.hp)
        if damage <= 0:
            return best
        item = (pokemon, damage, len(getattr(pokemon, "energies", [])), active)
        if best is None or item[1] > best[1]:
            return item
        return best

    def _active_can_aura_jab(self, obs):
        st = obs.current
        me = st.players[st.yourIndex]
        active = me.active[0] if me.active else None
        if active is None:
            return False
        data = _cd(active.id)
        return bool(data and AURA_JAB_ID in (data.attacks or []) and len(active.energies) >= 1)

    def _opponent_has_stage2(self, obs):
        opp = obs.current.players[1 - obs.current.yourIndex]
        for pokemon in list(opp.active or []) + list(opp.bench or []):
            data = _cd(pokemon.id) if pokemon else None
            if data and data.stage2:
                return True
        return False
