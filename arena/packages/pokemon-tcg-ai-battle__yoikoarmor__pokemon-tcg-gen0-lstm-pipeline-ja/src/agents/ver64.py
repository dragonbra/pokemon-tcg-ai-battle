"""Ver6.4: matchup switches for Crustle, Lucario mirror, and Dragapult."""

from cg.api import AreaType, CardType, OptionType, SelectContext

from .ver5 import GUST_IDS, _cd, _prevents_ex_damage
from .ver61 import (
    HARIYAMA_ID,
    LUNATONE_ID,
    MAKUHITA_ID,
    MEGA_LUCARIO_ID,
    RIOLU_ID,
    SOLROCK_ID,
    SWITCH_ID,
    Ver61Agent,
)
from src import engine


DWEBBLE_ID = 344
CRUSTLE_ID = 345
DREEPY_ID = 119
DRAKLOAK_ID = 120
DRAGAPULT_EX_ID = 121

CRUSTLE_LINE_IDS = {DWEBBLE_ID, CRUSTLE_ID}
DRAGAPULT_LINE_IDS = {DREEPY_ID, DRAKLOAK_ID, DRAGAPULT_EX_ID}


class Ver64Agent(Ver61Agent):
    """Add explicit matchup plan gates.

    V63 scored well, but the logs suggest two matchup-specific leaks:
    Crustle is answered only after it becomes an active wall, and Lucario mirror
    sometimes trades Mega ex into Mega ex without setting a clean KO line. This
    version makes those branches visible before the generic attack plan wins.
    """

    def _score_main(self, obs, op, plan):
        score = super()._score_main(obs, op, plan)
        st = obs.current
        me = st.players[st.yourIndex]

        if op.type == OptionType.PLAY:
            card = self._hand_card(obs, op)
            data = _cd(card.id) if card else None
            if card and card.id in GUST_IDS:
                if self._crustle_line_seen(obs) and self._opponent_bench_has(obs, CRUSTLE_LINE_IDS):
                    return 10800.0
                if self._dragapult_line_seen(obs) and self._opponent_bench_has(obs, {DREEPY_ID, DRAKLOAK_ID}):
                    return 7600.0
            if card and card.id == SWITCH_ID:
                if self._should_wall_switch(obs) or self._should_dragapult_switch(obs):
                    return 10400.0
            if data and data.cardType == int(CardType.POKEMON):
                score = self._score_matchup_pokemon_play(obs, data, score)

        if op.type == OptionType.EVOLVE:
            target = self._pokemon_at(obs, op.inPlayArea, op.inPlayIndex, st.yourIndex)
            hand_card = self._hand_card(obs, op)
            data = _cd(hand_card.id) if hand_card else None
            if data and target is not None:
                if data.cardId == HARIYAMA_ID and self._crustle_line_seen(obs):
                    return 11800.0 + len(getattr(target, "energies", [])) * 400.0
                if data.cardId == MEGA_LUCARIO_ID and self._dragapult_line_seen(obs):
                    return 11200.0 + len(getattr(target, "energies", [])) * 450.0
                if data.cardId == MEGA_LUCARIO_ID and self._lucario_mirror_seen(obs):
                    return 10400.0 + len(getattr(target, "energies", [])) * 350.0

        if op.type == OptionType.ATTACH:
            card = self._hand_card(obs, op)
            data = _cd(card.id) if card else None
            target = self._pokemon_at(obs, op.inPlayArea, op.inPlayIndex, st.yourIndex)
            if data and data.cardType in (int(CardType.BASIC_ENERGY), int(CardType.SPECIAL_ENERGY)):
                if target is not None:
                    score = self._score_matchup_energy_attach(obs, target, op.inPlayArea, score)

        if op.type == OptionType.RETREAT:
            if self._should_wall_switch(obs) or self._should_dragapult_switch(obs):
                return 10100.0

        if op.type == OptionType.ATTACK:
            active = me.active[0] if me.active else None
            target = self._opponent_active(obs)
            active_cd = _cd(active.id) if active else None
            target_cd = _cd(target.id) if target else None
            if active_cd and target_cd and (active_cd.ex or active_cd.megaEx) and _prevents_ex_damage(target_cd):
                return -1000000.0
            if active and self._crustle_line_seen(obs) and active.id in (HARIYAMA_ID, MAKUHITA_ID, SOLROCK_ID, LUNATONE_ID, RIOLU_ID):
                if target and target.id in CRUSTLE_LINE_IDS:
                    return max(score, 1420.0)
            if active and self._lucario_mirror_seen(obs) and not self._is_ex(active):
                if target and target.id == MEGA_LUCARIO_ID:
                    return max(score, 1360.0)
            if active and self._dragapult_line_seen(obs) and active.id == MEGA_LUCARIO_ID:
                if self._should_dragapult_switch(obs) and not self._active_can_ko_active(obs):
                    return min(score, 250.0)

        return score

    def _score_matchup_pokemon_play(self, obs, data, score):
        if self._crustle_line_seen(obs):
            if data.cardId == MAKUHITA_ID:
                return 26800.0
            if data.cardId == HARIYAMA_ID:
                return score

        if self._dragapult_line_seen(obs):
            mega_bodies = self._count_in_play(obs, MEGA_LUCARIO_ID) + self._count_in_play(obs, RIOLU_ID)
            if data.cardId == RIOLU_ID and mega_bodies < 3:
                return 24800.0 if self._has_mega_in_play(obs) else 23800.0
            if data.cardId == MAKUHITA_ID:
                return min(score, 900.0)
            if data.cardId in (LUNATONE_ID, SOLROCK_ID):
                if mega_bodies < 2:
                    return min(score, 1600.0)
                if self._count_in_play(obs, data.cardId) >= 1:
                    return min(score, 500.0)

        if self._lucario_mirror_seen(obs):
            if data.cardId in (SOLROCK_ID, LUNATONE_ID):
                return max(score, 9400.0)
            if data.cardId == RIOLU_ID and self._count_in_play(obs, RIOLU_ID) + self._count_in_play(obs, MEGA_LUCARIO_ID) < 3:
                return max(score, 11800.0)
        return score

    def _score_matchup_energy_attach(self, obs, target, area, score):
        target_cd = _cd(target.id)
        if target_cd is None:
            return score

        if self._crustle_line_seen(obs):
            if target.id in (HARIYAMA_ID, MAKUHITA_ID):
                need = max(0, 3 - len(getattr(target, "energies", [])))
                return 9800.0 + need * 300.0
            if target.id == SOLROCK_ID and self._has_in_play(obs, LUNATONE_ID):
                return max(score, 7900.0)
            if target_cd.megaEx and not self._active_can_ko_active(obs):
                return score

        if self._dragapult_line_seen(obs):
            if target.id in (MEGA_LUCARIO_ID, RIOLU_ID):
                hp_bonus = getattr(target, "hp", 0) * 4.0 if target.id == MEGA_LUCARIO_ID else 0.0
                bench_bonus = 500.0 if area == AreaType.BENCH else 0.0
                return max(score, 8200.0 + hp_bonus + bench_bonus)
            if target.id in (MAKUHITA_ID, HARIYAMA_ID):
                return min(score, 1600.0)

        if self._lucario_mirror_seen(obs) and target.id in (MEGA_LUCARIO_ID, RIOLU_ID):
            return max(score, 7600.0)
        return score

    def _score_sub(self, obs, op, plan, ctx):
        if ctx in (SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.TO_ACTIVE, SelectContext.SWITCH):
            owner = op.playerIndex if op.playerIndex is not None else obs.current.yourIndex
            if owner == obs.current.yourIndex:
                pokemon = self._pokemon_at(obs, op.area, op.index, owner)
                if pokemon is not None:
                    if self._should_wall_switch(obs):
                        wall_score = self._wall_breaker_active_score(obs, pokemon)
                        if wall_score > 0:
                            return wall_score
                    if self._should_dragapult_switch(obs):
                        drag_score = self._dragapult_active_score(pokemon)
                        if drag_score > 0:
                            return drag_score
        if ctx == SelectContext.ATTACH_FROM:
            owner = op.playerIndex if op.playerIndex is not None else obs.current.yourIndex
            if owner == obs.current.yourIndex:
                pokemon = self._pokemon_at(obs, op.area, op.index, owner)
                if pokemon is not None:
                    if self._crustle_line_seen(obs) and pokemon.id in (HARIYAMA_ID, MAKUHITA_ID):
                        return 6000.0 - len(getattr(pokemon, "energies", [])) * 250.0
                    if self._dragapult_line_seen(obs) and pokemon.id in (MEGA_LUCARIO_ID, RIOLU_ID):
                        return 5600.0 + getattr(pokemon, "hp", 0) * 2.0
                    if self._lucario_mirror_seen(obs) and pokemon.id in (MEGA_LUCARIO_ID, RIOLU_ID):
                        return 5200.0 - len(getattr(pokemon, "energies", [])) * 180.0

        if ctx in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH, SelectContext.TO_FIELD):
            owner = op.playerIndex if op.playerIndex is not None else obs.current.yourIndex
            if owner == obs.current.yourIndex:
                pokemon = self._pokemon_at(obs, op.area, op.index, owner)
                if pokemon is not None:
                    if self._crustle_line_seen(obs) and pokemon.id == MAKUHITA_ID:
                        return 5000.0
                    if self._dragapult_line_seen(obs) and pokemon.id == RIOLU_ID:
                        return 5200.0

        return super()._score_sub(obs, op, plan, ctx)

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

        if self._crustle_line_seen(obs):
            if card.id == MAKUHITA_ID:
                return 36000.0
            if card.id == HARIYAMA_ID and self._has_in_play(obs, MAKUHITA_ID):
                return 35500.0
            if card.id == SOLROCK_ID and self._has_in_play(obs, LUNATONE_ID):
                return max(score, 18000.0)

        if self._dragapult_line_seen(obs):
            if card.id == RIOLU_ID:
                return 35000.0
            if card.id == MEGA_LUCARIO_ID and self._count_in_play(obs, RIOLU_ID) > 0:
                return 34800.0
            if card.id in (MAKUHITA_ID, HARIYAMA_ID):
                return min(score, 800.0)

        if self._lucario_mirror_seen(obs):
            if card.id == SOLROCK_ID and self._lunar_pair_progress(obs, SOLROCK_ID):
                return max(score, 20500.0)
            if card.id == LUNATONE_ID and self._lunar_pair_progress(obs, LUNATONE_ID):
                return max(score, 19000.0)
        return score

    def _crustle_line_seen(self, obs):
        return self._opponent_has_any(obs, CRUSTLE_LINE_IDS)

    def _dragapult_line_seen(self, obs):
        return self._opponent_has_any(obs, DRAGAPULT_LINE_IDS)

    def _lucario_mirror_seen(self, obs):
        return self._opponent_has_any(obs, {RIOLU_ID, MEGA_LUCARIO_ID})

    def _opponent_has_any(self, obs, ids):
        opp = obs.current.players[1 - obs.current.yourIndex]
        for zone in (opp.active or []), (opp.bench or []), (opp.discard or []):
            for pokemon in zone:
                if pokemon is not None and pokemon.id in ids:
                    return True
        return False

    def _opponent_bench_has(self, obs, ids):
        opp = obs.current.players[1 - obs.current.yourIndex]
        return any(pokemon is not None and pokemon.id in ids for pokemon in (opp.bench or []))

    def _opponent_active(self, obs):
        opp = obs.current.players[1 - obs.current.yourIndex]
        return opp.active[0] if opp.active else None

    def _is_ex(self, pokemon):
        data = _cd(pokemon.id) if pokemon else None
        return bool(data and (data.ex or data.megaEx))

    def _should_wall_switch(self, obs):
        if not self._crustle_line_seen(obs):
            return False
        active_zone = obs.current.players[obs.current.yourIndex].active or []
        active = active_zone[0] if active_zone else None
        if active is None or not self._is_ex(active):
            return False
        target = self._opponent_active(obs)
        if target is not None and target.id == CRUSTLE_ID:
            return self._best_wall_breaker_index(obs) >= 0
        return self._best_wall_breaker_index(obs, require_powered=True) >= 0

    def _best_wall_breaker_index(self, obs, require_powered=False):
        me = obs.current.players[obs.current.yourIndex]
        best = (-1, -1.0)
        for idx, pokemon in enumerate(me.bench or []):
            score = self._wall_breaker_active_score(obs, pokemon, require_powered=require_powered)
            if score > best[1]:
                best = (idx, score)
        return best[0]

    def _wall_breaker_active_score(self, obs, pokemon, require_powered=False):
        if pokemon is None or self._is_ex(pokemon):
            return 0.0
        data = _cd(pokemon.id)
        if data is None:
            return 0.0
        energy = len(getattr(pokemon, "energies", []))
        if pokemon.id == HARIYAMA_ID:
            if require_powered and energy < 3:
                return 0.0
            return 12000.0 + energy * 500.0
        if pokemon.id == MAKUHITA_ID:
            if require_powered and energy < 1:
                return 0.0
            return 7600.0 + energy * 450.0
        if pokemon.id == SOLROCK_ID and self._has_in_play(obs, LUNATONE_ID):
            if require_powered and energy < 1:
                return 0.0
            return 6900.0 + energy * 350.0
        if pokemon.id == LUNATONE_ID:
            if require_powered and energy < 2:
                return 0.0
            return 6100.0 + energy * 320.0
        if pokemon.id == RIOLU_ID:
            if require_powered and energy < 1:
                return 0.0
            return 5700.0 + energy * 300.0
        return 0.0

    def _should_dragapult_switch(self, obs):
        if not self._dragapult_line_seen(obs):
            return False
        active_zone = obs.current.players[obs.current.yourIndex].active or []
        active = active_zone[0] if active_zone else None
        if active is None or active.id != MEGA_LUCARIO_ID:
            return False
        if self._active_can_ko_active(obs):
            return False
        if getattr(active, "hp", 999) > 110:
            return False
        return self._best_healthy_mega_index(obs) >= 0

    def _best_healthy_mega_index(self, obs):
        me = obs.current.players[obs.current.yourIndex]
        best = (-1, -1.0)
        for idx, pokemon in enumerate(me.bench or []):
            score = self._dragapult_active_score(pokemon)
            if score > best[1]:
                best = (idx, score)
        return best[0]

    def _dragapult_active_score(self, pokemon):
        if pokemon is None or pokemon.id != MEGA_LUCARIO_ID:
            return 0.0
        energy = len(getattr(pokemon, "energies", []))
        if energy < 1:
            return 0.0
        return 9000.0 + getattr(pokemon, "hp", 0) * 8.0 + energy * 350.0

    def _mirror_chip_mode(self, obs):
        if not self._lucario_mirror_seen(obs):
            return False
        target = self._opponent_active(obs)
        if target is None or target.id != MEGA_LUCARIO_ID:
            return False
        if getattr(target, "hp", 0) <= 300:
            return False
        if self._active_can_ko_active(obs):
            return False
        me = obs.current.players[obs.current.yourIndex]
        active = me.active[0] if me.active else None
        return active is not None and not self._is_ex(active) and self._mirror_chipper_score(obs, active) > 0

    def _best_mirror_chipper_index(self, obs):
        me = obs.current.players[obs.current.yourIndex]
        best = (-1, -1.0)
        for idx, pokemon in enumerate(me.bench or []):
            score = self._mirror_chipper_score(obs, pokemon)
            if score > best[1]:
                best = (idx, score)
        return best[0]

    def _mirror_chipper_score(self, obs, pokemon):
        if pokemon is None or self._is_ex(pokemon):
            return 0.0
        energy = len(getattr(pokemon, "energies", []))
        if pokemon.id == SOLROCK_ID and self._has_in_play(obs, LUNATONE_ID) and energy >= 1:
            return 9800.0
        if pokemon.id == LUNATONE_ID and energy >= 2:
            return 8200.0
        if pokemon.id == RIOLU_ID and energy >= 1:
            return 7600.0
        if pokemon.id == HARIYAMA_ID and energy >= 3:
            return 7400.0
        if pokemon.id == MAKUHITA_ID and energy >= 1:
            return 5400.0
        return 0.0

    def _active_can_ko_active(self, obs):
        st = obs.current
        me = st.players[st.yourIndex]
        active = me.active[0] if me.active else None
        target = self._opponent_active(obs)
        if active is None or target is None:
            return False
        active_cd = _cd(active.id)
        if active_cd is None:
            return False
        attacker_is_ex = bool(active_cd.ex or active_cd.megaEx)
        current_bonus = 30 * self._power_protein_this_turn + (40 if self._black_belt_this_turn else 0)
        for attack_id in active_cd.attacks:
            attack = engine.attack_db().get(attack_id)
            if attack is None or len(getattr(active, "energies", [])) < len(attack.energies):
                continue
            damage = self._damage_vs(
                attack,
                active_cd.energyType,
                target,
                attacker_is_ex,
                current_bonus,
            )
            if damage >= getattr(target, "hp", 9999):
                return True
        return False
