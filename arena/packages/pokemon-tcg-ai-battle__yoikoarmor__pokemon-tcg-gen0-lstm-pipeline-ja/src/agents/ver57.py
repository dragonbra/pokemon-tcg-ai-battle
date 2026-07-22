"""Ver5.7: gated sub-plan switching for Mega Lucario decks."""

from cg.api import AreaType, CardType, OptionType

from .ver5 import _cd, _dmg_vs, _prize
from .ver5 import _prevents_ex_damage
from .ver55 import Ver55Agent
from src import engine


SWITCH_IDS = {1123}
_WALL_NAMES = None


class Ver57Agent(Ver55Agent):
    """Keep the Mega Lucario line as default, then switch only on clear signals.

    The Archaludon observations suggest the useful lesson is not "use the sub
    plan more"; it is "hand the attack to a benched Pokemon only when the board
    proves it is the higher-value attacker." This variant keeps Ver55's main
    line/ramp behavior and adds a conservative tempo-switch check.
    """

    def reset(self):
        super().reset()
        self._sub_mode = "main"

    def _plan(self, obs):
        plan = super()._plan(obs)
        self._sub_mode = "wall" if self._wall else "main"
        if not self._wall:
            self._maybe_tempo_switch(obs, plan)
        return plan

    def _maybe_tempo_switch(self, obs, plan):
        if plan.ko:
            return
        st = obs.current
        me = st.players[st.yourIndex]
        opp = st.players[1 - st.yourIndex]
        active = me.active[0] if me.active else None
        opp_active = opp.active[0] if opp.active else None
        if active is None or opp_active is None:
            return

        best = self._best_bench_ko(me, opp_active)
        if best is None:
            return
        bench_idx, attack_id, prize = best

        active_cd = _cd(active.id)
        active_is_mega = bool(active_cd and active_cd.megaEx)
        # Do not abandon a healthy Mega for a one-prize tempo KO unless it has
        # failed to produce meaningful pressure. A prize-taking bench attacker
        # is still allowed when the active is not the established main line.
        if active_is_mega and prize <= 1 and plan.score >= 450:
            return

        plan.attack_id = attack_id
        plan.target_owner = 1 - st.yourIndex
        plan.target_is_active = True
        plan.target_bench_idx = -1
        plan.needs_energy = False
        plan.needs_gust = False
        plan.ko = True
        plan.score = 580000.0 + prize * 1000.0
        plan.needs_switch = True
        plan.switch_to = bench_idx
        self._sub_mode = "tempo"

    def _best_bench_ko(self, me, opp_active):
        adb = engine.attack_db()
        best = None
        for bi, pokemon in enumerate(me.bench or []):
            if pokemon is None:
                continue
            cd = _cd(pokemon.id)
            if cd is None:
                continue
            attacker_is_ex = bool(cd.ex or cd.megaEx)
            for attack_id in cd.attacks:
                attack = adb.get(attack_id)
                if attack is None or len(pokemon.energies) < len(attack.energies):
                    continue
                damage = _dmg_vs(attack, cd.energyType, opp_active, attacker_is_ex)
                if damage < opp_active.hp:
                    continue
                prize = _prize(opp_active)
                candidate = (prize, damage, bi, attack_id)
                if best is None or candidate > best:
                    best = candidate
        if best is None:
            return None
        prize, _damage, bi, attack_id = best
        return bi, attack_id, prize

    def _score_main(self, obs, op, plan):
        score = super()._score_main(obs, op, plan)
        st = obs.current
        me = st.players[st.yourIndex]

        if op.type == OptionType.PLAY:
            card = self._hand_card(obs, op)
            data = _cd(card.id) if card else None
            if data and data.cardType == int(CardType.POKEMON):
                return self._gate_pokemon_play(obs, data, score)

            if card and card.id in SWITCH_IDS and plan.needs_switch:
                return 9700.0

        if op.type == OptionType.ATTACH and self._sub_mode == "main":
            card = self._hand_card(obs, op)
            data = _cd(card.id) if card else None
            if data and data.cardType in (int(CardType.BASIC_ENERGY), int(CardType.SPECIAL_ENERGY)):
                target = self._pokemon_at(obs, op.inPlayArea, op.inPlayIndex, st.yourIndex)
                if target is not None and self._is_sub_attacker(obs, target):
                    if self._opponent_wall_seen(obs):
                        return score
                    line = self._line_state(obs)
                    if line["parent_in_play"] or self._has_mega_in_play(obs):
                        return min(score, 5200.0)

        return score

    def _gate_pokemon_play(self, obs, data, score):
        if self._sub_mode != "main":
            return score
        if self._opponent_wall_seen(obs):
            return score
        if data.megaEx or data.name in self._mega_parent_names():
            return score

        st = obs.current
        me = st.players[st.yourIndex]
        bench_count = len(me.bench or [])
        if bench_count == 0:
            return score

        line = self._line_state(obs)
        if not line["parent_in_play"]:
            # Keep one fallback body, but do not fill the Bench before the Mega
            # line exists. This is the V2.3 failure mode we want to avoid.
            return min(score, 9000.0 if bench_count < 2 else 500.0)
        if not self._has_mega_in_play(obs) and bench_count >= 3:
            return min(score, 3500.0)
        return score

    def _is_sub_attacker(self, obs, pokemon):
        data = _cd(pokemon.id)
        if data is None:
            return False
        if data.megaEx or data.name in self._mega_parent_names():
            return False
        return data.cardType == int(CardType.POKEMON)

    def _has_mega_in_play(self, obs):
        me = obs.current.players[obs.current.yourIndex]
        for pokemon in list(me.active or []) + list(me.bench or []):
            data = _cd(pokemon.id) if pokemon else None
            if data and data.megaEx:
                return True
        return False

    def _opponent_wall_seen(self, obs):
        global _WALL_NAMES
        if _WALL_NAMES is None:
            wall_names = set()
            for data in engine.card_db().values():
                if _prevents_ex_damage(data):
                    wall_names.add(data.name)
                    if data.evolvesFrom:
                        wall_names.add(data.evolvesFrom)
            _WALL_NAMES = wall_names
        opp = obs.current.players[1 - obs.current.yourIndex]
        for zone in (opp.active or []), (opp.bench or []), (opp.discard or []):
            for card in zone:
                data = _cd(card.id) if card else None
                if data and data.name in _WALL_NAMES:
                    return True
        return False
