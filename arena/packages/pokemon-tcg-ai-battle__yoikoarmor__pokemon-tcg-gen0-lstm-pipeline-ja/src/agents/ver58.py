"""Ver5.8: public-style Mega Lucario ex with Hariyama and Lunar Cycle support."""

from cg.api import AreaType, CardType, OptionType, SelectContext

from .ver5 import GUST_IDS, Plan, _cd, _prevents_ex_damage, _prize
from .ver57 import Ver57Agent
from src import engine


BASIC_FIGHTING_ID = 6
ROCK_FIGHTING_ID = 20
MAKUHITA_ID = 673
HARIYAMA_ID = 674
LUNATONE_ID = 675
SOLROCK_ID = 676
RIOLU_ID = 677
MEGA_LUCARIO_ID = 678

POWER_PROTEIN_ID = 1141
BLACK_BELT_ID = 1211
FIGHTING_GONG_ID = 1142
POKE_PAD_ID = 1152
ULTRA_BALL_ID = 1121
NIGHT_STRETCHER_ID = 1097
SECRET_BOX_ID = 1092
AIR_BALLOON_ID = 1174
BROCK_ID = 1210
JUDGE_ID = 1213
LILLIE_ID = 1227
GRAVITY_MOUNTAIN_ID = 1252

AURA_JAB_ID = 982
MEGA_BRAVE_ID = 983
WILD_PRESS_ID = 978

SEARCH_ITEM_IDS = {FIGHTING_GONG_ID, POKE_PAD_ID, ULTRA_BALL_ID, NIGHT_STRETCHER_ID, SECRET_BOX_ID}
PUBLIC_DRAW_SUPPORTERS = {LILLIE_ID, JUDGE_ID, BROCK_ID}


class Ver58Agent(Ver57Agent):
    """Rule scorer tuned for the public Mega Lucario / Hariyama list.

    V57 already keeps the Mega Lucario line as the default and only pivots to a
    sub attacker on clear signals. V58 keeps that shape, but fixes two public
    Lucario details that matter a lot: damage boosts should be spent when they
    change a KO line, and Aura Jab should be valued as bench-energy acceleration.
    """

    def __init__(self, deck=None, rng=None):
        super().__init__(deck=deck, rng=rng)
        self._tracked_turn = -1
        self._power_protein_this_turn = 0
        self._black_belt_this_turn = False

    def reset(self):
        super().reset()
        self._tracked_turn = -1
        self._power_protein_this_turn = 0
        self._black_belt_this_turn = False

    def scores_for(self, obs):
        self._track_turn_effects(obs)
        return super().scores_for(obs)

    def _track_turn_effects(self, obs):
        if obs.current is None:
            return
        turn = obs.current.turn
        if turn != self._tracked_turn:
            self._tracked_turn = turn
            self._power_protein_this_turn = 0
            self._black_belt_this_turn = False
        me_idx = obs.current.yourIndex
        for log in obs.logs or []:
            if log.playerIndex != me_idx:
                continue
            if log.cardId == POWER_PROTEIN_ID and log.toArea == AreaType.DISCARD:
                self._power_protein_this_turn += 1
            elif log.cardId == BLACK_BELT_ID and log.toArea == AreaType.DISCARD:
                self._black_belt_this_turn = True

    # ---- planning ---------------------------------------------------------
    def _plan(self, obs):
        st = obs.current
        me = st.players[st.yourIndex]
        opp = st.players[1 - st.yourIndex]
        opp_idx = 1 - st.yourIndex
        best = Plan()
        self._wall = False
        self._breaker_bi = -1
        self._sub_mode = "main"
        if st.turn < 2:
            return best
        active = me.active[0] if me.active else None
        if active is None:
            return best
        active_cd = _cd(active.id)
        if active_cd is None:
            return best

        gust_ok = self._has_playable(obs, GUST_IDS) and not st.supporterPlayed
        targets = []
        if opp.active and opp.active[0] is not None:
            targets.append((opp.active[0], True, -1))
        if gust_ok:
            for bench_idx, pokemon in enumerate(opp.bench or []):
                if pokemon is not None:
                    targets.append((pokemon, False, bench_idx))

        cur_energy = len(active.energies)
        can_attach = (not st.energyAttached) and self._has_energy_in_hand(obs)
        damage_bonus = 30 * self._power_protein_this_turn
        if self._black_belt_this_turn:
            damage_bonus += 40
        aura_bonus = self._aura_accel_bonus(obs)

        adb = engine.attack_db()
        attacker_is_ex = bool(active_cd.ex or active_cd.megaEx)
        for attack_id in active_cd.attacks:
            attack = adb.get(attack_id)
            if attack is None:
                continue
            cost = len(attack.energies)
            if cur_energy >= cost:
                needs_energy = False
            elif can_attach and cur_energy + 1 >= cost:
                needs_energy = True
            else:
                continue
            for target, is_active, bench_idx in targets:
                damage = self._damage_vs(
                    attack, active_cd.energyType, target, attacker_is_ex, damage_bonus,
                )
                if damage <= 0:
                    continue
                ko = target.hp <= damage
                prize = _prize(target) if ko else 0
                score = prize * 10000 + (target.hp if ko else damage)
                if is_active:
                    score += 300
                if len(opp.prize) <= prize and prize > 0:
                    score = 500000
                if attack_id == AURA_JAB_ID and aura_bonus > 0:
                    score += aura_bonus
                if attack_id == MEGA_BRAVE_ID and not ko and aura_bonus > 0:
                    score -= min(350.0, aura_bonus * 0.45)
                if score > best.score:
                    best = Plan(
                        attack_id, opp_idx, is_active, bench_idx, needs_energy,
                        not is_active, ko, score,
                    )

        self._apply_wall_plan(obs, best, active_cd, active, opp_idx)
        if self._wall:
            self._sub_mode = "wall"
        else:
            self._maybe_tempo_switch(obs, best)
        return best

    def _damage_vs(self, attack, attacker_type, target, attacker_is_ex, bonus=0):
        damage = attack.damage
        if damage > 0 and "flip a coin" in (attack.text or "").lower():
            damage = int(damage * 0.7)
        target_cd = _cd(target.id)
        if target_cd is None:
            return damage
        if attacker_is_ex and _prevents_ex_damage(target_cd):
            return 0
        if damage > 0 and int(attacker_type) == 6:
            damage += bonus
        if target_cd.weakness is not None and int(target_cd.weakness) == int(attacker_type):
            damage *= 2
        if target_cd.resistance is not None and int(target_cd.resistance) == int(attacker_type):
            damage -= 30
        return max(0, damage)

    def _apply_wall_plan(self, obs, best, active_cd, active, opp_idx):
        me = obs.current.players[obs.current.yourIndex]
        opp = obs.current.players[1 - obs.current.yourIndex]
        opp_active = opp.active[0] if opp.active else None
        if opp_active is None:
            return

        adb = engine.attack_db()
        active_best = 0
        active_is_ex = bool(active_cd.ex or active_cd.megaEx)
        for attack_id in active_cd.attacks:
            attack = adb.get(attack_id)
            if attack is not None:
                active_best = max(
                    active_best,
                    self._damage_vs(attack, active_cd.energyType, opp_active, active_is_ex, 0),
                )
        if active_best >= 20:
            return

        best_bi, powered_ko_bi = -1, -1
        for bench_idx, pokemon in enumerate(me.bench or []):
            card = _cd(pokemon.id) if pokemon else None
            if card is None:
                continue
            attacker_is_ex = bool(card.ex or card.megaEx)
            for attack_id in card.attacks:
                attack = adb.get(attack_id)
                if attack is None:
                    continue
                damage = self._damage_vs(attack, card.energyType, opp_active, attacker_is_ex, 0)
                if damage <= 0:
                    continue
                if best_bi < 0 or pokemon.id == HARIYAMA_ID:
                    best_bi = bench_idx
                if len(pokemon.energies) >= len(attack.energies) and opp_active.hp <= damage:
                    powered_ko_bi = bench_idx
        if best_bi >= 0:
            self._wall = True
            self._breaker_bi = powered_ko_bi if powered_ko_bi >= 0 else best_bi
            if powered_ko_bi >= 0:
                best.target_owner = opp_idx
                best.target_is_active = True
                best.ko = True
                best.score = 600000.0
                best.needs_switch = True
                best.switch_to = powered_ko_bi

    # ---- main option scoring ----------------------------------------------
    def _score_main(self, obs, op, plan):
        score = super()._score_main(obs, op, plan)
        st = obs.current
        me = st.players[st.yourIndex]

        if op.type == OptionType.PLAY:
            card = self._hand_card(obs, op)
            data = _cd(card.id) if card else None
            if card and card.id in (LILLIE_ID, JUDGE_ID):
                if st.supporterPlayed or me.deckCount <= 14 or me.handCount >= 7:
                    return -1.0
                return 5200.0
            if card and card.id == POWER_PROTEIN_ID:
                return self._score_damage_boost_play(obs, plus=30, supporter=False)
            if card and card.id == BLACK_BELT_ID:
                return self._score_damage_boost_play(obs, plus=40, supporter=True)
            if card and card.id == BROCK_ID:
                if st.supporterPlayed or me.deckCount <= 6:
                    return -1.0
                return 7200.0 if self._needs_line_search(obs) else score
            if card and card.id in SEARCH_ITEM_IDS:
                if me.deckCount <= 6 and not self._needs_line_search(obs):
                    return -1.0
                return max(score, 5200.0)
            if data and data.cardType == int(CardType.POKEMON):
                return self._score_public_pokemon_play(obs, data, score)
            if card and card.id == GRAVITY_MOUNTAIN_ID:
                return 4300.0 if not st.stadiumPlayed else -1.0
            if card and card.id == AIR_BALLOON_ID:
                target = self._pokemon_at(obs, op.inPlayArea, op.inPlayIndex, st.yourIndex)
                target_cd = _cd(target.id) if target else None
                if target_cd and target_cd.megaEx:
                    return 7600.0

        if op.type == OptionType.ATTACK and op.attackId == AURA_JAB_ID and plan.attack_id == AURA_JAB_ID:
            return 1300.0
        if op.type == OptionType.ATTACH:
            card = self._hand_card(obs, op)
            if card and card.id == ROCK_FIGHTING_ID and op.inPlayArea == AreaType.ACTIVE:
                return max(score, 7600.0)
        return score

    def _score_damage_boost_play(self, obs, plus, supporter):
        st = obs.current
        if supporter and st.supporterPlayed:
            return -1.0
        if st.players[st.yourIndex].deckCount <= 2:
            return -1.0
        ko_gain = self._damage_boost_ko_gain(obs, plus)
        if ko_gain >= 3:
            return 9800.0
        if ko_gain == 2:
            return 9000.0
        if ko_gain == 1:
            return 8400.0
        return -1.0

    def _damage_boost_ko_gain(self, obs, plus):
        st = obs.current
        me = st.players[st.yourIndex]
        opp = st.players[1 - st.yourIndex]
        active = me.active[0] if me.active else None
        target = opp.active[0] if opp.active else None
        if active is None or target is None:
            return 0
        active_cd = _cd(active.id)
        if active_cd is None or int(active_cd.energyType) != 6:
            return 0
        current_bonus = 30 * self._power_protein_this_turn + (40 if self._black_belt_this_turn else 0)
        adb = engine.attack_db()
        best_gain = 0
        for attack_id in active_cd.attacks:
            attack = adb.get(attack_id)
            if attack is None or len(active.energies) < len(attack.energies):
                continue
            if attack.damage <= 0:
                continue
            base = self._damage_vs(attack, active_cd.energyType, target, bool(active_cd.ex or active_cd.megaEx), current_bonus)
            boosted = self._damage_vs(attack, active_cd.energyType, target, bool(active_cd.ex or active_cd.megaEx), current_bonus + plus)
            if base < target.hp <= boosted:
                best_gain = max(best_gain, _prize(target))
        return best_gain

    def _score_public_pokemon_play(self, obs, data, score):
        st = obs.current
        me = st.players[st.yourIndex]
        bench_count = len(me.bench or [])
        line = self._line_state(obs)
        if data.cardId == RIOLU_ID and not line["parent_in_play"]:
            return 24500.0 if line["mega_in_hand"] else 23000.0
        if data.cardId == MAKUHITA_ID:
            if self._opponent_wall_seen(obs):
                return 21500.0
            return max(score, 14500.0 if bench_count <= 2 else 6500.0)
        if data.cardId in (LUNATONE_ID, SOLROCK_ID):
            if bench_count >= 4 and not line["parent_in_play"]:
                return 500.0
            pair_bonus = 3500.0 if self._lunar_pair_progress(obs, data.cardId) else 0.0
            return max(score, 8800.0 + pair_bonus)
        if data.cardId == HARIYAMA_ID:
            return score
        return score

    # ---- sub selections ----------------------------------------------------
    def _score_sub(self, obs, op, plan, ctx):
        if ctx == SelectContext.ATTACH_FROM and self._is_aura_attach(obs):
            owner = op.playerIndex if op.playerIndex is not None else obs.current.yourIndex
            pokemon = self._pokemon_at(obs, op.area, op.index, owner)
            if owner == obs.current.yourIndex and pokemon is not None:
                return self._aura_attach_target_score(obs, pokemon)
        if ctx == SelectContext.ACTIVATE:
            effect = getattr(obs.select, "effect", None)
            if getattr(effect, "id", None) == LUNATONE_ID:
                ok = (
                    op.type == OptionType.YES
                    and self._basic_energy_in_hand_count(obs) >= 1
                    and obs.current.players[obs.current.yourIndex].deckCount > 8
                )
                return 100.0 if ok else 0.0
        return super()._score_sub(obs, op, plan, ctx)

    def _score_search(self, obs, op):
        score = super()._score_search(obs, op)
        st = obs.current
        me = st.players[st.yourIndex]
        card = self._pokemon_at(
            obs, op.area, op.index,
            op.playerIndex if op.playerIndex is not None else st.yourIndex,
        )
        if card is None and obs.select.deck and op.index is not None and op.index < len(obs.select.deck):
            card = obs.select.deck[op.index]
        if card is None:
            return score
        data = _cd(card.id)
        if data is None:
            return score

        line = self._line_state(obs)
        if card.id == RIOLU_ID and not line["parent_in_play"]:
            return 33000.0
        if card.id == MEGA_LUCARIO_ID and line["parent_in_play"]:
            return 33500.0
        if card.id == MAKUHITA_ID:
            if self._opponent_wall_seen(obs):
                return 31500.0
            if self._has_card_in_hand(obs, HARIYAMA_ID):
                return 25500.0
            return max(score, 16500.0)
        if card.id == HARIYAMA_ID:
            if self._has_in_play(obs, MAKUHITA_ID):
                return 31000.0
            return max(score, 15500.0)
        if card.id in (LUNATONE_ID, SOLROCK_ID):
            return max(score, 14500.0 if self._lunar_pair_progress(obs, card.id) else 9000.0)
        if card.id == BASIC_FIGHTING_ID:
            return 13500.0 if self._needs_basic_energy(obs) else 3000.0
        if card.id in (POWER_PROTEIN_ID, BLACK_BELT_ID):
            return 12000.0 if self._damage_boost_ko_gain(obs, 30) else score
        if card.id in PUBLIC_DRAW_SUPPORTERS:
            return 8500.0 if me.handCount <= 3 and me.deckCount > 8 else score
        return score

    # ---- helpers -----------------------------------------------------------
    def _has_energy_in_hand(self, obs):
        me = obs.current.players[obs.current.yourIndex]
        for card in me.hand or []:
            data = _cd(card.id)
            if data and data.cardType in (int(CardType.BASIC_ENERGY), int(CardType.SPECIAL_ENERGY)):
                return True
        return False

    def _basic_energy_in_hand_count(self, obs):
        me = obs.current.players[obs.current.yourIndex]
        return sum(1 for card in (me.hand or []) if card.id == BASIC_FIGHTING_ID)

    def _needs_basic_energy(self, obs):
        st = obs.current
        me = st.players[st.yourIndex]
        if not st.energyAttached:
            return True
        return any(card.id == BASIC_FIGHTING_ID for card in (me.discard or []))

    def _has_card_in_hand(self, obs, card_id):
        me = obs.current.players[obs.current.yourIndex]
        return any(card.id == card_id for card in (me.hand or []))

    def _has_in_play(self, obs, card_id):
        me = obs.current.players[obs.current.yourIndex]
        return any(pokemon and pokemon.id == card_id for pokemon in list(me.active or []) + list(me.bench or []))

    def _needs_line_search(self, obs):
        line = self._line_state(obs)
        if not line["parent_in_play"] or (line["parent_in_play"] and not line["mega_in_hand"]):
            return True
        if self._opponent_wall_seen(obs) and not (self._has_in_play(obs, MAKUHITA_ID) or self._has_in_play(obs, HARIYAMA_ID)):
            return True
        return False

    def _lunar_pair_progress(self, obs, card_id):
        counterpart = SOLROCK_ID if card_id == LUNATONE_ID else LUNATONE_ID
        return self._has_in_play(obs, counterpart) or self._has_card_in_hand(obs, counterpart)

    def _is_aura_attach(self, obs):
        effect = getattr(obs.select, "effect", None)
        if getattr(effect, "id", None) == MEGA_LUCARIO_ID:
            return True
        return any(getattr(log, "attackId", None) == AURA_JAB_ID for log in (obs.logs or []))

    def _aura_accel_bonus(self, obs):
        me = obs.current.players[obs.current.yourIndex]
        discard_energy = sum(1 for card in (me.discard or []) if card.id == BASIC_FIGHTING_ID)
        if discard_energy <= 0:
            return 0.0
        needs = [self._target_energy_for(pokemon.id) - len(pokemon.energies)
                 for pokemon in (me.bench or []) if pokemon is not None]
        needs = [need for need in needs if need > 0]
        if not needs:
            return 0.0
        total_need = min(3, discard_energy, sum(needs))
        return 620.0 + 140.0 * total_need + (130.0 if len(needs) >= 2 else 0.0)

    def _aura_attach_target_score(self, obs, pokemon):
        target = self._target_energy_for(pokemon.id)
        if target <= 0:
            return 0.0
        need = max(0, target - len(pokemon.energies))
        if need <= 0:
            return 50.0
        base = 1000.0 + 700.0 * need
        if pokemon.id in (HARIYAMA_ID, MAKUHITA_ID):
            base += 900.0 if self._opponent_wall_seen(obs) else 350.0
        if pokemon.id in (MEGA_LUCARIO_ID, RIOLU_ID):
            base += 550.0
        return base

    def _target_energy_for(self, card_id):
        if card_id in (MEGA_LUCARIO_ID, RIOLU_ID):
            return 2
        if card_id in (HARIYAMA_ID, MAKUHITA_ID):
            return 3
        if card_id == SOLROCK_ID:
            return 1
        if card_id == LUNATONE_ID:
            return 2
        return 0
