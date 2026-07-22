"""Ver5.2: Mega Lucario line-aware setup with a Koraidon ex fallback plan."""
from cg.api import AreaType, CardType, OptionType, SelectContext

from .ver5 import Ver5Agent, _cd


class Ver52Agent(Ver5Agent):
    """Prefer a complete Mega line before searching redundant evolutions.

    Mega Lucario ex cannot be useful until a Riolu survives a turn. The base
    Ver5 scorer values every Pokemon search similarly, which can leave a Mega
    stranded in hand. This agent makes that dependency explicit and treats a
    Basic ex such as Koraidon as the tempo-preserving fallback.
    """

    def _mega_parent_names(self):
        names = set()
        for card_id in self.deck or []:
            card = _cd(card_id)
            if card and card.megaEx and card.evolvesFrom:
                names.add(card.evolvesFrom)
        return names

    def _line_state(self, obs):
        st = obs.current
        me = st.players[st.yourIndex]
        parents = self._mega_parent_names()
        board = list(me.active or []) + list(me.bench or [])

        def is_parent(card):
            data = _cd(card.id)
            return data is not None and data.name in parents

        def is_mega(card):
            data = _cd(card.id)
            return data is not None and data.megaEx

        return {
            "parent_in_play": any(card is not None and is_parent(card) for card in board),
            "parent_in_hand": any(is_parent(card) for card in (me.hand or [])),
            "mega_in_hand": any(is_mega(card) for card in (me.hand or [])),
        }

    def _score_main(self, obs, op, plan):
        score = super()._score_main(obs, op, plan)
        if op.type != OptionType.PLAY:
            return score

        card = self._hand_card(obs, op)
        data = _cd(card.id) if card else None
        if data is None or data.cardType != int(CardType.POKEMON):
            return score

        line = self._line_state(obs)
        parents = self._mega_parent_names()
        if data.name in parents and not line["parent_in_play"]:
            # Bench Riolu now; a Mega already in hand can evolve it next turn.
            return 24000.0 if line["mega_in_hand"] else 22500.0
        if data.ex and not data.megaEx and not line["parent_in_play"]:
            # Koraidon ex is the early attacker when the Mega line is missing.
            return 21000.0
        return score

    def _score_sub(self, obs, op, plan, ctx):
        if ctx in (SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.TO_ACTIVE, SelectContext.SWITCH):
            owner = op.playerIndex if op.playerIndex is not None else obs.current.yourIndex
            pokemon = self._pokemon_at(
                obs, op.area, op.index, owner,
            )
            data = _cd(pokemon.id) if pokemon else None
            if owner == obs.current.yourIndex and data is not None and data.name in self._mega_parent_names():
                # When Mega is ready, start charging Riolu immediately.
                if self._line_state(obs)["mega_in_hand"]:
                    return 7000.0
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
        data = _cd(card.id) if card else None
        if data is None:
            return score

        line = self._line_state(obs)
        parents = self._mega_parent_names()
        if data.name in parents and not line["parent_in_play"]:
            return 28000.0 if line["mega_in_hand"] else 24000.0
        if data.megaEx:
            if line["parent_in_play"]:
                return 30000.0
            if line["parent_in_hand"]:
                return 18000.0
            # Do not strand an evolution while neither its Basic nor a way to
            # play it is available; the Koraidon fallback can keep pressure up.
            return max(score, 100.0)
        return score
