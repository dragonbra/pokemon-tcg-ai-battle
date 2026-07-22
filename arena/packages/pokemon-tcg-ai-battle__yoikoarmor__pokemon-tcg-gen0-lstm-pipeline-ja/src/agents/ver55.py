"""Ver5.5: route Cornerstone Mask Ogerpon's energy to the Mega Lucario line."""
from cg.api import SelectContext

from .ver5 import _cd
from .ver52 import Ver52Agent


CORNERSTONE_OGERPON_ID = 386


class Ver55Agent(Ver52Agent):
    """Keep Ver52's line search, but make Rock Kagura a real acceleration plan."""

    def _score_sub(self, obs, op, plan, ctx):
        effect = getattr(obs.select, "effect", None)
        if ctx == SelectContext.ATTACH_FROM and getattr(effect, "id", None) == CORNERSTONE_OGERPON_ID:
            owner = op.playerIndex if op.playerIndex is not None else obs.current.yourIndex
            pokemon = self._pokemon_at(obs, op.area, op.index, owner)
            data = _cd(pokemon.id) if pokemon else None
            if owner == obs.current.yourIndex and data is not None:
                energy_count = len(getattr(pokemon, "energies", []))
                if data.megaEx:
                    return 5000.0 - 100.0 * energy_count
                if data.name in self._mega_parent_names():
                    return 4500.0 - 100.0 * energy_count
        return super()._score_sub(obs, op, plan, ctx)
