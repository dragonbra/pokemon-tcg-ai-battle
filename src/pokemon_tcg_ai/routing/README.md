# Public-information routing

The final router is deliberately separated from model training. It observes only opponent Pokémon
identities marked public and certain by the official observation contract, updates monotonic
per-game memory, and selects a pre-audited complete policy head. It cannot read exact-deck labels,
hidden cards, Critic outputs, or match outcomes.

`router.py` contains the final Deck Router implementation; `rules.py` is the pure routing contract.
`base_router.py`, `meta_router.py`, and `oracle_audit.py` are internal compatibility and
storage-isolation utilities required to reconstruct the released router semantics.
