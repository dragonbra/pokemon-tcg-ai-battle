# Decision 001: V8 Inheritance, Lopunny Override, And Ogerpon Weight

## Decision

0024 uses `marnies_grimmsnarl_ex_froslass_001` as its only focal policy. The
exact deck is the 48-member, rank-2-best Marnie's Grimmsnarl ex / Froslass
identity from the 2026-07-30 Top 100 snapshot.

All 51 Live branches initialize from 0023
`V8_add_rmy_ogerpon_default` update 0 model-only assets. The exact source
catalog, status, Foundation hash, source deck ID, source deck SHA-256, source
checkpoint path, and checkpoint SHA-256 are validated before V1 allocation.

The explicit exception is the Mega Lopunny family. All of
`mega_lopunny_ex_001`, `mega_lopunny_ex_mega_froslass_ex_001`, and
`mega_lopunny_ex_mega_froslass_ex_002` load 0023 final focal
`mega_lopunny_ex_001` decoder/value weights. The two different exact decks are
then saved as their own 0024 model-only branches, with their own target deck
metadata, fresh PPO optimizers, and independent later updates.

## Secondary Policy

`rmy_teal_mask_ogerpon_001` is the secondary training policy. Its Live
opponent selection weight is 10; all other Live opponent identities have
weight 1. This is a curriculum variable recorded in the formal config and
metrics. It does not alter Frozen policy behavior, complete Frozen evaluation,
or the definition of win-rate metrics.

Marnie remains the only focal policy. Ogerpon trajectories are optimized only
by the Ogerpon decoder/value optimizer, never treated as Marnie actions.
