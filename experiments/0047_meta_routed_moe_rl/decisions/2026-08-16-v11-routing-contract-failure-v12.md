# V11 routing-contract failure and V12 restart

Date: 2026-08-16

V11 was stopped at durable checkpoint U4 by explicit user direction. It is a
failed experiment and must not be resumed or used as evidence for the intended
Meta-routed architecture.

Two implementation facts violated the intended contract:

1. The actor-side identifier loaded decks 001–070, maintained an exact-deck
   candidate set, and only then folded candidates to a 29-class Meta ID. The
   intended actor-side boundary is an ordered if-else classification from
   already-public opponent key Pokémon directly to a Meta ID. No exact opponent
   deck identity may be inferred, represented, or consumed.
2. `router_logits` contained six scalar E0-to-specialist gates. The intended
   Router is a trainable `29 × 7` lookup table: an identified Meta selects one
   row and a seven-way softmax produces all expert mixture weights. UNKNOWN has
   no row and remains fixed on E0.

V12 therefore attempted to start from immutable Policy-0814 with a newly initialized
seven-expert model and optimizer. It loads no V11 checkpoint or delta. The
public rulebook is `0047_public_meta29_priority_rules_v1`; the compact checkpoint
schema is `0047_meta_routed_moe_compact_fp32_delta_v2_meta29x7`.

Required pre-launch evidence:

- public/hidden observation boundary tests;
- full-public Pokémon signature audit matching all 70 frozen deck labels;
- `router_logits.shape == (29, 7)` and seven-way softmax/gradient tests;
- compact checkpoint strict round-trip;
- official-engine CUDA PPO smoke for the full seven-way execution path.

The full-softmax CUDA smoke passed before launch: 8/8 official-engine games were
valid, 694 PPO decisions passed the pre-update behavior-logprob hard gate, every
expert had nonzero effective use, the lane identity audit passed, and the Router
update norm was finite (`3.9427250158041716e-06`).

The V12 formal process then wrote its U0 compact checkpoint but failed before
W&B initialization and U0 evaluation because `save_router_table()` still called
the removed six-scalar `router_alphas()` API. V12 is retained as a startup-failed
version and is not reused. V13 adds a tested
`0047_meta29x7_lookup_softmax_router_v1` audit JSON containing all 29×7 logits
and rows, and restarts once more from immutable Policy-0814.

V13 passed runtime U0 execution and reached 192/512 games in its first rollout,
but the pre-commit audit found stale values in the FP16 candidate metadata:
`0047_public_exact_deck_candidates_v1` and `E0_to_Em_two_way_core6`. Although
the actual V13 runtime used the corrected classifier and 29×7 table, canonical
deployment evidence may not claim a different identity. V13 was stopped before
any PPO update. V14 fixes and tests those fields and restarts from Policy-0814.
