# Mega Kangaskhan ex + Crustle BC — Marnie v2 architecture

- model: `ptcg_kangaskhan_crustle_core_deck_conditioned_bc_epochmix_v2`
- deck hash: `e510e4d41bef`
- deck: `arena_agents/crustle/meta20260712_bestpolicy_crustle_e510e4d4`
- teacher window: 2026-07-22 through 2026-07-26 public replay wins
- strict core: `756 >= 2`, `344 >= 2`, `345 >= 2`
- teacher games: `1,221` wins across `17` exact deck hashes
- encoded decisions: `63,817` train / `7,025` validation
- validation leakage: `0` complete episodes
- best checkpoint: epoch 5
- validation loss: `0.464664`
- validation token accuracy: `83.34%`
- validation exact-action accuracy: `66.62%`
- runtime: two Tesla T4 GPUs with `DataParallel`; 7,380,162 parameters; 28.153 MiB fp32

This folder reuses Marnie v2's ID-only codec, count-aware deck mean-pool conditioning,
four-layer Transformer encoder, and pointer decoder. It is an independent target-deck
bundle; the existing Marnie and Crustle heuristic agents are unchanged.
