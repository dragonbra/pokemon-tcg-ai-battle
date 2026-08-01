# 0024 Marnie's Grimmsnarl ex / Froslass 001 League PPO

## Current Stage

0024 is a new self-contained continuous League PPO project. The focal policy is
`marnies_grimmsnarl_ex_froslass_001`, exact SHA-256
`c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af`.
Its 2026-07-30 Top 100 exact-deck evidence identifies 48 members, a best rank
of 2, and an aggregate effective win rate of 60.17%.

The policy runs against the fixed `0019_foundation_51_exact_decks_v4` pool.
Frozen opponents use the shared Foundation policy. Live opponents use isolated
decoder/value branches, each trained only from decisions made by that exact
deck branch. `source_id` remains neutral `0`; team and source identity are
provenance only, never actor-visible input.

## Model And Action Contract

The Foundation encoder is frozen. Each Live policy owns its pointer key/query,
option bias, decoder initialization, decoder, stop head, and scalar value head.
The actor selects only official-engine legal options. A stochastic behavior
policy produces rollout diagnostics; fixed-seed balanced-seat greedy Frozen
evaluation is the required strength evidence.

Terminal rewards are win `+1`, loss `-1`, and draw `0`; PPO uses GAE with
`gamma=1.0` and `lambda=0.95`. Checkpoints are model-only and contain decoder,
value, deck identity, Foundation hash, version, and update, never optimizer,
RNG, replay, or rollout state.

## Initialization

V1 is initialized from immutable 0023
`V8_add_rmy_ogerpon_default` update 0 materialized decoder/value checkpoints.
For ordinary exact decks, the source and target deck identities must match.
`mega_lopunny_ex_001`, `mega_lopunny_ex_mega_froslass_ex_001`, and
`mega_lopunny_ex_mega_froslass_ex_002` instead all load the final 0023 focal
Mega Lopunny decoder/value weights, then are saved as independent 0024
model-only branches bearing their own target deck IDs and SHA-256 values.

The focal Marnie branch inherits its own matching V8 decoder/value branch.
The Rmy Ogerpon branch inherits its independent V8 Foundation-default branch.
Optimizers are always newly initialized for 0024.

## Primary And Secondary Training

Marnie is the sole focal player and receives every scheduled focal trajectory.
`rmy_teal_mask_ogerpon_001` is the secondary Live policy. During training only,
its Live-opponent matchup has integer weight 10; every ordinary non-focal Live
opponent has weight 1. Frozen rollout and all Frozen/Live greedy evaluations
remain unweighted, complete, and balanced by seat.

This weighting increases Ogerpon actor data and updates without mixing its
trajectory into Marnie's PPO batch. Each update records the secondary Live
weight, scheduled episodes, actor decisions, and Ogerpon's own PPO metrics.

## Operations

V1 is a continuous formal run with W&B online under
`dragon_bra/pokemon-tcg-policy-learning`, TensorBoard, atomic JSONL metrics,
bounded model-only checkpoint retention, 100 GiB launch free-space guard, 80
GiB clean-stop guard, and a 10 GiB version cap. The training child and local
process monitor run continuously in one foreground terminal session. The first
24 GPU-hours are treated as the minimum observation horizon, not an automatic
monitor shutdown.
The monitor reports process exits, failed status, stalled metrics/checkpoints,
GPU health, and engine errors fail-closed.
