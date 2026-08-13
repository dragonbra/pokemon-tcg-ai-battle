# 0044 Benchmark V2 Protocol

Status: **PUBLISHED CORE-16 REVISION — 2026-08-14**

## Purpose

Benchmark V2 compares deployment-effective focal checkpoints under common random
numbers. It is an official-engine CUDA-2048 greedy benchmark and does not alter PPO,
opponent training, uniform rollout sampling, or promotion automatically.

## Opponent and Core-16 deck distribution

Every opponent is `(exact deck_i, Policy-0809)`. `Policy-0809` means its complete
immutable effective policy, independently materialized before lane routing.

The authoritative Meta domain is exactly Own Archetype V2 IDs `00`–`13`, `17`, and
`27`. All other Meta IDs are excluded from strength aggregation. The 2,048 games are
allocated first across these 16 Meta identities, yielding exactly 128 games per Meta,
then evenly across the exact member decks of each selected Meta. Within one Meta,
member deck counts may differ by at most one. The resulting exact deck domain has 50
registered decks. This Core-16 distribution is the only current Benchmark V2 strength
contract.

## Common-random-number contract

- Contract ID: `0044_benchmark_v2_core16_meta_balanced_policy0809_common_seeds_cuda2048_v2`.
- Master seed: `341512902`.
- Deck allocation, seeded shuffle, engine seed, Search seed, policy seed and
  coin-winner seed are fixed across focal checkpoints.
- Seed derivation includes contract ID, master seed, namespace, immutable
  Policy-0809 identity, exact opponent deck hash and schedule slot.
- Seed derivation explicitly excludes focal policy/checkpoint/deployment identity.
- Focal deployment identity remains mandatory provenance and changes the complete
  schedule manifest hash, but not the common-random job table hash.
- The seeded coin fixes only the toss winner. That Agent still processes official
  context 41 and independently chooses first or second.

## Runtime and evidence gates

The focal candidate uses `kaggle_fp16_storage_fp32_runtime_v1`. Policy-0809 is a
complete immutable policy identity. CUDA Engine 2.0 executes all 2,048 games; 0 error,
0 unfinished, identity audit PASS, routing audit PASS, no focal/opponent storage alias,
no feature D2H fallback, exact selected Meta IDs and exactly 128 games per Meta are
required. Every report records all per-game seeds, toss winner, context-41 choice and
actual seat.

The earlier 28-nonempty-Meta V2 U0 report is retained as retired diagnostic evidence.
It must not be plotted or compared as current Benchmark V2 strength.

## Start and continuation baseline logging

A formal version must expose a Core-16 Benchmark V2 strength point before its first
new PPO rollout. A direct G2 start logs U0. A continuation logs the exact inherited
checkpoint update, including a non-tens boundary such as V6 U4. Subsequent periodic
evaluations remain at durable U10, U20, and later multiples of ten. Only PASS reports
are mirrored to W&B `eval/*`.
