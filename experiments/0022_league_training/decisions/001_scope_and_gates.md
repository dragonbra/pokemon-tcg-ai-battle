# 0022 Decision 001: Scope And Gates

**Date:** 2026-07-30

**Status:** accepted design boundary

**Decision:** 0022 first validates resident opponent throughput and Decoder-only League Training; it does not select or freeze the final 16-deck catalog yet.

## Context

Current RL profiling shows that opponent package loading, CPU opponent inference, observation/IPC and feature preparation are larger costs than the direct official `battle_select` call in the measured topology. A shared Frozen Encoder with deck-specific Decoder heads may allow 16 Frozen and 16 Live policies to remain resident and to be evaluated in batches.

The project needs evidence that this changes real end-to-end RL iteration speed before investing in a broader Arena evolution system.

## Decisions

1. The first hard gate is a three-arm throughput benchmark: tuned current CPU opponent path, resident CPU service, and resident GPU service.
2. The benchmark must preserve the official engine, completed-game contract, exact policy/deck inputs, balanced seat schedule, and error accounting.
3. Quality is anchored by a Frozen pool of complete policies, not by a Frozen Encoder alone.
4. Live policies own separate Decoder/Value parameters and may update from their own actor trajectories. Frozen opponents do not receive gradients.
5. A focal iteration may use 16 Frozen matchups and 16 Live matchups, 10 games per matchup, for 320 games, but this is a sampling schedule rather than a statistical guarantee.
6. The final 16-deck catalog remains a separate auditable asset. Candidate decks must be observed in the online environment, frequent enough to have BC evidence, and sufficiently covered by the learned card/construct semantics.
7. Live policies do not automatically become Frozen opponents. Promotion requires a new version, package validation, official-engine evaluation, and an immutable pool snapshot.

## Rejected shortcuts

- Calling policy-only GPU throughput an RL speedup;
- treating Live-only win rate as stable policy strength;
- mixing old and new policy versions in one PPO importance-ratio contract;
- selecting the 16 decks only by archetype names without exact deck identity and BC coverage evidence;
- introducing new card embeddings or Encoder updates in the first League version.

## Consequence

The first implementation can be evaluated without committing to the final catalog. If the throughput gate fails, the League Training experiment is stopped before expensive self-play. If the gate passes but Frozen-anchor win rate does not improve, the bottleneck is likely representation, exploration, value calibration or decoder capacity rather than opponent inference.
