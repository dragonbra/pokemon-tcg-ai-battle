# 0043 Benchmark V1 Protocol

Status: **PUBLISHED — 2026-08-14**

## 1. Purpose

Benchmark V1 is the standard 0043 checkpoint-strength benchmark after Champion-G2 promotion. It measures one exact focal deck/policy identity against a stable Champion-G2 league distribution. It is an official-engine frozen greedy evaluation, not PPO rollout telemetry and not an automatic Promote decision.

## 2. Opponent identity

Every game binds the opponent as the pair `(deck_i, Champion-G2)`:

- `deck_i` is one immutable exact deck ID from `001`–`067` plus its registered 60-card content SHA-256;
- the policy is always the complete immutable `Champion-G2` effective identity;
- policy and deck are resolved before CUDA lane routing;
- one resident G2 model is used for the run, while the exact 29-way own-deck row and deck static fields remain lane-bound;
- no Policy-0809, Champion-G1, hybrid policy, per-deck checkpoint reload, or cross-policy mutable storage sharing is permitted.

## 3. Meta-balanced CUDA-2048 schedule

The canonical schedule contains exactly 2,048 games. `deck_own_archetype_mapping_v2.json` is the class-membership authority.

1. Consider all 29 `own_archetypes_v2` reporting classes.
2. Class 14 `Other` contains no registered exact deck and receives no fabricated games.
3. Allocate the 2,048 games as evenly as possible over the 28 non-empty classes: each receives 73 or 74 games.
4. Within each class, allocate its games as evenly as possible over member exact decks: member counts differ by at most one.
5. Seeded rotations decide which classes/decks receive remainders; a seeded shuffle removes block ordering.

The report must still display all 29 classes. An empty class is reported as `n=0 / no registered deck`, never as a measured 0% win rate.

## 4. Determinism and seat semantics

- Contract ID: `0043_benchmark_v1_meta_balanced_g2_cuda2048_v1`.
- Master seed: `341512902`.
- Schedule seeds bind contract ID, focal deployment identity, Champion-G2 effective identity, exact opponent deck hash, schedule slot, and namespace.
- Engine, search, and policy seeds are non-zero and unique within a run.
- A seeded coin identifies the winning Agent; that Agent processes official context 41 and chooses first or second. The harness does not assign a seat.
- Two focal deployment identities receive different schedules; the same identity reproduces the same schedule byte-for-byte.

## 5. Deployment and runtime contract

- CUDA Engine 2.0, greedy selection, 2,048 official games.
- Focal checkpoint evidence uses `kaggle_fp16_storage_fp32_runtime_v1` or an already admitted immutable policy with identical FP16-storage/FP32-runtime semantics.
- PPO/master/checkpoint storage remains FP32 and is never mutated by evaluation conversion.
- Champion-G2 is strict-loaded from its immutable portable FP16 artifact into FP32 runtime.
- All 2,048 games must be terminal with zero error, unfinished game, routing failure, identity mismatch, parameter-storage alias, or feature D2H fallback.

## 6. Published report tree

```text
docs/evaluation/combat_mat/benchmark_v1/
├── index.html
└── <focal-policy-and-deck-slug>/
    ├── index.html
    └── manifest.json
```

Each focal report introduces the focal policy/checkpoint, source update and hashes; shows overall and actual-first/actual-second strength; and provides:

- all 29 Meta Archetypes with wins/losses/draws, win rate, game count, member deck IDs and representative card thumbnails;
- all 67 exact decks with wins/losses/draws, win rate, game count and representative card thumbnails;
- focal, opponent, schedule, CUDA engine and routing identities;
- explicit no-sample semantics wherever applicable.

The top-level `benchmark_v1/index.html` indexes all published focal reports. Existing Policy-0809 and Promote diagnostic directories remain independent historical evidence.
