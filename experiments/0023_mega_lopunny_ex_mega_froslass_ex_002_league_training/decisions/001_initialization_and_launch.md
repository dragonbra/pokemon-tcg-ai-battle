# 001 Initialization and continuous launch contract

Date: 2026-08-01

## Decision

The only focal deck is `mega_lopunny_ex_mega_froslass_ex_002`, exact deck SHA-256
`7fb1536b191c0e0ff5f8819c674ba784e9e0b77927b58b3faea423db66b90e8b`.
The formal League run has no time deadline. Twenty GPU hours is an operations milestone,
not a stop condition. Only an explicit user stop, update-failing error, CUDA failure, or
SSD low-water/version-cap guard may stop it.

0022 V11 published a complete update 81 for all 48 decks in its catalog. Those decks
inherit their own update-81 decoder/value state. The two decks absent from V11,
`mega_lopunny_ex_001` and the focal `_002`, initialize from the immutable 0019 Epoch 13
Foundation. A catalog member may not silently fall back to Foundation if its update-81
asset is missing or has a mismatched exact-deck hash.

The accepted 0022 Dragapult checkpoint remains update 75 for historical strength
provenance. It is distinct from the requested Live continuation rule, which uses each
deck's final complete update-81 state.

Each deck starts with a fresh optimizer and a fixed reference copied from its exact 0023
update-0 state. The reference is therefore update 81 for inherited decks and Foundation
for new decks. Behavior snapshots remain per update and are not confused with that fixed
reference.

## Preflight evidence

- Exact resolver: 50 decks, 48 inherited update-81 assets, 2 explicit Foundation assets.
- Focal official-engine CUDA smoke: 4/4 finished, 0 errors, 93,619,712 peak allocated bytes.
- Focal PPO canary: 4/4 finished, 287 actor decisions, finite PPO metrics, frozen Encoder
  hash unchanged, model-only save/load greedy parity passed, 484,529,152 peak bytes.
- Full 50-head CUDA smoke: four Frozen/Live official-engine games, 4/4 finished, 0 errors,
  319,007,744 peak allocated bytes.
- Storage at preflight: about 807 GiB free; launch/runtime/version guards remain 100/80/10 GiB.

## Checkpoint retention

Per deck, keep the latest two checkpoints and the most recent four evaluation-interval
snapshots. Do not retain every fifth update forever. No optimizer, scheduler, RNG,
rollout buffer, trace, replay, or duplicate full Encoder is serialized.

## Strength evidence

Sampled rollout and short Frozen/Live probes are diagnostics. Candidate promotion requires
fixed-seed, balanced-seat official-engine evaluation against the immutable Frozen Arena.

## Launch history

`V1_mega_lopunny_ex_mega_froslass_ex_002_continuous_league` failed before training while
serializing initialization provenance. Decoder identity validation and the first inherited
checkpoint load had succeeded, but a repository-relative source path was compared directly
with an absolute repository root. V1 produced no rollout, PPO update, W&B run, TensorBoard
event, or Live checkpoint and remains preserved as a failed version.

The fix anchors relative source paths to the repository root before normalization and has a
dedicated regression test. Formal continuous training therefore starts in strictly increasing
`V2_mega_lopunny_ex_mega_froslass_ex_002_continuous_league`.
