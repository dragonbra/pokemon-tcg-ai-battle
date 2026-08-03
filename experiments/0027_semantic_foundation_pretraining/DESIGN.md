# 0027 Winner-only Semantic Foundation BC

Status: **trained, exported, and locally evaluated**

## Objective

0027 tests whether the complete canonical semantic actor from 0025 can learn a
broad winner policy from recent Top battle records without a pretrained
checkpoint. It preserves exact-deck and causal semantic inputs while keeping
source/team identity outside actor forward.

## Actor contract

The model rejects missing or extra actor fields. The exact 22 accepted features
are:

```text
card_cat, card_num, card_parent,
event_cat, event_num,
global_cat, global_num,
max_count, min_count,
option_cat, option_effect_id, option_effect_parent, option_effect_role,
option_num, option_skill_id, option_skill_parent, option_skill_role,
option_source, option_state, option_target,
resource_cat, resource_num
```

Source and team identity are retained only in data provenance. They are not
embedded, routed to an adapter, or otherwise allowed to change logits.

## Model data flow

```text
global + card instances + exact-deck resource ledger + recent events
    -> typed projections and canonical prototype lookup
    -> 4-layer state Transformer
    -> full state memory and pooled state summary

legal options + source/target links + skill/effect relations
    -> typed option projection
    -> 3-layer Transformer decoder cross-attending state memory
    -> semantic option tokens

state summary + option tokens
    -> autoregressive GRU pointer
    -> unique ordered legal-option indices + STOP
```

Production configuration:

| Field | Value |
|---|---:|
| Model width | 320 |
| Attention heads | 8 |
| State layers | 4 |
| Option layers | 3 |
| FFN multiplier | 3 |
| Dropout | 0.10 |
| Maximum legal options | 128 |
| Maximum action steps | 64 |
| Trainable parameters | 21,837,082 |

Card, Attack, Skill, and Effect capacities are 2,048 / 2,048 / 512 / 4,096.
Static semantics come from the committed canonical prototype snapshot.

## Data contract

- inclusive date range: 2026-07-15 through 2026-08-01;
- strict winner view only;
- complete Episode-level train/validation split inside each day;
- four independently materialized and mountable date partitions;
- gzip canonical shards with finite numeric and relation-index validation;
- no cross-partition notebook dependency.

Training consumed every expected decision exactly once:

| Partition | Train decisions |
|---|---:|
| 2026-07-15..19 | 1,701,899 |
| 2026-07-20..24 | 1,759,203 |
| 2026-07-25..28 | 1,463,437 |
| 2026-07-29..08-01 | 1,477,040 |
| Validation, all parts | 711,086 |

## Training contract

- random initialization (`initialized_from_checkpoint = null`);
- two Nvidia T4 GPUs with `torch.nn.DataParallel`;
- one rolling epoch for each mounted partition, in date order;
- ordered legal-option pointer plus STOP behavior cloning objective;
- model-only checkpoints; no optimizer, scheduler, scaler, RNG, replay, or
  DataLoader resume state;
- no W&B by explicit user request for this run; `training_report.json` is the
  canonical metric record;
- no skipped AMP updates and no non-finite model tensors.

## Outcome

| Metric | Final value |
|---|---:|
| Train loss, final partition | 0.385987 |
| Train token accuracy, final partition | 83.5906% |
| Validation loss | 0.432314 |
| Validation token accuracy | 86.2273% |
| Teacher-forced exact action | 74.1182% |
| Runtime | 10,580.67 s |

The retained model is `artifacts/final_bc_v3/best_model.pt`, SHA-256
`274407890d17b70a5a331cac1855a7a2438946e4c5a2989a99345f1ab4af5dfc`.
It contains model weights and contract metadata only.

## Local official-runtime evidence

The same weights were exported without modification and tested with two exact
decks against six existing BC opponents. Each matchup used 20 games and balanced
candidate player index 10/10.

| Candidate deck | Games | W-L | Win rate | Errors | Corrected actions | Fallbacks |
|---|---:|---:|---:|---:|---:|---:|
| Dragapult | 120 | 59-61 | 49.17% | 0 | 0 | 0 |
| Raging Bolt | 120 | 27-93 | 22.50% | 0 | 0 | 0 |

The engine randomized actual first player, producing 70 first / 50 second games
for each candidate. These unseeded local samples establish runtime legality and a
preliminary strength signal; they do not replace a fixed-seed formal evaluation
or justify automatic opponent-pool promotion.

## Current boundary

The model is a research foundation checkpoint. It has not been promoted into the
formal arena opponent catalog. A later RL or formal evaluation version must use
an immutable opponent snapshot, official runtime, explicit seed/turn-order
contract, and a separately versioned result.
