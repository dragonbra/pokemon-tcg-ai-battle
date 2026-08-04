# 0032 Mega Lopunny ex / Mega Froslass ex 001 CUDA PPO Design

## Scope and evidence boundary

0032 validates high-throughput BC-to-RL transfer with the exact frozen deck
`mega_lopunny_ex_mega_froslass_ex_001`. The initialization is the 0031 V4
epoch-2 model-only checkpoint, SHA-256
`05f98e2a562f0037713ebd4b96a5a324151c91b187d7edaa879cc1c024134a51`.
The transfer copies 11,052,162 of 14,579,843 parameters (75.8044%) through
explicit mappings. It is initialization, not input or behavior parity.

V1 remains the historical POD/CUDA adapter validation. V2 freezes the focal
selection and preflight evidence. Formal PPO is
`V3_mega_lopunny_froslass_001_cuda_ppo`.

Official game legality and card behavior remain the responsibility of the
unmodified official engine and the audited CUDA port. Attack commitment,
evolution timing, and per-turn Supporter, Energy and Retreat limits are engine
facts. Reward, value, setup quality and long-horizon resource use are learned
policy concerns. Source/team identity is provenance only and never enters the
actor forward pass.

## Actor-visible tensor contract

| Field | Shape | Type | Role |
|---|---|---|---|
| `global_cat` | `[B, 8]` | int64 | categorical game/select/turn state |
| `global_num` | `[B, 16]` | float32 | normalized global resources |
| `entity_cat` | `[B, 128, 6]` | int64 | entity identity, kind, zone, position and flags |
| `entity_num` | `[B, 128, 10]` | float32 | HP, damage, counts and numeric state |
| `entity_parent` | `[B, 128]` | int64 | attachment/evolution parent, `-1` for none |
| `entity_mask` | `[B, 128]` | bool | valid entities |
| `option_cat` | `[B, 128, 12]` | int64 | legal option identity, family, source and target |
| `option_num` | `[B, 128, 4]` | float32 | option numeric attributes |
| `option_equiv` | `[B, 128]` | int64 | dense semantic-equivalence groups |
| `option_mask` | `[B, 128]` | bool | valid legal options |
| `min_count`, `max_count` | `[B]` | int64 | ordered selection bounds |

Ordered actions are `[B, 64]` int64 option indices padded with `-1`, plus
`[B]` lengths. Selection is unique and bounded by `min_count`/`max_count`;
STOP is legal only after the minimum and before the maximum. The official
codec uses 128 option slots because an observed official state exposed 81.

## Network and trainable boundary

`PodNativeActorCritic` has 14,579,843 parameters with `d_model=320`, eight
attention heads, four state Transformer layers, two option cross-attention
layers, and a copied GRU pointer decoder. A two-layer value head reads the
global state summary.

V3 freezes the encoder and trains only the focal action decoder and value head.
The optimizer is fresh AdamW with actor LR `1e-5`, value LR `1e-4`, two PPO
epochs, minibatch 512, clip ratio 0.10, entropy coefficient 0.01, reference-KL
coefficient 0.02, target behavior KL 0.02, and gradient clip 0.5. Encoder
outputs are detached and reused within each minibatch.

## Rollout, reward and PPO contract

The official result code maps from the focal player perspective as
`1 -> +1`, `2 -> -1`, and `3 -> 0`. Only complete Episodes are admitted.
Incomplete trajectories at the fixed collection boundary are discarded, so
formal V3 does not silently bootstrap them. Terminal-return GAE uses
`gamma=1.0` and `lambda=0.95`, with equal total weight per completed Episode.

The stochastic behavior log probability is stored at collection and replayed
before the first optimizer step; mismatch above `1e-4` fails closed. Metrics
record `rollout/source_policy_update=k-1` for the batch that creates
`checkpoint/update=k`. Sampled rollout win rates are diagnostics and are not
frozen greedy checkpoint-strength claims.

## Frozen opponents and resident execution

CUDA supports 38 frozen decks. Thirty-five have real frozen decoder assets in
0022 V11; the three absent heads are
`mega_lopunny_ex_001`, `mega_lopunny_ex_mega_froslass_ex_002`, and
`rmy_teal_mask_ogerpon_001`. V3 therefore freezes a 35-opponent snapshot.

Each update runs one real frozen opponent on 152 independent lanes, and the
opponent index rotates round-robin across all 35 updates. This avoids the
small-group GEMM and expanded per-lane GRU bottlenecks while preserving
auditable temporal coverage of the full pool. Observations, actor inference,
engine stepping and fixed rollout buffers remain on CUDA. One bulk host copy
occurs after collection for the immutable learner batch.

## Measured acceptance and storage

The preallocated 152-lane rollout canary reached 2,432.334 decisions/s. The
formal one-update PPO canary reached 2,346.092 rollout decisions/s and
1,390.477 end-to-end decisions/s including the PPO update, with 25 complete
Episodes, 272 focal training decisions, zero illegal/error rows, and behavior
log-prob MAE `1.84e-7`. GPU peak allocation was about 0.94 GB.

Every update retains an atomic model-only checkpoint containing schema,
update, model state and metadata. Optimizer, scheduler, scaler, RNG,
DataLoader position and rollout buffers are excluded. Canonical metrics are
flushed to JSONL, then TensorBoard, then W&B. Because the launch host has no
W&B API key, V3 uses audited offline W&B staging and does not claim online
sync. A foreground process watchdog owns the training child and records
heartbeat, GPU, host memory, swap, disk, stale metrics and fatal errors.

## Current and next stage

V3 is the formal CUDA PPO training stage. Its first bounded run covers one
complete 35-opponent rotation and is throughput/learning-path evidence. Policy
strength must later be established by balanced, fixed-seed, official-engine
greedy evaluation under a new immutable evaluation version; sampled training
rollouts alone cannot promote a candidate package.
