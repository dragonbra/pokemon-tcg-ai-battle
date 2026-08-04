# 0032 Mega Lopunny ex / Mega Froslass ex 001 CUDA PPO Design

## Scope and evidence boundary

0032 validates high-throughput BC-to-RL transfer with the exact frozen deck
`mega_lopunny_ex_mega_froslass_ex_001`. The initialization is the 0031 V4
epoch-2 model-only checkpoint, SHA-256
`05f98e2a562f0037713ebd4b96a5a324151c91b187d7edaa879cc1c024134a51`.
The transfer copies 11,052,162 of 14,579,843 parameters (75.8044%) through
explicit mappings. It is initialization, not input or behavior parity.

V1 remains the historical POD/CUDA adapter validation. V2 freezes the focal
selection and preflight evidence. V3 was stopped after update 4 when the run
budget and opponent contract changed. V4 was configured for 200 updates as
`V4_mega_lopunny_froslass_001_cuda_ppo_200u` and intentionally stopped after
81 complete updates once the user accepted the RL-throughput result.

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

V4 freezes the encoder and trains only the focal action decoder and value head.
The optimizer is fresh AdamW with actor LR `1e-5`, value LR `1e-4`, two PPO
epochs, minibatch 512, clip ratio 0.10, entropy coefficient 0.01, reference-KL
coefficient 0.02, target behavior KL 0.02, and gradient clip 0.5. Encoder
outputs are detached and reused within each minibatch.

## Rollout, reward and PPO contract

The official result code maps from the focal player perspective as
`1 -> +1`, `2 -> -1`, and `3 -> 0`. Only complete Episodes are admitted.
Incomplete trajectories at the fixed collection boundary are discarded, so
formal V4 does not silently bootstrap them. Terminal-return GAE uses
`gamma=1.0` and `lambda=0.95`, with equal total weight per completed Episode.

The stochastic behavior log probability is stored at collection and replayed
before the first optimizer step; mismatch above `1e-4` fails closed. Metrics
record `rollout/source_policy_update=k-1` for the batch that creates
`checkpoint/update=k`. Sampled rollout win rates are diagnostics and are not
frozen greedy checkpoint-strength claims.

## Frozen opponents and resident execution

CUDA supports 38 frozen decks. Every opponent uses the same immutable 0019
Epoch 13 Foundation decoder, SHA-256
`da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb`.
All 48 materialized 0022 V11 update-0 heads were verified tensor-identical to
that decoder. Deck identity changes the exact engine deck and observation; it
does not select different policy weights.

Every update includes all 38 decks with four independent lanes each, for 152
lanes total. The shared head restores a single large batched decoder and
removes grouped per-deck GRU routing. Observations, actor inference, engine
stepping, CUDA Graph and fixed rollout buffers remain resident and are reused
across updates. One bulk host copy occurs after collection for the immutable
learner batch.

## Measured acceptance and storage

The selected 256-step shared-0019 canary reached 2,282.910 rollout decisions/s
and 988.121 end-to-end engine decisions/s including PPO. It completed 176
Episodes and admitted 10,704 focal PPO decisions in one update, with all 38
decks represented, zero illegal/error rows, and behavior log-prob MAE
`2.33e-7`. GPU peak allocation was 2.60 GiB and reserved memory was 3.84 GiB.
This has substantially higher effective PPO-sample throughput than the 64- and
128-step windows even though its raw end-to-end engine-decision rate is lower.

Every completed update retains an atomic trainable-head checkpoint containing schema,
update, `action_decoder.*`, `value_head.*` and reconstruction metadata. The
1,130,243 saved FP32 parameters occupy about 4.31 MiB before serialization.
The 81 retained serialized checkpoints occupy 366,768,405 bytes (349.78 MiB).
The frozen
encoder is referenced by its immutable 0031 hash rather than copied. Optimizer,
scheduler, scaler, RNG,
DataLoader position and rollout buffers are excluded. Canonical metrics are
flushed to JSONL, then TensorBoard, then W&B online. An SDK preflight verifies
the existing host credential without recording it. A foreground process
watchdog owns the training child and records
heartbeat, GPU, host memory, swap, disk, stale metrics and fatal errors.

## Completed outcome and next stage

V4 stopped intentionally after update 81 of the planned 200. It produced
3,151,872 engine decisions, 13,100 complete Episodes and 1,027,978 admitted
focal training decisions. Aggregate rollout throughput was 2,332.086 engine
decisions/s, end-to-end throughput was 905.419 engine decisions/s, and
effective PPO-sample throughput was 295.301 focal decisions/s. Peak reserved
GPU memory was 4.041 GiB; engine-error lanes and illegal rows were both zero.

The sampled on-policy rollout record was 6,978 wins, 6,117 losses and 5 draws
(53.286% score rate). This measures the behavior policies that generated each
batch, not the strength of checkpoint 81. The throughput pathway is accepted;
policy strength remains unverified until balanced, fixed-seed, official-engine
greedy evaluation is recorded under a new immutable evaluation version.
