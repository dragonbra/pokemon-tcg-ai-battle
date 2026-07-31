# 0023 Mega Lopunny ex / Mega Froslass ex 002 League Training

Updated: 2026-08-01

## 1. Objective

0023 is a focal-primary continuation of 0022. Its only focal deck is
`mega_lopunny_ex_mega_froslass_ex_002`, exact deck SHA-256
`7fb1536b191c0e0ff5f8819c674ba784e9e0b77927b58b3faea423db66b90e8b`.
The run remains active until the user explicitly stops it, or a fail-closed CUDA,
official-engine, nonfinite-training, or SSD guard stops it. Twenty GPU hours is a
durability milestone, not a deadline.

The strength objective is improved fixed-seed, balanced-seat performance against
the immutable 50-deck Frozen Arena. Sampled rollout and Live-pool results are
training diagnostics, not checkpoint acceptance evidence.

## 2. Evidence Boundary

General game rules come from the official rulebook. Exact card effects and legal
actions come from the current official card data and unmodified official engine
runtime. The policy chooses only among legal engine options. Terminal outcome is
the only formal reward: win `+1`, loss `-1`, draw `0`, with `gamma=1`.

0023 does not modify `engine/source/`, does not claim a CUDA engine, and does not
claim global optimality. Encoder batching changes policy inference throughput but
not state transitions or game rules.

## 3. Foundation And Model Contract

The immutable Foundation is 0019 Epoch 13:

- asset ID: `0019-0730-epoch13`
- global step: `337194`
- full model parameters: `17,756,162`
- checkpoint SHA-256:
  `da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb`
- deployment `source_id`: always `0`

The self-contained model, feature compiler, causal ledger, action codec, ontology,
and schema live under
`train/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/`. Runtime code
does not import another numbered training project.

The existing observation and action tensor contract is unchanged from the frozen
0019 implementation:

```text
structured observation + legal options
  -> feature compiler
  -> shared frozen encoder, d_model=320
  -> encoded entities/options
  -> routed deck-local autoregressive decoder
  -> ordered full-action token sequence + STOP

encoded summary, shape [B, 320]
  -> deck-local LayerNorm/MLP/Tanh value head
  -> scalar V(s), shape [B]
```

Only these actor components are trainable per deck:

```text
pointer_key
pointer_query
option_bias
decoder_init
decoder
stop
value_head
```

All encoder, embedding, feature, ontology, persona residual, and action-contract
parameters remain frozen. The 50 policies share one resident encoder and own 50
isolated decoder/value branches.

## 4. Exact Initialization Contract

0022 V11 stopped after the complete publication of update 81. Its catalog contains
48 decks. 0023 resolves initialization by exact deck ID and exact deck SHA:

- all 48 prior catalog members load their own
  `V11_multidecoder_league_20h/checkpoint/live/<deck>/update-000081.pt`;
- `mega_lopunny_ex_001` was added later and starts from the 0019 Foundation;
- focal `mega_lopunny_ex_mega_froslass_ex_002` was added later and starts from the
  0019 Foundation;
- a prior catalog member with a missing update-81 checkpoint is a launch error;
- a reused deck ID with a changed exact-deck SHA is a launch error.

The accepted 0022 Dragapult strength checkpoint remains update 75. This is
provenance for the 0022 result, while Live continuation uses each deck's final
complete update-81 state as requested. Every resolved source path, update, model
SHA, deck SHA, and Foundation SHA is written to
`artifact/initialization_audit.json`.

The 0023 initializer copies each resolved decoder/value into its own immutable
update-0 model-only checkpoint. Loading that checkpoint must reproduce the same
greedy action as the source checkpoint.

## 5. Reference KL And Optimizers

Each deck receives its own new AdamW optimizer. Optimizer state is never inherited
or serialized. Each deck's fixed reference policy is a frozen copy of its exact
0023 update-0 decoder/value state:

- inherited deck reference: that deck's 0022 update-81 state;
- new deck reference: 0019 Foundation decoder plus newly initialized value head.

The reference is not the behavior snapshot and is not reset every update. PPO also
constructs a per-update frozen behavior snapshot for ratio/KL calculations. These
three identities remain distinct:

```text
reference policy = fixed 0023 initialization
behavior policy  = checkpoint that collected the current rollout
updated policy   = checkpoint produced from that rollout
```

Default PPO configuration:

```text
epochs=4, minibatch decisions=1024
actor lr=1e-5, value lr=1e-4
clip ratio=0.10, target behavior KL=0.02
reference KL coefficient=0.02
entropy coefficient=0.01, value coefficient=0.5
GAE lambda=0.95, gamma=1.0, max grad norm=0.5
```

## 6. Rollout And Update Contract

One update collects 512 complete official-engine games, balanced 256/256 by focal
seat. Matchups rotate deterministically over both views:

- Frozen: immutable 0019 Foundation policy routed by exact deck;
- Live: current deck-local decoder/value policy.

Focal self-play is excluded from on-policy training because both reward signs would
share one policy ID inside one episode. It remains available in explicit evaluation.

Both players' stochastic Live decisions retain:

```text
policy_deck_id
source_policy_update
actor-relative reward sign
action sequence and STOP
behavior log probability
behavior value
```

Each of the 50 trainers filters only decisions whose `policy_deck_id` equals its
deck. Opponent actions never enter focal loss under the focal identity. A batch that
does not contain real trajectory for every Live deck fails closed instead of
pretending the missing deck updated.

Every fifth update runs a balanced two-seat greedy diagnostic against all 50 Frozen
and all 50 Live identities, plus Alakazam and Marnie probe matchups. Formal checkpoint
selection uses the larger fixed Frozen Arena contract, not this short diagnostic.

## 7. Metrics

Canonical `training_metrics.jsonl` records:

- `trainer/update`, `rollout/source_policy_update`, `checkpoint/update`;
- focal sampled W/L/D and win rate under `rollout/focal/*`;
- full League sampled W/L/D under `rollout/league/*`;
- per-deck decisions and PPO loss/value/entropy/KL/gradient diagnostics;
- `eval/frozen/*`, `eval/live/*`, and fixed probe metrics at evaluation updates;
- shared batching throughput, engine selections/s, episodes/s, errors, unfinished;
- CUDA allocated/reserved/peak bytes and elapsed hours;
- free disk, version size, and storage guard state.

Metrics flush locally first, then TensorBoard, then W&B online to private project
`dragon_bra/pokemon-tcg-policy-learning`. W&B failure cannot roll back local metrics
or checkpoint publication, but must be visible in version status.

## 8. Checkpoints And SSD Safety

All checkpoints use `0023_league_decoder_model_only_v1` and contain only decoder and
value tensors plus identity metadata. The loader may read the explicitly compatible
0022 model-only schema for audited inheritance; new files always use the 0023 schema.

Per deck, retention is bounded to:

- latest 2 checkpoints;
- most recent 4 evaluation-interval snapshots;
- current checkpoint while publication is in progress.

No optimizer, scheduler, scaler, RNG, DataLoader state, rollout buffer, trace, replay,
or full encoder copy is saved. Launch requires at least 100 GiB free; runtime stops
at the configured 80 GiB low-water mark or 10 GiB version cap. These are safety
stops, not successful training completion.

## 9. Version And Acceptance Boundary

Formal run roots are:

```text
rl_runs/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/versions/<version>/artifact/
rl_runs/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/versions/<version>/checkpoint/
rl_runs/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/versions/<version>/tensorboard/
rl_runs/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/versions/<version>/wandb/
```

`latest` means only the newest published model. `champion` requires a fixed-seed,
balanced-seat official-engine Frozen Arena report. Live-pool improvement, sampled
rollout peaks, or lower PPO loss cannot promote a checkpoint by themselves.
