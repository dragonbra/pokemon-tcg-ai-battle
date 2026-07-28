# 0018 Alakazam Terminal RL

Status: **framework and rollout-throughput preflight implemented; formal PPO not started**

Date: 2026-07-29
Target: user-designed Alakazam + Dudunsparce deck, initialized from 0016 R15 epoch 10

## 1. Decision summary

0018 tests whether the strongest 0016 Alakazam BC representation can be improved by conservative
deck-specific PPO. The environment reward is only the official-engine terminal result:

| Result | Reward |
|---|---:|
| Win | `+1` |
| Loss | `-1` |
| Draw | `0` |

`gamma=1.0`. Damage, Prize tempo, evolution, card use and legality do not add reward. Invalid engine
episodes are discarded rather than labeled as draws. The initial actor update boundary is the
autoregressive decoder only; the R15 representation remains frozen and the new value head trains.

## 2. Evidence boundary

- **Official rules:** define legal turn order, evolution timing, once-per-turn resources, attack as
  a turn-ending commitment, and terminal victory. They do not prescribe PPO or an optimal policy.
- **Runtime facts:** every rollout is a complete real game in the unmodified official engine. The
  engine's terminal result is the sole reward source and its legal options define the action mask.
- **Project hypotheses:** R15's BC representation contains useful general card/board knowledge;
  decoder-only PPO may improve Alakazam choices without erasing that representation. These are
  hypotheses to evaluate, not rule facts.

Rules evidence: [Pokemon TCG rules research](../../docs/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md).
Exact card behavior remains governed by current official card data and engine runtime.

## 3. Frozen source and target deck

| Field | Value |
|---|---|
| Source project | `0016_alakazam_multideck_bc` |
| Source version | `V2_win_multideck_r15_gradual_option` |
| Checkpoint | `epoch-0010-9db3b4d37a4d4f5c.pt` |
| Checkpoint SHA-256 | `9db3b4d37a4d4f5cc483df1ac78ad4c5d39fed320c5325ebaa5fa61fd1114ebf` |
| Model family | source-conditioned R15 |
| Deployment source ID | `0` |
| Actor parameters | `17,601,282` |
| Source validation exact action | `81.8755%` |
| Source legal action | `100%` |
| Target deck SHA-256 | `267ce842b45f960afef843e68f3aad86573bf7d19329ea23e7469ec8265c017b` |

The exact 60-card target is the original user-designed 0016 deck: 4 Alakazam, 4 Abra, 4 Kadabra,
3 Dunsparce, 3 Dudunsparce, 1 Fezandipiti ex, 1 Shaymin; 3 Rare Candy, 4 Buddy-Buddy Poffin,
4 Battle Cage, 4 Poke Pad, 4 Hilda, 4 Dawn, 3 Boss's Orders, 2 Enhanced Hammer, and one each of
Night Stretcher, Wondrous Patch, Sacred Ash, Lana's Aid and Xerosic's Machinations; 1 Enriching
Energy, 4 Telepath Psychic Energy and 2 Basic Psychic Energy. The mirror-Xerosic ablation is not
the 0018 target.

0018 is self-contained under `train/0018_alakazam_terminal_rl/`. The 0016 checkpoint path is
immutable provenance; no executable code is imported from 0016 or 0017.

## 4. Actor input and network contract

RL preserves the 0016 R15 actor-visible schema and adds no privileged information:

| Tensor group | Shape before batch padding | Meaning |
|---|---|---|
| `global_cat`, `global_num` | `[4]`, `[12]` | selection context, turn budgets and global counts |
| `entity_cat`, `entity_num`, `entity_mask` | `[E,7]`, `[E,5]`, `[E]` | public board and zone entities, `E <= 192` |
| `option_cat`, `option_mask` | `[O,12]`, `[O]` | current official-engine legal options, `O <= 128` |
| `min_count`, `max_count` | scalar each | legal action sequence bounds |
| `zone_inventory_num` | `[2,16]` | visible resource totals by player |
| `registered_*`, `ledger_*` | `[R]`, `[R,4]`, `[R,15]` | exact own deck and causal resource ledger |
| `event_cat`, `event_num`, `event_mask` | `[H,8]`, `[H,4]`, `[H]` | causal public event memory |
| opponent-hand knowledge | `[K]` plus scalar | revealed cards and unknown remainder |
| `source_id` | scalar | fixed 0016 expert/source condition `0` |

The actor width is 320 with the inherited entity, event, resource, goal, scenario and ScaleGate
readers. One zero-initialized value head adds `103,681` parameters:

```text
official observation
  -> causal R15 feature compiler
  -> frozen entity/resource/event/goal/scenario encoder
  -> conditioned state [B,320] and options [B,O,320]
       |                                  |
       |                                  +-> autoregressive masked pointer decoder
       +-> LayerNorm -> MLP -> tanh -> value [B] in [-1,1]
```

Decoder-only PPO plus the value head exposes `1,130,883` trainable parameters. An action is an
ordered sequence of distinct option indices. STOP is available only after `minCount`; decoding
ends at STOP, `maxCount`, or 16 tokens. PPO replays the exact sampled prefix and STOP decision.

## 5. Rollout and dynamic batching

```text
16 isolated CPU official-engine workers
        | asynchronous observations
        v
2 ms dynamic coalescing window (never waits for a fixed full batch)
        |
        v
one real batched candidate forward on the RTX 5080
        |
        +-> actions returned to every requesting engine
        v
complete terminal Episodes held in bounded RAM -> value/PPO backward
```

The first ready request opens a 2 ms window. Any requests arriving in that window join the tensor
batch; timeout or the worker count immediately closes the batch. A batch may contain 1 through 16
requests. Raw replay, observation and optimizer state are not persisted.

Compatible BC opponent adapters can keep 12 current opponent models resident on GPU. They support
source-causal, full-action and ID-only model families, preserve per-game causal state, and perform
one tensor forward per same-policy request group. This is an explicit `--resident-opponents`
experimental route, not the single-GPU default.

## 6. Throughput decision

All measurements below used real official-engine games, greedy policies, identical seeds within
each A/B, and no external Arena process. JSON evidence is under the ignored local directory
`.tmp/evaluation/0018_gpu_opponent_benchmark/`.

| Workload | Workers | Routing | Dynamic wait | Episodes/s | Mean relevant batch |
|---|---:|---|---:|---:|---:|
| Cynthia, 64 games | 8 | opponent CPU | 0 ms | `0.613` | candidate not recorded |
| Cynthia, 64 games | 8 | opponent GPU | 0 ms | `0.465` | opponent `2.67` |
| Cynthia, 64 games | 8 | opponent GPU | 2 ms | `0.557` | opponent `3.65` |
| Cynthia, 64 games | 16 | opponent CPU | 0 ms | `0.951` | candidate `6.94` |
| Cynthia, 64 games | 16 | opponent CPU | 2 ms | **`1.114`** | candidate `10.13` |
| Cynthia, 64 games | 16 | opponent GPU | 2 ms | `0.835` | opponent `6.99` |
| 12-model pool, 96 games | 16 | opponents CPU | 0 ms | `0.912` | candidate `7.42` |
| 12-model pool, 96 games | 16 | opponents GPU | 2 ms | `0.537` | opponent `1.17` |

Different model weights cannot share one tensor forward. In the mixed pool, 12 opponent identities
fragmented 7,507 requests into 6,443 forwards; 5,386 batches had size one. On one GPU, opponent
forwards also compete with the trainable candidate. Therefore the measured default is CPU opponent
inference in isolated workers plus GPU candidate dynamic batching. Resident GPU opponents remain
useful for a second GPU, repeated games against one policy, or a future concurrent service design.

## 7. Learning and observability contract

Initial value calibration freezes the actor and fits terminal Monte Carlo targets with equal total
weight per Episode. Initial PPO uses 256 complete Episodes/update, `gamma=1`, GAE lambda `0.97`,
4 PPO epochs, decision minibatch 1024, actor LR `3e-6`, value LR `1e-4`, ratio clip `0.10`, value
coefficient `0.5`, entropy coefficient `0.01`, gradient norm `0.5`, and a fixed BC reference.

`training_metrics.jsonl` is canonical, then TensorBoard, then private W&B project
`dragon_bra/pokemon-tcg-policy-learning`. W&B run names are
`0018_alakazam_terminal_rl-<Vn_tag>`. Required curves include:

- rolling 100/500/2000 W/L/D, Wilson intervals, seat and per-opponent rates;
- Episodes/s, decisions/s, rollout/update wall time and invalid Episode count;
- candidate/opponent request count, batch count, mean/max batch and inference seconds;
- policy/value loss, entropy, behavior/reference KL, clip fraction and explained variance;
- parameter counts, GPU memory/utilization and both filesystem free-space guards.

Training rolling win rate is an optimization diagnostic, not final proof. Frozen greedy comparison
uses every enabled Arena opponent, balanced seats, identical opponent package snapshot and the
official engine. 0018 intentionally has no six-opponent holdout split.

## 8. Storage, versions and current stage

- Versions are immutable `rl_runs/0018_alakazam_terminal_rl/versions/V<n>_<tag>/` directories.
- Checkpoints contain model weights and provenance only; no optimizer/scheduler/RNG resume state.
- At most eight model-only checkpoints are retained per version.
- Rollout buffers are released after the update; raw traces and replays are off by default.
- Warn below 80 GiB free and stop before either monitored filesystem falls below 50 GiB or one
  version reaches 20 GiB.

Current stage: implementation, adapter compatibility, official-engine smoke and throughput A/B are
complete. The next gate is an end-to-end CUDA training smoke using the measured routing. Formal
PPO starts only as a new version after that smoke and an explicit review of this throughput choice.
