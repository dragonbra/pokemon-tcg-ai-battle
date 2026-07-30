# 0018 Alakazam Terminal RL

Status: **V1 value calibration complete; V2 audited; V3 smoke complete; V4 PPO completed at U100**

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
weight per Episode. V2 used 256 complete Episodes/update, `gamma=1`, GAE lambda `0.97`, four PPO
epochs, decision minibatch 1024, actor LR `3e-6`, value LR `1e-4`, ratio clip `0.10`, value
coefficient `0.5`, entropy coefficient `0.01`, gradient norm `0.5`, and a fixed BC reference.

V4 is a clean model-only warm start from V1 and changes the refinement contract for the strong
Alakazam BC actor: 512 complete Episodes/update, `gamma=1`, GAE lambda `1.0`, two PPO epochs,
decision minibatch 1024, actor LR `1e-5`, value LR `1e-4`, entropy coefficient `0`, reference-KL
coefficient `0.02`, behavior-KL guard `0.01`, ratio clip `0.10`, and gradient norm `0.5`. Lambda 1
passes terminal win/loss credit to early setup decisions without Value bootstrap attenuation. The
larger rollout batch and two epochs reduce the risk of repeatedly fitting one noisy batch. The
higher actor LR is permitted because V2 moved only `0.2599%` in Decoder relative L2 by update 10
and its KL/clip metrics remained far below their guards; it is not allowed to bypass those guards.

`training_metrics.jsonl` is canonical, then TensorBoard, then private W&B project
`dragon_bra/pokemon-tcg-policy-learning`. W&B run names follow
`0018 · alakazam_terminal_rl · V<n>_<tag>`. Required curves include:

- rolling 100/500/2000 W/L/D, Wilson intervals, seat and per-opponent rates;
- Episodes/s, decisions/s, rollout/update wall time and invalid Episode count;
- candidate/opponent request count, batch count, mean/max batch and inference seconds;
- policy/value loss, entropy, behavior/reference KL, clip fraction and explained variance;
- Decoder relative L2 movement per update and versus BC, plus greedy action flip rate on a fixed
  1,024-state canary captured from the first rollout of the current formal version;
- parameter counts, GPU memory/utilization and both filesystem free-space guards.

Training rolling win rate is an optimization diagnostic, not final proof. Frozen greedy comparison
uses every enabled Arena opponent, balanced seats, identical opponent package snapshot and the
official engine. 0018 intentionally has no six-opponent holdout split.

## 8. Storage, versions and current stage

- Versions are immutable `rl_runs/0018_alakazam_terminal_rl/versions/V<n>_<tag>/` directories.
- Checkpoints contain model weights and provenance only; no optimizer/scheduler/RNG resume state.
- Checkpoint cadence and retention are immutable version config. V4 saves every update and retains
  up to 100 model-only checkpoints (about 6.9 GiB at current model size).
- Rollout buffers are released after the update; raw traces and replays are off by default.
- Warn below 80 GiB free and stop before either monitored filesystem falls below 50 GiB or one
  version reaches 20 GiB.

### Candidate package runtime boundary

Candidate `main.py` must support Kaggle's `exec` loader, where `__file__` may be absent. It resolves
the package root from `__file__`, then `/kaggle_simulations/agent`, then the current directory, and
exports both `read_deck_csv()` and `agent()`. Portable inference reconstructs the exact frozen 0016
R15 source contract. A model-capacity overflow or recoverable encoder failure returns a count-valid
legal fallback instead of failing the validation Episode; these guards do not alter normal greedy
inference. Root-level archive membership and an actual no-`__file__` exec load are mandatory
submission gates.

Current stage: implementation, adapter compatibility, official-engine smoke and throughput A/B are
complete. `V1_value_calibration` collected 512 valid official-engine Episodes and 36,282 candidate
decisions, then trained only the value head for four epochs. Its final in-sample value RMSE was
`0.8374` and explained variance was `0.2538`; these are critic-initialization diagnostics, not policy
strength evidence.

`V2_ppo_decoder_terminal` was deliberately stopped after its update-10 checkpoint. Its sampled
rollouts totaled 1,687/2,560 wins (`65.90%`), but the fixed 300-job greedy diagnostic found BC/V1
at 228/299 (`76.25%`), update 1 at 224/300 (`74.67%`), and update 10 at 228/298 (`76.51%`, plus
one draw). V2 therefore showed neither a proven greedy gain nor lasting damage. Reference-KL
surrogate was only `0.000620` at update 10, behavior KL about `1.24e-5`, and Decoder relative L2
movement was `0.0768%` at update 1 and `0.2599%` at update 10. The interrupted update 11 was never
retained and is not part of the policy evidence.

V3 is the completed one-update official-engine smoke for the new contract. V4 completed all 100
updates from V1 rather than V2. It generated 51,179 valid official-engine Episodes and 3,555,884
candidate decisions, with cumulative 35,586 wins, 15,583 losses and 10 draws. At U100 the rolling
100/500/2000 win rates were 73.0% / 69.2% / 71.0%; the corresponding run maxima were 78.0% at U96,
75.4% at U92 and 72.7% at U92. Reference-KL surrogate ended at 0.02694, behavior KL at 4.19e-5,
and actor relative L2 versus the BC reference at 2.012%.

These rolling rates remain sampled optimization diagnostics. V5/V6/V7 froze U39/U63/U92 and ran
each against the same 30-opponent official-engine Arena catalog for 10 balanced-seat games per
opponent. U39 finished 235-65 (`78.33%`, zero errors), U63 finished 219-80 with one step-zero worker
crash (`73.00%` over all 300 scheduled games), and U92 finished 231-69 (`77.00%`, zero errors).
The authoritative reports and project index are under `experiments/0018_alakazam_terminal_rl/evaluation/`.
This independent-randomness 300-game screen selects U39 on observed overall win rate while retaining
U92 as the strongest post-U63 candidate; the four-win U39/U92 gap is not a paired significance claim.

The user explicitly authorized one new Kaggle submission each for U39 and U92. Kaggle accepted both:
U39 ref `55097268` and U92 ref `55097478` reached `SubmissionStatus.COMPLETE` with visible live public
scores, proving archive Validation passed and official evaluation started. Point-in-time receipts are
stored with V5 and V7; competition scores are rolling observations and may continue to change. Neither
submission automatically promotes either package into the fixed local opponent catalog.
