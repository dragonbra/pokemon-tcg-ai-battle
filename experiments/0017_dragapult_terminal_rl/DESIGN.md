# 0017 Dragapult Terminal RL

Status: **V13 selected as stable viable config at update 20; long-run audit continues**
Date: 2026-07-28
Target: Dragapult ex + Dusknoir, initialized from 0015 V2 R15 exact-best

## 1. Decision summary

0017 is not an Alakazam project and has no runtime dependency on 0016. It starts from the best-supported 0015 Dragapult checkpoint and builds a complete official-engine reinforcement-learning loop around it.

The first reward contract is intentionally narrow:

| Terminal result | Reward |
|---|---:|
| Win | `+1` |
| Loss | `-1` |
| Draw | `0` |

`gamma=1.0`. There is no damage, Prize, evolution, tempo, card-specific, or rule-shaped reward. Dragapult and Dusknoir events are diagnostics only. Engine errors and artificial step-limit truncations are invalid episodes and are discarded rather than labeled as draws.

The initial algorithm is staged rather than one uninterrupted run:

1. freeze and probe the exact BC actor;
2. freeze the actor and calibrate a value head from complete terminal outcomes;
3. run conservative KL-anchored PPO;
4. broaden actor unfreezing only in a new version after stability evidence;
5. select with frozen greedy official-engine evaluation, not sampled rollout win rate alone.

## 2. Evidence boundary

### Official rules

The official game rules define legal evolution, once-per-turn resources, attack-as-turn-end, Prize/deck-out victory, and the separation of public resource zones. They do **not** prescribe a reward function or a good Dragapult strategy. The project rules evidence is [the 2026-07-19 research note](../../docs/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md); exact card effects remain governed by current card data and the unmodified official engine runtime.

### Runtime facts

- A rollout is one real official-engine game from `battle_start` through a terminal engine result.
- One RL decision is one candidate-side engine selection, whose action is an ordered list of legal option indices.
- Declaring an attack ends the turn in the runtime, matching the official rule contract.
- A completed episode can terminate by Prize, no battle Pokémon, or draw failure at turn start; the terminal engine result is the sole reward source.

### Project hypotheses

- The BC actor is already a materially better starting distribution than random RL for this sparse terminal task.
- A calibrated critic and a fixed-BC KL anchor can reduce destructive updates.
- Dreepy/Drakloak/Dragapult and Duskull/Dusclops/Dusknoir timing metrics can explain changes, but they are not assumed to be universally optimal and never affect reward.

## 3. Source policy and provenance

| Field | Frozen value |
|---|---|
| Source project | `0015_dragapult_conditioned_bc` |
| Source version | `V2_t1_plus_pure_dragapult` |
| Checkpoint | `epoch-0010-7f75894270984186.pt` |
| SHA-256 | `7f7589427098418682f3a53e0763f3fd24850d641a0a664955f6094f90530532` |
| Family | `SourceConditionedR15Policy` |
| Actor parameters | `17,416,642` |
| Target persona | `third_ptcg_club`, source ID `1` |
| BC target validation | exact action `58.7%`; legal action `100%` |
| Historical engine evidence | `39/200 = 19.5%` on the older 20-opponent snapshot |

The historical 19.5% is not the 0017 baseline because the current catalog has 26 enabled opponents. V1 must establish a fresh, seat-balanced, 10-game-per-opponent greedy baseline against an immutable current snapshot before any update.

0017 is self-contained. Required R15 model, feature compiler, codec, card semantics, causal online runtime, source conditioning, and portable inference code are copied and frozen under `train/0017_dragapult_terminal_rl/`; executable imports from `train/0015_*` and `train/0014_*` are forbidden. Cross-project paths above are provenance only.

## 4. Actor-visible input contract

The actor preserves all 22 R15 tensor groups; RL adds no privileged game information to the actor.

| Group | Shape before batch padding | Meaning |
|---|---|---|
| `global_cat` | `[4]` | select type/context, first-player identity, turn-resource flags |
| `global_num` | `[12]` | turn/action count, deck/hand/Prize pressure, option and board counts |
| `entity_cat` | `[E, 7]`, `E <= 192` | public card ID, owner, zone, slot, kind, status, parent |
| `entity_num` | `[E, 5]` | damage, Energy/tool/evolution counts, appear-this-turn |
| `entity_mask` | `[E]` | real entity slots |
| `option_cat` | `[O, 12]`, `O <= 128` | action type, source/target area/card/slot/entity, position |
| `option_mask` | `[O]` | current legal options |
| `min_count`, `max_count` | scalar each | engine selection count bounds |
| `zone_inventory_num` | `[2, 16]` | actor-visible resource-zone totals by side |
| `registered_card_ids` | `[R]` | exact registered 60-card deck multiset IDs |
| `registered_multiplicity` | `[R]` | multiplicity aligned to registered IDs |
| `registered_mask` | `[R]` | real registered-card rows |
| `ledger_cat` | `[R, 4]` | known/unknown resource state and deck-order flags |
| `ledger_num` | `[R, 15]` | per-card visible resource ledger |
| `event_cat` | `[H, 8]` | causal public event categories |
| `event_num` | `[H, 4]` | causal event numerics |
| `event_mask` | `[H]` | real event rows |
| `known_opponent_hand_card_ids` | `[K]` | cards revealed and still causally known |
| `known_opponent_hand_mask` | `[K]` | known-hand rows |
| `unknown_opponent_hand_count` | scalar | unrevealed hand remainder |
| `source_id` | scalar | fixed target expert persona `1` |

`targets` is a BC training label and is not actor-visible during rollout. PPO stores the sampled action token sequence separately and replays it through the decoder to compute log-probabilities.

## 5. Network and action contract

The unchanged R15 actor uses width 320, 8 attention heads, four entity encoder layers, one causal event layer, four goal roles, two scenario layers, and a gradual option ScaleGate initialized at 0.35. Exact registered-deck evidence and source persona condition both state and option representations.

0017 adds one scalar value head over the final source-conditioned state representation:

```text
official observation
  -> frozen 0017 causal feature compiler (22 actor-visible groups)
  -> R15 entity/resource/event/goal/scenario readers
  -> source-conditioned state [B, 320] and options [B, O, 320]
       |                                  |
       |                                  +-> autoregressive legal pointer actor
       +-> LayerNorm -> MLP -> tanh -> value [B] in [-1, 1]
```

The actor represents one engine selection as an autoregressive sequence:

1. score every not-yet-chosen legal option;
2. add STOP only after `minCount` is satisfied;
3. sample an option or STOP at temperature 1 during rollout;
4. force termination at `maxCount` or 16 selections;
5. return the selected option indices in sampled order.

The PPO action log-probability is the sum of conditional token log-probabilities, including STOP when sampled. Recomputing new-policy and frozen-reference log-probabilities uses the same stored prefix. Greedy decoding is retained for formal evaluation. Legal-option masking is an action-space contract, not reward shaping; legality must remain 100%.

## 6. Rollout architecture

```text
frozen opponent snapshot + balanced seat scheduler
                     |
        isolated official-engine workers (CPU)
                     |
      compact candidate observations over bounded IPC
                     |
       centralized batched actor-critic inference (GPU)
                     |
 sampled legal action + old log-prob + old value
                     |
         complete terminal episode in RAM
                     |
       value/PPO update -> release rollout buffer
```

Only candidate decisions enter the learning batch. Opponent actions and engine state still participate in the real game but are not optimized. Each stored decision contains compact feature tensors, the sampled action tokens, behavior log-probability, behavior value, episode/seat/opponent identifiers, and diagnostic counters. The terminal result is attached only after a complete game.

The collector has bounded queues and backpressure. Raw observation JSON, complete traces, visualization frames, and engine replays are not written by default. Debug traces require an explicit diagnostic run outside a formal training version. No stale trajectory can cross a version or policy update.

## 7. Opponent and evaluation contract

The 26-package snapshot is frozen by catalog hash, package hashes, display identity, sampling weight, engine hash, seed contract, and seat schedule.

Initial split:

| Slice | Membership | Use |
|---|---|---|
| Train | 20 opponents from the arena snapshot immediately before the 2026-07-28 promotion | uniform opponent sampling, balanced seats |
| Holdout | one Alakazam SOTA, two Cynthia BC, two new Marnie BC, and one new Lucario BC package | never sampled for PPO; periodic frozen greedy check |
| All 26 | train + holdout | V1 baseline and final confirmation |

The six holdout IDs are `alakazam_dudunsparce_04_sota`, `cynthias_garchomp_ex_roserade_01_bc`, `cynthias_garchomp_ex_roserade_02_bc`, `marnies_grimmsnarl_ex_froslass_04_bc`, `marnies_grimmsnarl_ex_froslass_05_bc`, and `mega_lucario_ex_solrock_07_bc`. Exact membership is re-audited when V1 freezes the snapshot; a catalog change requires a new logical version and prevents direct continuation/comparison.

Training rollout win rate measures the current sampled policy against seen opponents and is an optimization diagnostic. Model selection uses periodic frozen greedy evaluation with explicit train/holdout/all-26 breakdown and Wilson intervals. Final claims require the same official engine, package snapshot, seat contract, seeds, game count, and metric profile as baseline.

## 8. Learning objectives

### Value calibration

The actor is frozen. Every valid episode has terminal outcome `z in {-1, 0, +1}`. With `gamma=1`, the Monte Carlo target at every candidate decision is `z`. To prevent long games from dominating, timestep losses in an episode receive weight `1/T`, so each episode contributes equal total value weight.

Initial loss is weighted mean squared error. MAE, RMSE, outcome Brier-style error, explained variance, and calibration buckets are reported. A critic that is constant, seat-biased, opponent-memorizing, or poorly calibrated late in games does not pass the value gate.

### PPO pilot

Initial proposed settings, to be frozen in V3 metadata:

| Setting | Initial value |
|---|---:|
| Complete episodes per update | 256 |
| `gamma` | `1.0` |
| GAE `lambda` | `0.95` |
| PPO epochs | `4` |
| decision minibatch | `1024` |
| actor LR | `1e-5` |
| value LR | `1e-4` |
| ratio clip | `0.10` |
| value coefficient | `0.5` |
| entropy coefficient | `0.01` |
| max gradient norm | `0.5` |
| behavior target KL | `0.02` |

Advantages are normalized within the completed update batch. The fixed V2 BC actor provides a reference KL regularizer; this is a training constraint, not an environment reward. V3 initially unfreezes only the autoregressive pointer decoder, while all representation and conditioned blocks remain frozen. Broader unfreezing is a separate version-level decision.

`gae_lambda` is an explicit immutable-version parameter (`--gae-lambda`, constrained to `[0, 1]`)
and is written into `training_config.json`. V9 remains fixed at `0.95`. A later `lambda=1.0`
branch is a credit-assignment experiment: it gives every decision the undiscounted terminal outcome
through Monte Carlo returns, without changing the terminal reward contract. It must start as a new
version and be compared against V9 rather than appended to it.

An update is skipped on nonfinite loss/gradient. PPO epochs stop early on behavior KL breach. Any adaptive reference-KL coefficient and every freeze transition are logged and versioned.

## 9. W&B and local observability

`training_metrics.jsonl` remains canonical. Each record is flushed locally, then mirrored to TensorBoard, then to the private W&B project `dragon_bra/pokemon-tcg-policy-learning`. W&B failure never rolls back local evidence and must be recorded in status.

### Axes

- `env/episodes`: valid terminal episodes consumed or evaluated;
- `env/decisions`: candidate engine selections;
- `trainer/update`: value/PPO parameter updates.

W&B binds rolling/cumulative/seat/opponent strength namespaces explicitly to `env/episodes`;
PPO health remains bound to `trainer/update`. The SDK-internal `Step` is only a log-row counter.

### Rollout outcome

- rolling win rate over trailing 100, 500, and 2,000 valid episodes;
- matching W/L/D counts, sample count, and 95% Wilson interval;
- cumulative W/L/D and reward mean/std;
- first-seat and second-seat rolling/cumulative rates;
- per-opponent and per-archetype rolling rates and sample counts;
- complete rounds, candidate decisions, terminal reason, engine errors, truncations, and discarded episodes;
- scheduled versus realized opponent distribution, policy version/hash, engine and pool snapshot hashes.

### PPO health

- policy loss, value loss, entropy, total loss;
- approximate KL to behavior policy and KL to frozen BC reference;
- clip fraction; ratio mean, p95, and max;
- advantage/return mean and std; explained variance;
- gradient norm, parameter/update norm, actor/value learning rates;
- optimizer epochs completed, target-KL early stops, skipped and nonfinite updates.

### Value calibration

- MAE, RMSE, Brier-style outcome error, explained variance;
- predicted value mean/std/min/max;
- calibration by early/mid/late phase, seat, opponent, archetype, and final outcome;
- reliability buckets with predicted versus empirical terminal outcome.

### Performance and system

- episodes/sec, candidate decisions/sec, rollout/update wall time;
- inference request and batch-size distribution, p50/p95 latency;
- engine wait time, IPC queue depth, backpressure, worker restarts;
- GPU utilization, allocated/reserved/peak memory, CPU/RAM where available;
- free GiB on `/` and `/mnt/c`, version/artifact/checkpoint bytes.
- project-level `gpu_runtime_audit.jsonl` samples the matching CUDA training PID once per minute and
  accumulates only intervals where that process is present in NVIDIA compute-app telemetry; the
  12-hour requirement is not inferred from rollout episode counts or process intent.

### Dragapult diagnostics, never rewards

- Dreepy -> Drakloak -> Dragapult evolution timing and completed lines;
- Duskull -> Dusclops -> Dusknoir timing and completed lines;
- Cursed Blast availability, use, target, self-KO, and resulting Prize/relay state;
- post-self-KO active replacement and next effective attack;
- Phantom Dive submissions, active damage, bench counter placement, KO and Prize conversion;
- manual Energy, Supporter, Retreat, Stadium, and attack timing budgets;
- complete rounds and attack/non-attack terminal patterns.

High-cardinality opponent/calibration data is emitted as bounded W&B tables at checkpoints; stable aggregate scalars remain on the main curves.

## 10. Storage and restart contract

- Rollout buffer lives in bounded RAM and is discarded immediately after its update.
- Checkpoints contain actor weights, value weights, model/config hashes, update counters, metrics, and provenance only.
- Optimizer, scheduler scaler, replay buffer, raw observations, full traces, and visualization are excluded.
- Keep at most eight model-only checkpoints per version; remove only checkpoints covered by the declared retention policy, never version evidence.
- No exact resume claim exists. Restarting from any saved weights allocates a new `V<n>_<tag>`, creates a fresh optimizer, and collects fresh on-policy episodes.
- Warn when either monitored filesystem is below 80 GiB free. Gracefully stop before `/` or `/mnt/c` falls below 50 GiB, or the current version reaches 20 GiB.

At design time `/mnt/c` has about 184 GiB free and WSL `/` about 775 GiB free, but both are monitored because the WSL virtual disk can still consume Windows storage.

## 11. Gates and failure modes

| Gate | Action |
|---|---|
| Legal action rate below 100% | stop updates; treat as action-contract bug |
| Engine error/truncation above 1% over 500 attempts | pause collection and diagnose; never convert to reward |
| Any nonfinite actor/value state | skip update, preserve status, stop after repeated occurrence |
| Behavior KL exceeds 0.02 during PPO epoch | stop remaining epochs for that batch |
| Reference KL or entropy changes abruptly | reduce actor step or branch a new version |
| Holdout greedy win rate materially collapses with adequate sample | do not promote; stop or branch |
| Training win rate rises while frozen evaluation falls | treat as opponent/policy overfitting |
| Disk warning/hard limit | prune only declared transient/checkpoint retention; otherwise graceful stop |

Sparse terminal reward does not guarantee monotonic real strength. PPO can exploit a finite opponent pool, forget BC behavior, collapse exploration, overfit seat/matchup quirks, or improve noisy sampled reward while degrading greedy evaluation. The fixed reference, holdout pool, confidence intervals, and immutable evaluations are therefore part of the algorithm contract, not optional reporting polish.

PPO separates the live actor, a frozen behavior snapshot created at each update for the ratio
denominator, and the fixed BC reference used for long-horizon anchoring. Rollout-time log-prob is
an audit metric because dynamic inference batches can have different padding widths.

## 12. Version roadmap

| Version | Parameters updated | Evidence produced | Exit condition |
|---|---|---|---|
| `V1_contract_probe` | none | 260/260 valid games, 38W-221L-1D, 14.62% baseline | complete; zero engine errors/discards |
| `V2_value_calibration` | none | 490 rollout episodes | stopped before backward: 2,000-episode batch underused the GPU |
| `V3_value_calibration_512` | none | 10 rollout episodes | stopped for the requested worker-throughput calibration |
| `V4_value_calibration_512` | value head only | 512 fresh episodes, then four value epochs | finite and useful critic without actor-logit drift |
| `V5_ppo_pilot` | pointer decoder + value | update 1 reached KL 0.0494 | stopped: epoch-level KL guard reacted too late |
| `V6_ppo_lr2e6` | pointer decoder + value | update 1 reached KL 0.0504 | stopped: lower LR alone did not fix guard granularity |
| `V7_ppo_minibatch_kl_guard` | pointer decoder + value | first minibatch KL was already 0.0493 | stopped: dynamic batch log-prob mismatch isolated |
| `V8_ppo_frozen_behavior` | pointer decoder + value, LR 2e-6 | 10 stable updates; KL 4e-6 to 1e-5 | viable contract, but rolling-2000 ended 10.5% and learning was too slow |
| `V9_ppo_lr1e5` | pointer decoder + value, LR 1e-5, lambda 0.95 | rolling-2,000 peaked at 18.9%; update 60 ended 17.85% with seat divergence | stop after update 60; retain update 50 as stable branch point |
| `V10_ppo_lambda1` | pointer decoder + value, LR 1e-5, lambda 1.0 | update 1 rollout-log-prob MAE 0.181 exposed train-mode dropout | stopped before update 2 backward; do not use checkpoint |
| `V11_ppo_lambda1_eval_mode` | same V10 hypothesis with pre-rollout eval contract | undiscounted terminal credit with exact behavior probabilities | require first-update log-prob audit near numerical tolerance |
| `V12_ppo_lambda097` | same V9 branch point, LR 1e-5, lambda 0.97 | interpolate lower-variance bootstrap and earlier terminal credit | seek V9 strength with V11 seat balance |
| `V13_ppo_lambda097_lr5e6` | V12 update 5, actor LR 5e-6, lambda 0.97 | preserve high explained variance while slowing actor drift | stabilize 19% sampled strength without second-seat collapse |
| later explicit version | broader actor blocks if justified | controlled capacity comparison | only after decoder-only evidence and user review |

Formal V1 and final evaluation use 10 games per each frozen opponent with balanced seats. Periodic probes may use fewer games but retain official-engine provenance and are not allowed to overwrite formal reports.

The preflight implementation completed three CPU official-engine episodes and a two-episode CUDA training smoke. The CUDA smoke covered 135 candidate decisions, one value epoch, one PPO epoch, finite loss/KL/ratio/entropy, and a model-only checkpoint round trip. Peak allocated CUDA memory was 205,267,456 bytes. These are contract checks, not strength evidence.

V9 established that corrected decoder-only PPO can improve sampled train-pool strength. Its
rolling-2,000 rate rose from 9.4% at update 2 to a peak of 18.9% at update 54. By update 60 it was
17.85%; rolling seat rates had diverged to 20.3% first versus 15.4% second while entropy fell to
0.579. All 15,360 episodes were valid, behavior KL remained 0.000152, and no update guard fired.
The run is therefore stopped as a successful but plateauing configuration, not as a failure.
Update 50 is the stable V10 branch point because it is an actually retained checkpoint near the
long-window peak and precedes the persistent seat gap.

V10 then exposed a pre-existing first-update defect rather than valid lambda evidence. The loaded
model remained in train mode during the first rollout, so its 0.10 dropout produced actions under
a different distribution from the eval-mode frozen behavior snapshot. The first update's
`ppo/rollout_log_prob_mae` was 0.181; the same one-update-only pattern was visible retrospectively
in V9. V10 was stopped before update 2 backward. From V11 onward, PPO loads the live model into
eval mode before collection and `RolloutCollector` rejects any train-mode policy. This keeps legal
categorical sampling as the only rollout stochasticity.

V11 passed its first five updates with rollout-log-prob MAE between `3.85e-7` and `3.97e-7`.
Explained variance ranged from 0.17 to 0.22. After 1,280 valid episodes its sampled-policy rate
was 17.81%, with first/second seat rates of 18.59%/17.03%. This validates the probability and
credit-assignment implementation. At update 20, rolling-2,000 reached 17.95%, below the V9
update-50 branch point's 18.55%, while seat rates improved to 17.7%/18.2% and entropy remained
0.628. Lambda=1 therefore improves the balance/exploration diagnostics but does not show a strength
gain. V12 tests `gae_lambda=0.97` from the same clean branch point with all other variables fixed.

V12 confirmed the credit-assignment interpolation: explained variance was 0.42-0.53 and value
loss about 0.09-0.12. Rolling-2,000 reached 19.01% at update 6, but fell for four consecutive
updates to 17.8% at update 10; second-seat rate fell from 16.9% to 14.3%. V13 branches from the
retained V12 update-5 checkpoint and halves only actor LR to `5e-6`, retaining lambda 0.97 and
value LR `1e-4` to test whether slower actor movement stabilizes the early gain.

V13 reached rolling-2,000 19.36% at update 7 and 18.25% at update 10. First/second seat rates
at update 10 were 19.0%/17.5%, a much smaller gap than V12. Behavior KL was about `3e-5` and
explained variance was 0.48. This supports the lower-LR stabilization hypothesis, but the run must
continue beyond 10 updates before selecting it over the V9 update-50 checkpoint.

At update 20 V13 reached rolling-2,000 19.55% (19.75% at update 18), with rolling-500 19.2%.
Explained variance was 0.515, behavior KL `3.23e-5`, and rollout-log-prob MAE remained about
`4e-7`. V13 therefore passes the viable-configuration gate and is selected for uninterrupted
long-run training through the remaining 12-hour audited CUDA target, subject to existing guards.

## 13. Later version decisions

The 2026-07-28 representative 40-game CUDA benchmark selected 12 isolated workers: 4/8/12/16
workers produced 0.750/1.117/1.397/1.317 episodes/s with zero errors. Formal collection therefore
uses 12 workers with one CPU thread inside each worker. The value rollout batch was reduced from
2,000 to 512 episodes; PPO retains 256 episodes per update so collection and backward alternate on
a several-minute cadence rather than leaving the GPU in inference-only mode for about half an hour.

Later version-level choices are:

- whether the six newly promoted SOTA/BC opponents should remain a strict holdout or some should enter a later curriculum;
- whether a later version should open final scenario/source-conditioned blocks after decoder-only evidence;
- whether the 256-episode update size and 20-update frozen-evaluation cadence fit the observed engine throughput.
- whether a post-V9 version should use `gae_lambda=1.0` to reduce early-action credit decay while
  keeping the actor LR, frozen behavior, BC reference, and opponent pool fixed.

None of these changes the reward contract. Any approved change is frozen in the version manifest before collection begins.
