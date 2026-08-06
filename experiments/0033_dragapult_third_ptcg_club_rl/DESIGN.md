# 0033 Dragapult Third PTCG Club RL Design

## Scope and evidence boundary

0033 trains the exact user-provided `Dragapult Third PTCG Club` 60-card deck,
file SHA-256 `5db1e0d52fc723e8b2f688d76f780715fd726171f4d804f1d4d55673ba9b0ac2`.
The actor is the exact PT0805 0031 Epoch 13 / step 109135
`best_validation_loss` `SemanticPolicy`, checkpoint SHA-256
`285b88f5e30c40ad07ad025b429c5bd1594f0359cc0d44849c368056222d63ae`.

V1 is the immutable zero-shot baseline: 510/510 official CPU engine games
against all 51 Frozen 0019 exact decks, 243-267-0, or 47.6471%. V2 used a
reduced POD/CUDA feature path and was stopped after update 6. V2 checkpoints
are invalid for selection or continuation. V3 is the first valid PPO run.

Official rules, legal options, observations and terminal outcomes come only
from the unmodified official CPU engine runtime. Feature interpretation, PPO,
the critic and sampled rollout statistics are project choices. Sampled
rollout win rate is not frozen greedy checkpoint strength.

## Exact actor-visible contract

Every decision uses schema `0031_rule_faithful_semantic_decision_v2` with all
39 actor keys. No field is omitted and there is no fallback codec.

| Family | Ragged shape | Width / role |
|---|---:|---|
| Global | `[B, 12]`, `[B, 24]`, `[B, 24]` | categorical, numeric and field-state facts |
| Card | `[B, C, 9]`, `[B, C, 7]`, `[B, C, 7]` | visible/remembered cards, zones, instances, HP, attachments and resolved Energy |
| Resource | `[B, R, 4]`, `[B, R, 15]`, `[B, R, 15]` | registered-deck ledger, zone counts and known/bounded deck/prize state |
| Event | `[B, E, 31]`, `[B, E, 4]`, `[B, E, 4]` | chronological official logs and typed event facts |
| Option | `[B, O, 19]`, `[B, O, 2]`, `[B, O, 2]` | every legal option, select context, source/target and numeric facts |
| Relations | ragged one-based indices + masks | card parent; event source/target/before/after; option source/target/context/effect card |
| Prototype | ragged IDs, roles, parents + masks | full card/attack/skill/effect prototype relations |
| Selection | `[B]` | exact engine `min_count` and `max_count` |

Each player in each Episode owns an independent `OnlineCausalEncoder` and
`CausalKnowledge`. It preserves the registered deck, known and possible hand,
resource ledger, event chronology and card-instance history across decisions.
Immutable 5.7 MB prototype tables are loaded once per rollout parent and
shared read-only; no game state is shared. Missing fields, actor mismatch,
selection overflow, parity mismatch, worker failure, illegal action or
incomplete Episode fails closed.

The runtime gate compares eight consecutive official observations with the
evaluated PT0805 package: 39 exact keys, 312 tensors and 31,861 values are
bitwise equal; actor tensors and greedy actions are identical, with maximum
logit error `7.153e-7` under `atol=2e-6`.

## Network and trainable boundary

The original actor has 56,352,322 parameters, `d_model=320`, eight heads,
four hierarchical state Transformer layers, one event layer, two option
cross-attention layers and the original autoregressive GRU pointer decoder.
The full PT0805 checkpoint loads strictly with all 293 tensors.

Only the original `actor.action_decoder.*` (1,027,202 parameters) and a new
non-actor value head (103,681 parameters) are trainable, 1,130,883 total. The
value head reads the frozen state summary and never feeds action logits. The
frozen representation SHA-256 is
`15fedc68246c00377a8fd4eca07f88f303fa4af95f5b17493e93ecf9ba564361`.
PPO checks it before and after each update. Model-only checkpoints retain all
updates and exclude optimizer, RNG, rollout and other recovery state.

## Reward, credit and PPO

Only the official terminal outcome supplies net reward: win `+1`, loss `-1`,
draw `0`; intermediate reward is zero. There is no Prize, damage, attack or
setup bonus. `gamma=1.0` preserves the undiscounted win objective. V6 records
the official observation turn on each focal decision and uses a semantic GAE
clock: adjacent decisions within the same official turn use lambda `1.0`, and
only a turn change uses `lambda_turn=0.97`. Atomic card-effect selections do
not create artificial time distance. V6 observed about 10.7 focal turn
boundaries per Episode and retained about 72.7% of the direct terminal
residual at the first decision. V6 gave each Episode total loss weight one and
split it equally across that Episode's decisions. V7 keeps Episode total
weight one, gives every official focal turn equal mass, and splits that turn's
mass across its atomic decisions. This is a loss-sampling hypothesis, not a
new reward or actor feature.

The credit experiments use fresh AdamW, decoder LR `1e-5`, critic LR `1e-4`, two PPO epochs,
minibatch 512 decisions, clip ratio 0.10, entropy coefficient 0.01, reference
decoder KL coefficient 0.02, target behavior KL 0.02, gradient clip 0.5 and
zero weight decay. Behavior log probability must replay within `1e-4` before
updates. Rollout from `source_policy_update=k-1` produces checkpoint `k`.

## Opponents, schedule and throughput

Every update contains exactly 204 complete Episodes: all 51 Frozen 0019 exact
decks, both focal seats, two seeds per seat. Every opponent uses the immutable
0019 Epoch 13 policy, SHA-256
`da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb`.
The catalog and every exact deck hash are snapshotted in the version artifact.

Official engine workers are isolated spawned processes with one internal CPU
thread each. Profiling selected 24 workers and a 5 ms inference coalescing
window on the 8-core/16-thread host; 32 workers added only 0.7% selection
throughput and was rejected. A full 204-Episode plus PPO gate completed with
38,025 engine selections, 22,214 focal decisions and zero errors: rollout
186.67 s, PPO 23.91 s, about 203.7 selections/s and 210.6 s per update before
periodic evaluation. Peak CUDA allocated/reserved memory was 2.66/5.29 GiB.

## Version, logging and current stage

V3 is `V3_full_semantic_cpu_ppo_200u`; it was stopped by the user after 84 of
200 planned updates, 17,136 Episodes and 1,857,882 focal decisions. Every tenth
checkpoint receives a separate 102-game fixed-seed, balanced-seat greedy
evaluation with `eval/*` metrics. Canonical write order is JSONL flush,
TensorBoard, then W&B online in private project
`dragon_bra/pokemon-tcg-policy-learning`, stable run ID
`0033-v3-full-semantic-cpu-ppo-200u`. All model-only checkpoints are retained.

V3 may be selected only by comparable official-engine frozen greedy evidence,
not by sampled rollout rolling peaks. Update 20 is the periodic-evaluation
candidate: its 55-47-0 result (53.9216%) is the best among updates 10-80.
Update 74 is the newer candidate because rollout row 75, collected from policy
update 74, reached a recent high of 69-135-0 (33.8235%). Both packages store
weights in FP16, expand them to FP32 for inference and use GPU-batched local
policy inference. The current stage is two separate 510-game official CPU
engine evaluations against the complete Frozen51 pool under the V1 base-seed
and seat contract. Update 20 completed at 252-258-0 (49.4118%); update 74
completed at 256-254-0 (50.1961%). Both had 510/510 finished games and zero
errors. Update 74 is provisionally preferred because it is newer and has the
higher point estimate, but the four-win margin is not statistically
conclusive.

V6 `V6_turn_clock_lambda097_20u` completed 20 updates, 4,080 Episodes and
443,819 focal decisions with zero errors and unchanged representation. Its
best periodic frozen checkpoint was update 15 at 49-53; the formal 510-game
Frozen51 evaluation scored 273-237 (53.5294%), including 58.8235% first and
48.2353% second. Return standard deviation averaged 0.8328 over the final five
updates versus 0.8963 for V3's first-20 final window, while explained variance
rose from 0.0996 to 0.1769 across V6. Relative to V3 update 74, paired outcomes
were 119 negative-to-positive and 102 positive-to-negative (exact McNemar
`p=0.282`), so the higher point estimate is promising but not conclusive.

V7 `V7_turn_equal_loss_weighting_20u` completed zero-error training and its
periodic update-10 point was 56-46, but the formal 510-game evaluation scored
243-267 (47.6471%), 49.4118% first and 45.8824% second. Against V6 on matched
game IDs it gained 91 and lost 121 games (exact McNemar `p=0.046`). The
turn-equal weighting delta is rejected.

V8 `V8_turn_clock_lambda097_408ep_20u` completed 20 updates, 8,160 Episodes
and 884,204 focal decisions with zero rollout errors. It reverted to V6's
Episode-equal, decision-equal weights and preserved terminal reward, turn
clock, lambda 0.97 and every PPO coefficient; its only experimental variable
was 408 instead of 204 fresh on-policy Episodes per update. The best periodic
checkpoint was update 10 at 52-50 over the 102-game frozen greedy probe, but
the formal 510-game Frozen51 evaluation scored 252-258 (49.4118%) with zero
errors. V8 therefore failed the predeclared threshold of at least 15 more
formal wins than V6's 273-237 result.

V9 `V9_action_clock_lambda095_control_20u`, the legacy control triggered by
V8's formal result, completed 20 updates with zero rollout errors. V9 remained
full-semantic official CPU PPO: terminal reward, gamma 1.0, 204
Episodes/update, Episode-equal decision loss weighting and all PPO coefficients
unchanged, but credit assignment returned to the older per-selection GAE lambda
0.95. Its gate passed full 0031 semantic runtime parity before training,
including 312 tensor comparisons, 25,519 value comparisons, exact greedy action
parity and an unchanged shared representation hash. Analysis selected update
10, but the formal 510-game Frozen51 evaluation scored 242-268 (47.4510%) with
zero errors, below both V6 and V8.

V10 `V10_turn_clock_lambda097_200u` was the selected fresh 200-update run from
the strongest formal evidence, starting from PT0805 with a fresh optimizer and
using the V6 design: terminal-only reward, gamma 1.0, turn-clock lambda 0.97,
204 Episodes/update, 24 official CPU workers, 5 ms inference coalescing and
Episode-equal decision weighting. The formal selection evidence was V6 273
wins, V8 252 wins and V9 242 wins across comparable zero-error 510-game
Frozen51 evaluations. The user stopped V10 at update 44 before the planned 200
updates; it produced 8,976 official CPU Episodes, 969,191 focal decisions,
retained checkpoints through `update-000044.pt`, kept full schema parity true
and left the shared representation hash unchanged. Its last periodic frozen
diagnostic was update 40 at 49-53-0 (48.0392%) over 102 games. That probe and
all rollout spikes remain diagnostics only, not formal strength conclusions.
