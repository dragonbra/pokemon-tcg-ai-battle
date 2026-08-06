# 0033 Decisions

## 2026-08-06: Restore full W&B display-name prefixes

The seven 0033 W&B runs had stable `0033-...` IDs and the correct project group, but their display names contained only `V<n>_...`. Their display names are normalized in place to `0033 · dragapult_third_ptcg_club_rl · V<n>_...`; IDs, URLs, groups, metrics and run states remain unchanged. Future 0033 launches use the same full display-name contract.

## 2026-08-06: Select the Third PTCG Club exact Dragapult list

The user replaced the earlier Limitless Dragapult 001 proposal with the exact
current Third PTCG Club list. The stored deck has 18 Pokemon, 32 Trainers and
10 Energy, exactly 60 cards, with file SHA-256
`5db1e0d52fc723e8b2f688d76f780715fd726171f4d804f1d4d55673ba9b0ac2`.
Its order-invariant sorted-multiset SHA-256 is
`f0314f941a872d37dcbaf70371eb576945cc78dbd9aebb8896fbea5e6d8777eb`.
Third PTCG Club is provenance only and is not actor-visible.

## 2026-08-06: Establish the PT0805 Epoch 13 zero-shot baseline first

V1 used the external PT0805 0031 `best_validation_loss` checkpoint, Epoch 13 /
step 109135, SHA-256 `285b88f5e30c40ad07ad025b429c5bd1594f0359cc0d44849c368056222d63ae`.
Against the complete 51-deck Frozen 0019 pool it finished 510/510 official
engine games with no error: 243-267-0, 47.6471%. This is the immutable pre-RL
baseline, not evidence about future POD-native checkpoints.

## 2026-08-06: Reuse the accepted 0032 resident PPO architecture as self-contained code

0033 freezes a physical copy of the accepted 0032 POD-native model, transfer,
rollout, PPO, checkpoint and watchdog implementation. It has no executable
import from another numbered training project. The focal deck is read from the
0033 package; Frozen opponent deck and shared Foundation assets remain allowed
repository infrastructure.

The deployed learner policy updates only its action decoder. The separately
trained value head is an optimization critic and never feeds action logits.
All encoder parameters remain frozen, are omitted from per-update checkpoints,
and are hash-checked by the canary. Opponents share the immutable 0019 Epoch 13
decoder across the 38 CUDA-supported exact decks.

## 2026-08-06: Use V2 for the first formal PPO run

`V2_third_ptcg_club_cuda_ppo_200u` starts from PT0805 initialization with a
fresh optimizer. It uses 152 lanes, 256 resident steps, all 38 CUDA-supported
Frozen decks every update, W&B online and atomic decoder/value model-only
checkpoints retained after every completed update. Sampled rollout metrics are
diagnostics; frozen greedy strength evaluation requires a later version.

The 152-lane one-update official-engine canary completed 9,728 engine
decisions with zero engine errors and zero illegal rows. Behavior log-prob MAE
was `1.53e-7`; the frozen-state SHA-256 was identical before and after PPO,
while the decoder/value state changed. The canonical evidence is
`.tmp/evaluation/0033_dragapult_third_ptcg_club_canary/ppo_one_update.json`.

## 2026-08-06: Stop V2 because it did not preserve the 0031 actor contract

V2 was stopped after update 6 (233,472 engine decisions and 942 completed
Episodes). Its resident `PolicyCodecV1` omitted the 0031 resource ledger,
chronological event memory, registered-deck memory and full card/attack/skill/
effect prototype path. The POD-native actor was therefore not the evaluated
PT0805 `SemanticPolicy`, even though its decoder had been initialized from that
checkpoint. All V2 checkpoints are invalid for selection and continuation.

The replacement path must run the original 56,352,322-parameter 0031
`SemanticPolicy`, its exact `0031_rule_faithful_semantic_decision_v2` online
compiler and per-game causal state over official CPU observations. Any feature,
logit or greedy-action mismatch against the evaluated PT0805 package fails
closed before optimizer construction.

## 2026-08-06: Launch V3 with the full semantic CPU contract

V3 uses the exact 56,352,322-parameter PT0805 actor and all 39 keys from
`0031_rule_faithful_semantic_decision_v2`. Each game/player has independent
causal state. Only immutable prototype tables are cached read-only. The
official CPU engine supplies every observation, legal option and terminal
outcome. The original decoder and a separate critic are the only trainable
modules.

Each update contains all 51 Frozen 0019 decks, both focal seats and two seeds
per seat: 204 complete Episodes. Terminal-only reward uses `gamma=1.0` and
`lambda=1.0`, giving every action the same undiscounted terminal-return target.
PPO retains minibatch 512, two epochs, decoder/value LR `1e-5`/`1e-4`, clip
0.10, entropy 0.01, reference KL 0.02 and target behavior KL 0.02.

Throughput profiling selected 24 official-engine workers and a 5 ms inference
coalescing window. A 32-worker point improved selection throughput by only
0.7% and was rejected for lower operating margin. The complete 204-Episode
PPO gate passed with 38,025 selections, 22,214 focal decisions, zero errors,
186.67 s rollout, 23.91 s PPO, behavior log-prob MAE `8.11e-6`, unchanged
representation hash and changed original decoder. The gate report is
`.tmp/evaluation/0033_full_semantic_cpu_gate/full_204_ppo_w24_c5.json`.

## 2026-08-06: Stop V3 at update 84 and select update 20

The user stopped V3 after 84 complete updates, 17,136 Episodes and 1,857,882
focal decisions. The trainer, watchdog, tmux session and official-engine worker
processes were terminated without deleting any model-only checkpoint. V3
remains valid for selection because every completed update preserved the full
39-key semantic contract and the frozen representation hash.

Checkpoint selection uses only the comparable 102-game periodic frozen greedy
evaluations, never sampled rollout rates. Updates 10 through 80 scored 45.10%,
53.92%, 50.00%, 43.14%, 46.08%, 50.98%, 50.00% and 48.04%, respectively.
Update 20 is selected with 55-47-0 (53.9216%). Update 74 is added as a newer
candidate because the rollout collected from checkpoint 74 and recorded on
trainer row 75 reached 69-135-0 (33.8235%), a recent sampled-rollout high.
That diagnostic does not establish checkpoint strength. Both candidates will
receive separate 510-game official CPU engine evaluations against all 51
Frozen 0019 decks under the same seed and balanced-seat contract as V1. Both
packages use FP16 weight storage, FP32 inference and GPU-accelerated local
batched policy inference.

Both formal evaluations completed 510/510 games with no error. Update 20 scored
252-258-0 (49.4118%), including 51.7647% when acting first and 47.0588% when
acting second. Update 74 scored 256-254-0 (50.1961%), including 51.3725% first
and 49.0196% second. Update 74 is the provisional candidate because it is newer
and has the higher same-contract point estimate, but its four-win margin over
update 20 is not statistically conclusive. Both outperform the V1 zero-shot
point estimate of 47.6471%; neither 510-game comparison alone proves a true
strength gain.

## 2026-08-06: Begin sequential turn-semantic credit experiments

The next stage is three sequential 20-update experiments, with a mandatory
analysis and formal evaluation gate before the next design is allocated. If
none credibly improves on the shared frozen evaluation contract, a fourth
20-update control restores legacy per-selection `lambda=0.95` while retaining
the valid full 0031 semantic policy and official CPU engine. The invalid V2
reduced feature path is never an eligible control.

V6 changes only temporal credit. Net reward remains official terminal outcome
`+1/0/-1`, `gamma=1.0`, with no Prize, damage, attack or setup bonus. Each focal
decision records official observation `current.turn`. Same-turn transitions
use lambda 1.0; only a turn change uses `lambda_turn=0.97`. At six to ten
boundaries, direct terminal residual retention is approximately 83.3% to
76.0%, instead of repeatedly applying decay to atomic selections. V6 uses 204
Episodes/update, 20 updates, full Frozen51 seat balance and frozen greedy
evaluation every five updates. V7 and V8 remain unallocated until the prior
analysis artifact and formal evaluation are complete.

## 2026-08-06: V6 supports the turn clock; V7 tests turn-equal loss weighting

V6 completed 20/20 updates with 4,080 official-engine Episodes, 443,819 focal
decisions, zero errors and an unchanged representation hash. Its periodic
greedy checkpoints scored 46-56, 46-56, 49-53 and 42-60 at updates 5, 10, 15
and 20. The predeclared rule therefore selected update 15. Its formal Frozen51
evaluation completed 510/510 at 273-237-0 (53.5294%), 58.8235% first and
48.2353% second. This is above V3 update 74's 256-254 point estimate, but the
paired comparison has 119 V6-only wins and 102 V3-only wins (exact McNemar
`p=0.282`), so superiority is not established.

The training diagnostics support retaining the turn clock: final-five return
standard deviation was 0.8328 versus 0.8963 for the corresponding V3
first-20 window; V6 explained variance rose from a first-five mean of 0.0996
to a final-five mean of 0.1769; value loss fell from 0.5881 to 0.5617. The
remaining concern is selection-count bias inside long main phases. V7 is
therefore `V7_turn_equal_loss_weighting_20u`: terminal reward, gamma 1.0,
turn-clock lambda 0.97, 204 Episodes/update, optimizer, PPO coefficients,
Frozen51 schedule and 24-worker execution remain unchanged. The only delta is
that each Episode assigns equal total loss mass to each official focal turn,
then divides that turn's mass among its atomic decisions. V7 starts fresh from
PT0805 with a fresh optimizer.

## 2026-08-06: Reject V7 formally; V8 doubles on-policy Episodes

V7 completed 20/20 updates and zero-error operation. Its periodic frozen
greedy curve was 54-48, 56-46, 49-53 and 54-48, selecting update 10. The
larger formal evaluation did not reproduce that apparent advantage: update 10
finished 243-267-0 (47.6471%), with 49.4118% first and 45.8824% second. Against
V6 update 15 on the same 510 game IDs, V7 gained 91 games and lost 121 (exact
McNemar `p=0.046`). Turn-equal loss weighting is therefore rejected as a
regression despite its stronger 102-game periodic points.

V7 also retained visible checkpoint variation (54.90% at update 10 versus
48.04% at update 15), while return standard deviation rose from a first-five
mean of 0.7728 to a final-five mean of 0.8262. V8 follows the predeclared
regression branch: revert to V6's Episode-equal, decision-equal loss weights
and keep terminal-only reward, gamma 1.0, turn-clock lambda 0.97 and every PPO
coefficient unchanged. Its sole new hypothesis relative to V6 is doubling
fresh on-policy sampling from 204 to 408 Episodes/update. The version is
`V8_turn_clock_lambda097_408ep_20u`, starts from PT0805 with a fresh optimizer,
and retains frozen greedy evaluation every five updates.

## 2026-08-06: Conditional V9 legacy control and V10 long-run gate

The remaining selection rule is formalized to avoid wasting the 12-hour GPU
window on a slower but not demonstrably stronger batch size. V8 must complete
20 updates, produce `analysis.json`, export the selected periodic checkpoint
and run the same 510-game official CPU Frozen51 evaluation contract as V6 and
V7. If that formal zero-error result is not at least 15 wins better than V6's
273-237 score, run V9 `V9_action_clock_lambda095_control_20u` as the legacy
control requested by the user.

V9 is intentionally not a reduced-feature fallback. It keeps the full 0031
semantic observation contract, official CPU engine rollout, terminal-only
reward, gamma 1.0, 204 Episodes/update, Episode-equal decision weighting and
all PPO coefficients, but returns credit assignment to the older per-selection
clock with lambda 0.95. It starts fresh from PT0805 with a fresh optimizer and
therefore produces an independent W&B run, checkpoint tree and formal report.

The selected V10 200-update run also starts fresh from PT0805. It uses V8's
408-Episode schedule only if V8 clears the +15 formal-win threshold; otherwise
it selects the best zero-error 204-Episode design by formal Frozen51 wins,
which remains V6 unless V9 beats it. Rollout win-rate spikes are diagnostics
only and cannot select V10.

## 2026-08-06: V8 fails formal threshold; V9 legacy control starts

V8 completed 20/20 updates with 8,160 official CPU Episodes, 884,204 focal
decisions, online W&B logging and zero rollout errors. Its periodic frozen
greedy checkpoints were 45-57, 52-50, 49-53 and 47-54-1; analysis selected
update 10 as the best comparable point. The exported candidate
`0033_v8_turn_clock_lambda097_408ep_update10` validated as an exact 60-card
Dragapult Third PTCG Club package with FP16 storage and FP32 runtime.

The formal 510-game Frozen51 evaluation for V8 update 10 finished 252-258-0
(49.4118%) with zero errors. This is 21 wins below V6's 273-237 formal result,
so V8 does not clear the predeclared +15-win threshold for using the slower
408-Episode/update schedule in V10.

The supervisor therefore triggered V9 `V9_action_clock_lambda095_control_20u`
as the user-requested legacy control. The V9 gate passed before training:
official CPU rollout plus PPO smoke completed, full 0031 semantic runtime
parity passed with 312 tensor comparisons and 25,519 value comparisons,
greedy actions matched exactly, and the shared representation hash remained
unchanged. V9 started from PT0805 with a fresh optimizer, W&B run
`0033-v9-action-clock-lambda095-control-20u`, per-selection lambda 0.95,
204 Episodes/update and the same full semantic observation contract.

## 2026-08-06: V9 loses the control comparison; V10 selects V6 turn-clock

V9 completed 20/20 updates with 4,080 official CPU Episodes, 442,590 focal
decisions and zero rollout errors. Periodic frozen probes scored 47-55 at
update 5, 52-50 at update 10, 50-52 at update 15 and 49-53 at update 20;
analysis selected update 10. The exported candidate
`0033_v9_action_clock_lambda095_update10` validated as the exact Third PTCG
Club 60-card deck and kept FP16 storage with FP32 CUDA inference.

The first V9 formal-evaluation attempt failed before launching workers because
the project manifest `status` field had been repurposed for dynamic experiment
state. Formal evaluation preflight requires project manifests to keep
`status: initialized`; the dynamic state now lives in `experiment_status`.
After restoring that contract, the official CPU Frozen51 evaluation completed
510/510 games with zero errors and scored 242-268-0 (47.4510%).

The V10 long-run selection therefore rejects the legacy action-clock control:
V6 remains the strongest comparable zero-error formal result at 273 wins, while
V8 scored 252 wins and V9 scored 242 wins. V10 is allocated as
`V10_turn_clock_lambda097_200u`, starts fresh from PT0805 with a fresh
optimizer, and uses the V6 design: terminal-only reward, gamma 1.0, turn-clock
lambda 0.97, 204 Episodes/update, 24 official CPU workers, 5 ms inference
coalescing, Episode-equal decision weighting, W&B online logging and full
model-only checkpoint retention.

## 2026-08-06: V10 stopped by user at update 44

The V10 long run was stopped at the user's request before reaching the planned
200 updates. The stop was issued at 2026-08-06T21:35:36+08:00 by interrupting
the tmux session `0033_v10_turn_clock_200u`; a follow-up process check found
no remaining tmux session, monitor process or `run_full_semantic` process for
V10.

The latest fully persisted training state is update 44, with 8,976 official
CPU Episodes, 969,191 focal decisions, full schema parity true and unchanged
shared representation hash. Checkpoints were retained from update 0 through
`update-000044.pt`. The last periodic frozen diagnostic evaluation was
update 40, scoring 49-53-0 (48.0392%) over 102 games; this remains a diagnostic
probe only, not a formal strength conclusion. Because V10 did not complete 200
updates, it is recorded as user-stopped rather than as the final selected
long-run result.

## 2026-08-06: V11 archives the independent PT0805 zero-shot repeat

V11 preserves the temporary `run-b9c8eae58ec14123b5328bb00c67cb42`
report as an immutable formal evaluation. The source `pt0805.tar.gz`
checkpoint hash is `285b88f5e30c40ad07ad025b429c5bd1594f0359cc0d44849c368056222d63ae`,
the exact Third PTCG Club deck hash is
`5db1e0d52fc723e8b2f688d76f780715fd726171f4d804f1d4d55673ba9b0ac2`,
and the Frozen 0019 pool remains `0019_foundation_51_exact_decks_v4`.

The official-engine outcome contract completed 510/510 games with zero game
errors at 242-268-0 (47.4510%), one win below the original V1 result. This is
a repeat evaluation of unchanged weights, not a new training version or a new
policy checkpoint. The `league_deck_quality` diagnostic parser returned a
`ValueError` for all 510 traces, so its process metrics are unavailable and
must not be interpreted as zero-valued gameplay behavior; official outcomes,
seat splits and per-opponent records remain valid.
