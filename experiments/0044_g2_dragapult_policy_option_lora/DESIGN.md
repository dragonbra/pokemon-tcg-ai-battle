# 0044 · G2 Dragapult Policy-only Option LoRA

## Current stage

0044 is a self-contained PPO lineage whose latest immutable policy is promoted
`Champion-G3`. V1–V10 are preserved historical runs. The latest completed training version is
`V10_g3_v9_u50_generalist_001_067_core16_benchmark_v2`, stopped at durable local
U110. The user explicitly promoted this exact U110 to immutable Champion-G3 on
2026-08-14 after accepting the available partial Full-67 CUDA-2048 evidence.
V11 is the immutable Full-67 evaluation version. The first G4 attempt,
`V12_g4_g3_meta_balanced_512_meta_residual`, completed U0 evaluation and one 512-game
rollout but failed before its first optimizer step because the focal Agent's pre-seat
context-41 choice had been staged as a semantic PPO transition. V12 is retained as a
failed immutable run and produced no U1. V13 fixed that boundary and completed its
first optimizer update, but the historical telemetry layer still required exactly
256 games and failed after writing U1 but before canonical rollout metrics. V13 is
also retained as failed and is not resumed. `V14_g4_g3_meta_balanced_512_rollout_only`
restarts from immutable G3 with a fresh optimizer and fresh on-policy data.

V14 completed durable local U2 and then received a transient WDDM/CUDA
`device not ready` error partway through the U3 optimizer update. There was no U3
checkpoint, no allocator OOM, and an independent CUDA smoke passed after the device
recovered. V14 remains immutable at U2. V15 strict-loads that model-only U2 as local
U0, reinitializes the optimizer, discards the partial U3 rollout/update, keeps all
evaluation disabled, and reduces only the physical forward microbatch from 1024 to
768. The logical minibatch remains 4096 and the PPO/model/sampling semantics do not
change. Reference and frozen opponent remain immutable Champion-G3 U110.

V15 stopped cleanly at durable U60. U57 was selected as the next focal parent from
the sampled-rollout candidate region. V16's Champion-G3 versus U57 full-pool
evaluation was paused by user direction after retaining its completed reports.
On 2026-08-14 the append-only deck registry added Alakazam decks `068` and `069`,
then explicitly admitted both to the formal training/opponent pool. V17 started a
new local U0 from exact V15 U57, completed its first rollout and optimizer update,
then fail-closed after writing U1 because the new `meta_balanced_training_pool` mode
was omitted from telemetry's closed enum. U1 has no canonical metrics and is not
used as a continuation parent. V18 is the clean same-semantics retry from the
original exact V15 U57, with a fresh optimizer and fresh on-policy data. It fixes
focal deck `069`, keeps Champion-G3 immutable, disables PFSP/evaluate, and uses
rollout-only monitoring. V18 stopped at durable local U17; a partial U18 optimizer
pass was discarded and produced neither a checkpoint nor canonical metrics.

The common-schedule Benchmark V2 CUDA-2048 checkpoint audit compared V18 U14,
U15, U16, U17 and a finite normalized EMA over U14–U17. U14 was the observed best
at 1269-749-30 (61.9629%); the EMA scored 1259-760-29 (61.4746%) and is retained
only as an audited derived candidate. V19 therefore starts a new local U0 from the
exact model-only V18 U14 checkpoint, creates a fresh optimizer and fresh on-policy
data, fixes focal deck 069, uses 1,024 rollout games per update, and applies a
run-only half-standard learning-rate profile. Champion-G3 and the live Meta-balanced
001–069 opponent contract remain unchanged; PFSP and evaluate remain disabled.
Within that opponent contract, display Meta `01` (`archetype_id=0`, Dragapult ex;
decks 007/018/067) receives weight 2.0 while each other active Meta receives 1.0.

V19 completed its 1,024-game U0 rollout but failed during the partial U1 PPO Value
cross-attention with `CUDA driver error: device not ready`. It wrote no U1 and the
independent post-failure CUDA smoke passed. PPO had reached 15,866 MiB on the
16,303 MiB device, so V20 is a new immutable retry from the same V18 U14 parent
with fresh optimizer/data. It changes only physical forward microbatch 768 → 512;
logical minibatch 4,096, 1,024 episodes, three epochs, weights, losses, LR profile,
sampling and policy identities are unchanged.

V20 reproduced the reset in its first U1 backward and also produced no U1;
microbatch 512 alone left the peak at 15,834 MiB. V21 therefore applies the
execution-level root fix: reference logits are still computed in full on CUDA from
the exact immutable V18 U14 reference, then those frozen reference weights move to
CPU only after the complete per-decision cache exists; behavior probe and PPO
physical chunks are 256. Reference log-probs, logical sample membership, gradients,
optimizer boundaries and all training semantics are unchanged.

V21 completed all 60 U1 optimizer steps and durably wrote canonical metrics plus
`update-000001.pt`, then the device reset at the start of U2 rollout. The current
safe boundary is V21 U1. A fourth retry is intentionally not automatic: preserving
1,024-game semantics now requires a phase-boundary memory architecture that unloads
the frozen rollout opponent and releases CUDA Engine buffers before PPO, then
restores them before the next rollout; otherwise the run must return to 512 games.

`V22_v21_u1_rollout512_meta6_x3` is the user-authorized continuation from that
exact durable V21 U1 model-only checkpoint. It starts at a new local U0 with a fresh
optimizer and fresh on-policy data; the incomplete V21 U2 rollout is discarded. V22
returns to 512 terminal games per update and changes the opponent Meta-first quota
weights: internal Own Archetype V2 Meta IDs `00/01/02/03/05/27` each receive weight
3.0, while every other active Meta retains weight 1.0. On the current 28-active-Meta
pool this gives each weighted Meta 38 lanes and each other Meta 12–13 lanes, while
still covering all exact decks `001`–`069`. Focal deck 069, the complete immutable
Champion-G3 opponent, half-standard LR, three PPO epochs, logical minibatch 4,096,
physical/probe microbatch 256, reference-after-cache offload, PFSP=0 and evaluate=0
remain unchanged.

V22 was stopped by explicit user direction after durable U9 in preparation for
the final pre-deployment fine-tune. Its canonical metrics and model-only checkpoints
end together at U9; partial U10 produced neither checkpoint nor metrics and is
discarded. `V23_v22_u9_final_entropy_0015` strict-loads exact V22 U9 as local U0,
starts a fresh optimizer and fresh on-policy collection, and changes exactly one
training objective coefficient: entropy coefficient `0.003 → 0.0015`. It retains
V22's half-standard learning rates, 512 games/update, three PPO epochs, logical
minibatch 4,096, physical/probe microbatch 256, six 3.0× Meta weights, complete
immutable Champion-G3 opponent, reference-after-cache offload, PFSP=0 and evaluate=0.
The lower entropy bonus does not change the official legal-action contract or make
rollout greedy; focal actions remain categorical samples from the current policy.

V23 was stopped by explicit user direction at durable U11. The subsequent U11
behavior-policy rollout completed at 294-216-2 (57.4219%), but the U12 optimizer was
interrupted during epoch 2 step 4/10; no U12 checkpoint or canonical metric exists,
and the partial U12 update is discarded. A deployment-audited common-schedule greedy
Benchmark V2 CUDA-2048 checkpoint audit then compared V23 U7/U8/U9/U10 and a finite
normalized decay-0.5 EMA over U7–U10. All five candidates passed
`kaggle_fp16_storage_fp32_runtime_v1`, complete Policy-0809 opponent identity, routing,
and 2,048/2,048 terminal-game gates. EMA U7–U10 was the observed best at
1267-753-28 (61.8652%), only one win above U9/U10 and two wins below the prior V18 U14
observed best on the identical common schedule. This is checkpoint-selection evidence
only; no Champion promotion or continuation parent is chosen automatically. The
[comparison report](../../docs/evaluation/combat_mat/benchmark_v2/0044_v23_deck069_u7_u10_ema_checkpoint_selection_core16_policy0809_cuda2048_v2/index.html)
is the detailed Meta/exact-deck audit.

The user subsequently selected the historical V18 U14 observed best for the deck
069 Kaggle payload. It is archived as
`archive/submission/0044_alakazam_dudunsparce_069_v18_u14_fp16_storage_fp32_runtime/`
and materialized under `kaggle_fp16_storage_fp32_runtime_v1`. Its qualifying
common-schedule CUDA-2048 evidence remains 1269-749-30 (61.9629%); source FP32,
portable FP16 and deployment-effective hashes are bound in the package manifest.
The exact archive passed fresh-extraction validation, raw Kaggle-style execution
without `__file__`, and 10/10 terminal official-engine smoke games with zero errors.
This is a user-selected submission payload, not an automatic Champion promotion,
and package creation does not itself authorize or execute a Kaggle submission.

`V24_v23_u11_deck023_standard_lr_entropy` is the next user-authorized formal PPO
version. It strict-loads exact durable V23 U11 as local U0, then switches the focal
main-view exact deck from 069 to deck 023 (`Hydrapple ex / Meganium`, Own Archetype
V2 class 27). It starts a fresh optimizer and discards all earlier rollout data so
the first V24 batch is newly sampled on-policy for deck 023. The run restores the
project default `0044_standard_lr_v1` rates (decoder/policy adapter/allocation/meta
residual 5e-6, Option LoRA 1e-5, Value/Prize 2e-5) and the standard entropy
coefficient `0.003`; these replace V23's half-standard LR and entropy `0.0015`.
All other live-training semantics remain V23-compatible: 512 terminal games per
update, Meta `00/01/02/03/05/27` at 3.0×, complete immutable Champion-G3 opponent,
three PPO epochs, logical minibatch 4,096, physical/probe microbatch 256,
reference-after-cache offload, PFSP=0, evaluate=0, FP32 PPO, and model-only/all
checkpoint retention. The reference is an independent strict-load of V23 U11;
changing focal deck conditioning must not alter the opponent's effective identity.

## Policy identities and decks

- V17 focal initialization: exact V15 U57 model-only FP32 weights; optimizer state
  is not reused and the new version begins at local U0.
- V17 focal deck is exact `069`; only own-deck conditioning uses its existing
  29-class Own Archetype V2 class `03`. Deck069 is Deck068 with Night Stretcher −1
  and Boss's Orders +1. The embedding shape and Value/opponent-Meta tensors do not change.
- Reference model: an independent frozen strict-load of exact V15 U57. Opponent
  policy pool: singleton immutable complete `Champion-G3`; no Champion-G4 is promoted.
- V22 training reference: an independent frozen strict-load of exact V21 U1. This
  changes the new-run reference anchor only; the rollout opponent remains the
  separately resolved immutable complete `Champion-G3` policy.
- Opponent deck pool: exact numeric IDs `001`–`069`. Every lane binds the exact
  deck, its G3 own-deck embedding row, and G3 effective weights. Opponent weights
  never update and are resident once per rollout collection.
- The frozen 15-way opponent Meta prediction head is unchanged and not expanded.

## Model data flow

The 320-wide Option Encoder has two Transformer blocks. Inputs and block 0 run
once. The final block runs twice:

1. Value branch: immutable final block with no LoRA. Its option tensor feeds the
   Value network, opponent Meta auxiliary prediction, prize Value head, and the
   strategic context used by the policy adapter.
2. Policy branch: the same immutable final block with Q/V LoRA on self-attention
   and cross-attention. This tensor alone feeds the action decoder and allocation
   policy.

The adapter is exactly rank 4, alpha 8, block index 1, eight tensors and 10,240
parameters. A matrices use Kaiming initialization and B matrices start at zero.
Therefore the original 0044 U0 was bit-identical to G2 for both branches, while
Value gradients can never reach LoRA. V12 inherits G3's trained LoRA tensors rather
than reinitializing them. CUDA Engine 2.0 uses the same project-local dual-option adapter;
it shares Option inputs/block 0 and routes the two final outputs explicitly.

V12 adds a policy-only Meta Actor Residual after the strategic adapter and before
decoder logits. It has 29 experts, width 320, rank 4, and exactly 74,240 parameters.
Each expert has a `[4,320]` down matrix and `[320,4]` up matrix; up matrices start
at zero, so V12 U0 is exactly G3. The selected own-Meta ID chooses one expert.
This residual never enters the Value branch, the frozen 15-way opponent Meta head,
or opponent policy inference. The G3 archive contains the same module at exact zero
only to make the successor deployment schema complete; it was not trained in G3.

## PPO and optimizer

V12 uses PPO Protocol V2 with 512 rollout games, three epochs, logical
minibatch 4,096 and physical forward microbatch 1,024. This keeps approximately the
same optimizer-step count while doubling episode coverage. Behavior KL target 0.015,
hard guard 0.025, clip 0.10, entropy
coefficient 0.003, and reference KL coefficient 0.02. Reference KL is computed
against an independent, fully frozen complete G3 model, not a decoder-only copy.

The following values are the named project default `0044_standard_lr_v1` (the
“standard LR” profile). This default is project-wide and is not bound to deck 069
or any other exact deck:

- action decoder: `5e-6`
- policy strategy adapter: `5e-6`
- allocation head: `5e-6`
- policy-only Option LoRA: `1e-5` (new in 0044; preserves the approved 2× relative multiplier)
- policy-only Meta Actor Residual: `5e-6`
- Value trunk/value adapter/prize auxiliary: `2e-5`
- AdamW weight decay: `0`

V19 alone uses `0044_v19_half_standard_lr_run_override`, exactly 0.5× every
learning-rate field above: decoder/strategy adapter/allocation/Meta Actor Residual
`2.5e-6`, Option LoRA `5e-6`, and Value/prize `1e-5`. Its scope is `run_only`, its
`deck_id` is null, and the default after this run returns to
`0044_standard_lr_v1`. V19 keeps three PPO epochs, logical minibatch 4,096 and
physical forward microbatch 768 while increasing episode coverage to 1,024 per
update; increasing rollout coverage does not multiply the logical minibatch or the
number of epochs.

V22 preserves that same run-only half-standard profile after strict-loading V21 U1,
but starts a fresh optimizer as required for a new version. It uses 512 rollout
episodes, three PPO epochs, logical minibatch 4,096, physical forward microbatch 256
and behavior-probe batch 256. The smaller rollout is a declared sampling-semantic
change and does not alter policy identity or checkpoint retention.

V23 keeps this exact half-standard profile. Its sole optimization-semantic delta is
the entropy coefficient reduction from `0.003` to `0.0015`; LR, PPO clip/KL guards,
reference coefficient, Value losses and checkpoint retention remain unchanged.

## V9 opponent sampling

Every V9/V10 update independently sampled all 256 opponent decks uniformly with replacement
from exact IDs `001`–`067`. The opponent policy is always the immutable complete
`Champion-G2`. PFSP and latest-champion branches both have zero lanes; V9 neither
loads, writes nor updates a PFSP state. This is exact-deck IID uniform sampling,
not Meta-first sampling. The future opponent-deck distribution remains a separately
versioned decision from the first-player correction below.

## V12 G3 → G4 sampling

PFSP remains disabled. For each 512-game update, focal and opponent independently
sample the 28 non-empty Own Archetype V2 Meta classes first. Every Meta receives
18 or 19 games, using one shared quota seed so both sides have the same Meta
marginal. Within each Meta, games are balanced over its member exact decks with
count spread at most one. Focal and opponent use different member-remainder and lane
shuffle seeds, so equal marginals do not create artificial paired matchups. All 67
decks appear on both sides every update. Opponent policy is always Champion-G3.

## V17 deck-069 specialization sampling

V17 fixes every focal lane to exact deck `069` and own class `03`. The opponent
policy is always immutable Champion-G3. Opponent decks use the live
`meta_balanced_training_pool` contract: sample the 28 non-empty Own Archetype V2
Meta classes first, then balance each Meta quota over its member exact decks.
The live pool is contiguous `001`–`069`; all 69 decks receive at least one of the
512 opponent lanes per update. PFSP is disabled. The seeded toss winner still runs
the complete Agent's official context-41 choice; the scheduler does not choose seats.

V19 preserves this exact focal/opponent routing but expands the update to 1,024
lanes. Meta classes remain quota-balanced first, then each Meta quota is balanced
over its member exact decks. The larger batch reduces episode-sampling variance; it
does not introduce a focal-deck execution grouping or alter policy identity. Its
only sampling customization is a 2.0× opponent weight for display Meta 01
(internal class 0, Dragapult ex). This deterministically gives that Meta 71 of
1,024 lanes and the other active Meta classes 35 or 36 lanes; its 71 lanes remain
balanced across exact decks 007/018/067 with count spread at most one.

V22 returns this routing to 512 lanes and replaces V19–V21's single class-0 2.0×
override with six 3.0× overrides for internal Meta IDs `00/01/02/03/05/27`.
All other active Meta IDs use 1.0×. The deterministic U0 quota is 38 lanes for each
weighted Meta and 12–13 for each unweighted Meta; per-Meta exact-deck balancing still
has count spread at most one and all 69 opponent decks remain present. This is an
explicit opponent-distribution experiment variable recorded in V22's config; it
does not mutate Champion-G3 or any opponent effective weights.

## Periodic Benchmark V2 evaluation

V14 explicitly disables both baseline and periodic evaluation. A single focal deck
is not treated as representative evidence for 67-deck group intelligence, and no
CUDA-2048 evaluate interrupts the long run. Selection diagnostics come from the same
512 terminal rollout episodes: global and rolling win rates, exact focal/opponent
deck and Meta slices, prize margin, actual seat, KL/loss and throughput. These remain
sampled rollout diagnostics rather than frozen greedy strength evidence.

The following is the retained V1–V13 historical evaluation contract:

Before each new formal version collects its first rollout, it materializes its local
U0 weights
as the new version's local U0 checkpoint,
materializes it under `kaggle_fp16_storage_fp32_runtime_v1`, and runs official-engine
Benchmark V2 CUDA-2048 in greedy mode. The result is logged at
`eval/checkpoint_update=0`. The same evaluation repeats after each durable checkpoint
divisible by ten (`U10`, `U20`, ...). Focal deck is `066`; opponent policy is the
complete immutable `Policy-0809`. Games are restricted to Meta IDs `00`–`13`, `17`
and `27`: exactly 128 games per Meta, then balanced across its member decks with count
spread at most one. Deck allocation, engine/Search/policy seeds and coin-winner seeds
are common random numbers independent of focal identity.

The Core-16 schedule pins the pre-068 taxonomy SHA `9d9bc5cb…1f60` and mapping
SHA `6b7d9702…120`; it also hard-pins opponent membership to numeric IDs `001`–`067`
instead of following the live training role. Admitting 068/069 to the live training
pool therefore cannot change a historical Benchmark V2 opponent, seed, job order,
or common-random schedule hash.

V12 retains this Core-16 Benchmark V2 contract but uses focal Deck `007` as its
longitudinal sentinel; this evaluate is not an aggregate of the 67 focal decks.

Only a PASS report is mirrored to W&B under `eval/*`, including checkpoint update,
W/L/D, overall and seat-split win rates, game count, and throughput. These metrics
are separate from stochastic `rollout/*`, do not enter PPO, and do not affect
training sampling.

## Checkpoint and deployment boundary

All update checkpoints are retained as atomic model-only FP32 files. Periodic
evaluation candidates store the full dual-branch effective policy in FP16,
strict-load every floating tensor to FP32, record the source checkpoint hash,
portable artifact hash, effective tensor hash, exact focal deck hash, and refuse
legacy single-option schemas that would apply LoRA to Value.

## Launch record and version history

The CPU tests, G2 identity audit, U0 parity, gradient isolation, CUDA dual-route
smoke, behavior-logprob parity, periodic candidate round-trip, and official-engine
CUDA rollout gates passed before the initial launch. V1 exposed excessive repeated
reference-model work during PPO and failed before U1; V2 cached the complete G2
reference logits once per rollout batch and durably produced U1, then exposed a
telemetry gate that incorrectly expected three opponent weight loads. V3 corrected
that singleton-resident G2 gate and was stopped after U21.

V4 attempted these declared variables but failed before any Benchmark game or PPO
rollout because the new Benchmark runner lacked the module-inventory adapter for
the complete Policy-0809 runtime. Its used directory and W&B run are retained as a
failed version. V5 is the pristine, same-semantics retry after that adapter fix.

The intended V4/V5 experiment changes only the declared variables: fresh G2
initialization, 100% uniform opponent-deck sampling, the 0042 V1 actor learning-rate
scale, and Benchmark V2 fixed-seed Policy-0809 evaluation including the U0 baseline.
V5 safely stopped at durable U4 when the evaluation domain was revised. V6 preserves
its training semantics and changes the evaluation contract to the final Core-16 scope.
V6 stopped after durable U33 at the user's request. V7 then failed before W&B,
checkpoint, Benchmark, or rollout creation because its first generalized Deck-003
load exposed a legacy loader hardcode that resolved every exact deck as `007`.
The identity gate rejected the mismatch. V8 is the same-semantics retry after the
loader and candidate materializer were changed to require the explicit deck ID.
V8 was the fresh-optimizer Deck-003 continuation and stopped after durable U20 plus
its completed Core-16 Benchmark. V9 is a fresh-optimizer, fresh-rollout continuation
of those exact V8 U20 weights with only the focal deck/own row changed from 003 to
066 and a version-local update reset to U0. The opponent pool,
low learning rates, policy-only LoRA/value detach, Reference-G2 anchor and Core-16
Benchmark V2 contract are unchanged.

## V9 U50 → V10/G3 handoff

V9 is bounded by an external fail-closed terminal gate rather than by an in-process
update limit. It may be interrupted only after all three representations of local
U50 agree: atomic model-only `update-000050.pt`, a PASS official-engine Benchmark
V2 CUDA-2048 report bound to checkpoint U50, and the matching canonical
`eval/checkpoint_update=50` metrics row. The supervisor freezes their SHA-256
identities, sends SIGINT only to the validated formal V9 process, refuses a handoff
if U51 became durable, and preserves V9 as the fixed-066 experiment.

The successor is
`V10_g3_v9_u50_generalist_001_067_core16_benchmark_v2`, the first G3 training
version. It strict-loads the frozen V9 U50 model-only state as local U0, creates a
fresh optimizer, discards all PFSP state, and begins fresh on-policy collection.
It is intentionally unbounded (`updates=None`) for long-running training.

For every 256-game G3 rollout, focal exact decks are frequency-balanced across
ordered IDs `001`–`067`: each deck occurs three or four times, the extra 55 slots
rotate deterministically, and seed `430044711 + source_policy_update` shuffles the
lane assignment. Every lane carries its exact 60-card bytes and its Own Archetype
V2 row; the same IDs are copied into every stored transition and restored for both
the behavior/reference PPO forwards. Focal deck is therefore a model feature, not
an execution grouping key. The CUDA collector retains one resident focal model and
groups only by opponent policy (the singleton immutable Champion-G2), avoiding a
67-way batch-fragmentation or weight-loading regression. Routing audits, zero
feature D2H, batch size, cache hit/miss, and per-focal rollout W/L/D remain visible
under `rollout/*`.

G3 opponents retain V9 semantics: complete immutable Champion-G2 with uniformly
sampled exact decks `001`–`067`, no PFSP/latest branch, and no opponent weight
updates. Reference KL remains anchored to an independently materialized complete
Champion-G2; V9 U50 is the trainable initialization, not the reference policy.

Periodic Benchmark V2 continues on fixed focal Deck066 at local U0 and every ten
updates. In V10 this is explicitly a longitudinal sentinel that is directly
comparable to V9 U50, not an aggregate estimate of all 67 focal decks. Multi-deck
G3 health is instead diagnosed from exact per-focal rollout telemetry and routing
coverage unless a future separately versioned all-focal benchmark is approved.

## First-player contract correction after V10

Historical V1–V10 PPO rollout schedules preassigned exactly 128 focal-first and 128
focal-second games and bypassed the winning Agent's official context-41 choice. That
legacy training behavior is retained as historical fact and must not be relabeled as
the Agent-owned contract. Benchmark V2 already used the correct seeded-toss contract,
so its reports and the Full-67 G2/G3 comparison remain valid.

Every successor after V10 must use
`seeded_toss_winner_agent_context_41_choice_v1`:

1. the schedule derives unique engine, Search, policy and coin-winner seeds;
2. the coin-winner seed determines only whether focal or the immutable complete
   Champion-G3 opponent won the toss;
3. that complete winning policy processes official context 41 and chooses first or
   second;
4. the harness never assigns, alternates or balances actual seats;
5. PPO and rollout telemetry consume the recorded Agent-selected actual seat, not
   the toss winner or a schedule slot;
6. missing seed, toss winner, Agent choice, or actual seat is fatal before PPO.

Context 41 is an auditable Agent policy decision, but it occurs before an actual seat
exists and is outside the strategy model's `relative_first_player ∈ {1,2}` contract.
V13 therefore retains the choice in first-player evidence and rollout diagnostics but
excludes it from PPO trajectories. Every staged semantic transition is post-seat and
`prepare_episodes` fails closed if its relative-first-player value is not 1 or 2.
V12's failure established this boundary before any parameter update; its U0 weights
remain identical to G3.

Actual focal-first/focal-second counts are therefore observed rollout metrics. They
are not schedule quotas. This correction does not mutate V10 checkpoints and must be
used only by a new version with a fresh optimizer and fresh on-policy collection.
