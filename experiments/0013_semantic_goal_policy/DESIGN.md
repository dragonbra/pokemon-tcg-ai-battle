# 0013 Semantic Goal Policy

**Project ID:** `0013_semantic_goal_policy`
**Status:** raw `dataset_reference_v3` and source-frozen V3 model-ready tensors are published and
fully validated; V4 completed the reproducible three-epoch M0 minimum; V5 stopped after 16 complete
epochs plus a discarded partial epoch 17 and selected epoch 14 as validation-loss-best; that M0
snapshot is archived and has an official-engine catalog report; V6 is preserved as a user-stopped
M5 diagnostic run at epoch 8; M5.1 remains implemented but unlaunched and unauthorized by the
current frozen protocol; F1 and its richer feature dataset remain deferred.

## Purpose and boundaries

This project tests a semantic and causal representation ladder for one deck-specific policy. The
initial source remains the Alakazam-Dudunsparce deck family played by the named expert Yushin Ito.
It does not mix decks, teams, or experts; infer an opponent's hidden deck; consume future frames,
outcomes, rewards, or omniscient visualizer state; modify `engine/source/`; promote candidates
without user approval; or perform a Kaggle submission.

The three project roots are:

- implementation: `../../train/0013_semantic_goal_policy/`
- tracked design and evidence: this directory
- generated runtime assets: `../../rl_runs/0013_semantic_goal_policy/`

Formal `V1_m0_causal_contract` was allocated, stopped before an epoch completed, and is preserved
with failure class `input_pipeline_throughput_defect`. It will not be resumed or overwritten.

## Frozen source and action contract

The source boundary is the eight official daily episode archives from 2026-07-18 through
2026-07-25 plus the declared 2026-07-24 gap patch. Eligibility is normalized exact expert name,
unique winner, complete all-DONE trajectory, causal actor-visible decisions, and the actual
registered deck for that episode-player. Submission IDs are unavailable, so this is a named-expert
boundary rather than a claim that every trajectory came from one verified binary policy.

A target is an ordered sequence of distinct legal option occurrences. STOP is impossible before
`minCount`. At `maxCount`, termination is forced and contributes zero STOP log-probability and
entropy. Live option counts and labels are never truncated; more than 64 legal options fail closed.
Occurrence identity is alignment/audit metadata only and is never a model feature.

The frozen protocol digest is
`da6482cf2d4cdc8d9e56fd4c03431e60dbf63b8327720c83185e907a039d44d3`.

## Shared deterministic feature compiler

The public compiler API is `features.compiler.compile_features(raw, registered_deck, knowledge,
registry)`, followed by `tensorize_compiled(...)`. Both chronological offline preparation and the
official-engine runtime `PolicySession` call this exact API. Offline preparation reconstructs one
`CausalKnowledgeState` per `(date, episode_id, player_index)` before DataLoader shuffle. Runtime
resets the same state at each new game and records each submitted action after the decision.

A valid `dataset_reference_v3`, checkpoint, candidate package, and evaluation manifest must bind:

- typed input schema version `semantic_goal_typed_input_v1`;
- feature compiler version `semantic_goal_feature_compiler_v2` and a transitive SHA-256 over the
  compiler, observation/schema, causal knowledge/ledger/visibility, and chronological batcher
  sources;
- deterministic `CardSemanticRegistry` SHA-256;
- source manifest, split, protocol, record schema, and action-contract digests.

Card semantics use audited structural card fields, ordered move/effect primitives, and functional
capabilities. Card text and names are not direct model features. Card ID is retained as an identity
residual. Unknown card, missing field, not-applicable field, and padding have separate encodings.
Numeric fields use paired `(clipped value, state code)` columns; the state code distinguishes
observed, remembered, inferred exact, bounded, unknown, missing, not applicable, padding, and the
overflow/padding bits. Serial values are used only to associate physical instances and are never
numeric strategy features.

The raw decision dataset remains the immutable provenance and action-label layer. Before formal
training, a second immutable model-ready layer reconstructs each episode-player state exactly once
and publishes hash-committed PyTorch tensor shards. Storage uses int16 categorical/target tensors,
float16 numeric/semantic tensors, boolean masks, and sparse int16 relation edges; loading restores
the runtime dtypes and dense relation tensor. Independent episode-player groups compile in bounded
ordered worker processes, while chronology within every group remains serial. Training then only
shuffles shards/rows, applies synchronized option permutation, reconstructs sparse relations,
prefetches pinned batches, and transfers them asynchronously.

A verified full `select.deck` membership initializes exact deck and inferred Prize identity
counters. Later actor-visible Draw and MoveCard transitions update those counters; Shuffle removes
order knowledge but preserves membership. Active/Bench attachments and pre-evolutions, Stadium,
and the resolving context card participate in serial-deduplicated resource conservation. An
unidentified transition degrades only the affected hidden-zone identity counter to unknown.

## Executable token specification

All categorical tensors use zero for padding. Non-padding integer values use an offset identity;
string enums use deterministic SHA-256-derived categorical IDs. All masks use `true` for live
positions. Semantic vectors are 64-wide deterministic structural/capability vectors.

| Family | Tensor contract | Maximum and overflow | Padding meaning |
|---|---|---|---|
| state | one token; `state_cat[8]`, `state_num[16]` | exactly one; categorical slots 5-7 record entity, event, and relation-endpoint overflow counts | not padded |
| entity | `entities_cat[N,8]`, `entities_num[N,12]`, `entity_semantic[N,64]` | `N<=128`; deterministic actor-visible player/zone/slot traversal, tail omitted and counted in state | no entity |
| deck | `deck_card_ids[D]`, multiplicity, `deck_semantic[D,64]` | unique registered identities, `D<=60`; no legal overflow | no registered identity |
| option | `options_cat[O,12]`, `options_num[O,8]`, `option_semantic[O,64]` | `O<=64`; overflow is a hard error | no legal occurrence |
| ledger | `ledger_cat[L,6]`, `ledger_num[L,12]`, `ledger_semantic[L,64]` | one token per unique own registered identity, `L<=60`; no legal overflow | no resource identity |
| event | `events_cat[E,6]`, `events_num[E,8]`, `event_semantic[E,64]` | newest 64 actor-visible incoming logs; omitted count recorded in state | no event |
| relation | `relations[N,N]` categorical adjacency | endpoints outside retained entities are omitted and counted in state | relation type zero means absent |
| goal | four learned `d_model` retrieval slots | exactly four when enabled | disabled variants use zero goal output |
| value | one finite player-relative scalar | M5 only; uncalibrated during BC | not a BC target |

State categorical fields are actor, first player, select type, select context, select effect, and the
three overflow counts. State numeric pairs are turn, turn-action count, own/opponent deck count,
own/opponent hand count, and select minimum/maximum. Entity categorical fields contain owner, zone,
card identity residual, instance-presence identity, and slot context; numeric pairs include HP,
maximum HP, and player summary counts where applicable. Option categorical fields contain action
type, card/effect/attack identity, count, energy/tool references, source/target player/area/index,
and special-condition type. Option numeric pairs contain count, number, energy index, and tool
index. Ledger values contain registered, current-deck, and inferred-prize counts plus their causal
knowledge states. Event values contain relative age and actor-visible event identity.

## M0-M5 variant matrix and M5.1 ablation

Every variant uses the same ordered-action decoder, state token, actor-visible entities, legal
options, masks, dataset split, and target representation.

| Variant | State | Entity | Deck | Option | Ledger | Event | Relation | Goal | Value |
|---|---|---|---|---|---|---|---|---|---|
| M0 | categorical + numeric | categorical identity/context | compiled but inactive | primitive/context categorical | inactive | inactive | inactive | inactive | inactive |
| M1 | M0 | + numeric + 64d semantics | compiled but inactive | + numeric + 64d semantics | inactive | inactive | inactive | inactive | inactive |
| M2 | M1 | M1 | semantic registered-deck masked mean added to state | M1 | inactive | inactive | inactive | inactive | inactive |
| M3 | M2 | M1 | Goal-QKV K/V memory | M1 | inactive | inactive | inactive | four deck retrieval slots | inactive |
| M4 | M2 | M1 | deck K/V memory | M1 | ledger K/V memory and state tokens | inactive | inactive | four deck+ledger retrieval slots | inactive |
| M5 | M2 | M1 | deck K/V memory | M1 | M4 | event state tokens | per-head directed entity attention bias | M4 | finite, uncalibrated head |
| M5.1 | M2 | M1 | masked mean only; Goal K/V disabled | M1 | M4 state tokens; Goal K/V disabled | event state tokens | same as M5 | zero goal output; no goal tokens | same finite, uncalibrated head |

M2's registered-deck masked summary is cumulative through M3-M5. M5 maps each available typed
entity relation to a learned per-head directed attention bias at its source/target pair; relation
type zero contributes no bias. The accepted V3 schema contains entity-indexed relations only, so
richer event/source/target relation schemas remain outside this ladder and belong to a future F1
dataset contract.

M5.1 is a full-minus-one ablation, not another cumulative ladder level. It instantiates every M5
module in the same order and has exactly the same state-dict tensors and initial model SHA when the
seed matches M5. Only Goal-QKV is disconnected from the policy forward and its four goal tokens are
omitted; deck summary, ledger state tokens, events, relation bias, options, decoder, and value
interface remain active. This isolates the marginal contribution of Goal-QKV without rebuilding V3.

The reference capacity is `d_model=384`, six pre-norm state layers, eight heads, FFN width 1536,
two option cross-attention layers, and dropout 0.1. This defines the 0013 ladder baseline. It is not
a strict causal comparison with 0012 V5, which used `d_model=320` and four state layers. A future
strict control, if required, must be named `M0_0012_matched` and retain the 0012 capacity and budget;
no 0012 strength claim may be made from current M0.

## Goal-QKV decision

M3 through M5 share one Goal-QKV module design and its Q/K/V parameters. Four learned role
embeddings represent setup/board development, attack/Prize progress, resource access/recovery, and
tempo/survival. They are latent retrieval roles, not supervised goal labels. M3 K/V contains
registered-deck tokens. M4 and M5 K/V contain the concatenation of registered-deck and causal ledger
tokens. The Q source is the raw state token before entity/state Transformer contextualization. No
diversity regularizer is used in the primary M0-M5 ladder. M5.1 retains the module parameters for
matched initialization but does not call it or append goal tokens.

`raw state + masked board/entity summary` is reserved for an explicit future `M3b/M4b` controlled
ablation and must not silently replace the primary Q source. Because the registered deck is mostly
constant in this deck-specific policy, M2/M3 gains would support conditional capability retrieval,
not cross-deck generalization.


## Option permutation and probability contract

BC training randomly permutes each live legal-option occurrence. The batcher applies one bijection
to option categorical, numeric, semantic, and mask tensors, then remaps every ordered expert target
with the inverse occurrence map. STOP remains the padded batch-wide terminal index. Validation runs
both canonical-order metrics and a deterministic permutation audit. Aligned teacher logits and
action loss must remain invariant within the frozen protocol thresholds.

The centralized decoder performs teacher forcing and batched greedy full-action decoding. Exact
action, token accuracy, legal completion, length, termination kind, select type/context, and
permutation slices are imitation/contract metrics only. Formal runs create W&B at startup and report
throttled train, train-eval, and validation-eval progress with iterations/s, decisions/s, fraction,
and elapsed time; complete split metrics remain one summary per epoch.

## Value and later PPO boundary

M0-M4 do not expose an active value objective. M5 and M5.1 create the same finite player-relative
action-before state scalar solely to establish the interface; BC does not optimize or interpret it. Before value
calibration or PPO, both DESIGN files must be updated with the target, reward, player perspective,
terminal/truncation mask, bootstrap and discount convention, loss weights, calibration metrics, and
official-engine rollout evidence. No value or PPO result currently exists.

## Current and next stage

Implemented: frozen source/protocol/action/split contracts; atomic raw shard publication and
audited orphan recovery; published raw v3 dataset with 145,961 train and 16,167 validation
decisions; typed schema and ontology; chronological causal knowledge with dynamically maintained
exact deck/Prize resources; transitive compiler v2; synchronized option permutation; M0-M5 model
paths plus the unlaunched M5.1 Goal-QKV ablation; centralized batched full-action decoder; model-ready tensor shard writer/reader; eager W&B
lifecycle and phase progress; and focused contract tests.

V1 is an immutable failed record: it repeated raw JSON/causal compilation in every train/evaluation
pass and wrote no complete epoch metric or checkpoint. Interrupted model-ready staging attempts are
audited and unpublished. A complete V2 materialization is also rejected: a format-only source edit
occurred after workers imported the compiler but before the final manifest digest was calculated,
so its exact source provenance is invalid even though behavior did not change.

The source-frozen deterministic 8-worker build `V3_model_ready_semantic_v2` is accepted. It contains
143 train and 16 validation shards, binds unchanged compiler/materializer digests, and passed full
commitment validation, 1,024-record raw/cache tensor parity, real forward/backward throughput,
complete untrained validation, and single-decision runtime smokes. At batch 64 the input path
reached 5,765 decisions/s while M0 optimization reached 638 decisions/s, so feature loading no
longer starves the model. Full evidence is in
`data_audit/materialized_v3_audit_2026-07-26.json`.

V4 completed all three planned epochs with validation loss `1.12633 -> 1.05466 -> 1.00068` and
validation exact action `0.55279 -> 0.58960 -> 0.61434`; its W&B run is synced and epoch 3 is best.
V5 restarted from the same bound initial model with the same dataset, M0 architecture, batch 64,
seed 20260726, learning rate 3e-4, weight decay 1e-2, AMP, and deterministic behavior. It completed
16 epochs before the process disappeared during epoch 17 at 38.58% optimization progress. Epoch 14
is the immutable validation-loss-best snapshot (`loss=0.69076082`, exact action `0.72474794`, legal
action `1.0`); epochs 15 and 16 improved train metrics but regressed validation loss, so partial
epoch 17 is excluded. V6 independently initialized M5 with the same seed and optimization settings,
but was user-stopped after epoch 8 when diagnostics established unnormalized activation scale and
slow optimization; its partial epoch 9 is excluded. Both interrupted boundaries and their W&B
lifecycle state are recorded under the corresponding version artifacts.

The epoch-14 M0 state is exported as the self-contained
`../../archive/submission/0013_v5_m0_epoch14_loss_best/` payload, with optimizer state excluded and
source checkpoint, model-state, compiler, dataset, protocol, deck, and runtime hashes retained. Its
matching tar archive is under `../../archive/submission/dist/`. Formal official-engine report
`evaluation/V5_m0_100epoch_monitor.html` ran 20 catalog opponents x 10 games with eight workers and
one CPU thread per worker. The immutable report records 92 wins, 98 losses, 10 errors, no unfinished
games, a fixed-denominator win rate of 46.0%, and 95.0% completion. All ten errors are engine errors
against `alakazam_dudunsparce_03` at the first 0-2 selections; the other 19 opponents produced 190
completed games without candidate runtime errors. Completed-game win rate is therefore 92/190 =
48.42% (first 47/95 = 49.47%; second 45/95 = 47.37%). This is official-engine evidence with an
explicit catalog correctness caveat, not a clean promotion result and not grounds for automatic
opponent-pool admission.

V5 and V6 were launched before the evaluation/telemetry runtime optimization and therefore retain
their already-loaded two-encoding FP32 train/validation evaluation path, `progress/iteration` W&B
axis, and disabled console capture for their immutable records. Subsequent formal versions consume
the train split only once, aggregate online teacher-forced optimization diagnostics from the same
forward used by backward, omit static full-train evaluation and train-time greedy decode, and retain
one complete epoch-end validation snapshot. Validation uses one shared encoding, on-device metric
aggregation, configured AMP, epoch-axis BC progress, and wrapped W&B Logs. The canonical
`training_metrics.jsonl` remains the scalar fact source in both contracts.

This M5 direction check deliberately skips M1-M4 and does not implement F1. F1's richer option,
event/relation, serial-exclusion, and board-conditioned Goal-query dataset/model contract is
paused until current M0/M5 evidence is reviewed. M5.1 is code-complete and tested but remains only a
prepared option: the current frozen protocol authorizes M0-M5, and both command entry points reject
M5.1 under its SHA. A real V3 batch-64 AMP optimizer smoke matched M5's initial model SHA, kept all
Goal-QKV gradients disconnected, produced finite loss, and used 1.742 GiB peak CUDA allocation. No
V7 directory, W&B run, checkpoint, metrics, or persistent GPU process has been created. A later
launch requires explicit confirmation and a new immutable ablation protocol.

The monitored M0/M5 runs are closed and M1-M4 and F1 remain paused. The archived M0 package is an
evaluated project artifact, but the report's 10 catalog engine errors prevent treating it as a clean
correctness-gate pass. Any resumed comparison must retain explicit raw source, model-ready feature,
split, seed, compiler, action, and evaluation contracts. Promotion into the formal opponent pool
still requires a separate clean official-engine report and explicit user confirmation.
