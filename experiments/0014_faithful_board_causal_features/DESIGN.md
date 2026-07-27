# 0014 Faithful Board Causal Features

**Project ID:** `0014_faithful_board_causal_features`  
**Status:** feature/visibility audits and the immutable full-feature cache passed. V8 A0 through
V13 R3 are immutable training records with selected official-engine reports. V12 R2 remains the
strongest result at 159/200 (79.5%). V13 R3 early fusion reached 157/200 (78.5%) and exposed
late overfitting from putting scenario tokens inside the board trunk. V14 R4's additive family
router regressed to 151/200 (75.5%). V15 R5 peaked at 156/200 (78.0%) before severe late
overfitting. V16 R6 also regressed to 148/200 (74.0%). V17 R7 recovered to 158/200 (79.0%) only
at its late exact-best checkpoint. V18 R8 regressed to 152/200 (76.0%) at loss-best and 151/200
(75.5%) at exact-best. V19 R9 reached 153/200 (76.5%) at both selected checkpoints. V20 R10
reached 150/200 (75.0%) at loss-best and 155/200 (77.5%) at exact-best. V21 R11 regressed to
143/200 (71.5%) at loss-best and 151/200 (75.5%) at exact-best. V22 R12 reached 154/200 (77.0%)
at loss-best and 150/200 (75.0%) at exact-best. V23 R13 reached 145/200 (72.5%) at loss-best and
153/200 (76.5%) at exact-best. V24 R14 reached 147/200 (73.5%) at loss-best and 148/200 (74.0%)
at exact-best, so structured scenario dropout is also rejected. V25 R15 deterministic gradual
option completed at epoch 19. Its epoch 8 loss-best reached 148/200 (74.0%), while epoch 14
exact-best reached 165/200 (82.5%) and is the selected project SOTA. The R14 result produced the
CPU-smoked R16 rule-contract reader, but the user ended model exploration and the allocated V26
run was intentionally stopped during epoch 1; it is not a SOTA candidate.

## Purpose

0014 asks two practical competition questions on one frozen single-expert corpus:

1. How well does the faithful 0010 ID-only pointer model learn on the newer Yushin Ito raw dataset
   and split?
2. After restoring every 0010 board/option feature, does adding complete actor-visible card,
   public-zone, registered-deck, causal resource, and event information produce a stronger policy?

The project does not treat the incomplete 0013 M5 representation as a base. The V1 full-feature
cache is frozen and audited; its former 0013 raw location is provenance only, deliberately not a
runtime dependency. Model, codec and feature-reader code are self-contained in 0014, while shared
`rl_environment`, official card data and official engine runtime remain repository infrastructure.

## Frozen source and labels

- Frozen model-ready dataset: `../../rl_runs/0014_faithful_board_causal_features/dataset/V1_full_feature_superset`
- Content SHA-256: `146ac0799b0a23a5089615361e76c33722c08df45b276cfd2b387171c70dc0c7`
- Expert: normalized exact `Yushin Ito`, unique winner, complete trajectories only
- Dates: 2026-07-18 through 2026-07-25
- Split: reused without modification
- Decisions: 145,961 train and 16,167 validation
- Labels: ordered full-action indices under `ordered_full_action_v1`

The raw row is immutable provenance. The published model-ready dataset is
`../../rl_runs/0014_faithful_board_causal_features/dataset/V1_full_feature_superset`, content
SHA-256 `e341a7cbc761797f4fa97a4a01fddf2d25d69b8c3c60e4e41bcb3b4e2708686a`.
It reconstructs chronological state once per `(date, episode_id, player_index)` and stores all
162,128 rows in 159 shards (1.124 GiB). Model views select fields from that cache; they never
require separate feature builds.

The exact 0010 config accepts 145,928 train and 16,164 validation decisions. The remaining 36
rows are retained in the full cache but marked `a0_eligible=false` because their ordered action
length is 17–21, above the original 16-step cap. A0 and the primary AC comparison use the same
162,092-row compatibility view.

## Actor-visible and rules contract

Features describe only information available to the acting player before the labeled action. They
never consume a future frame, reward, terminal outcome, opponent hidden-card identity, omniscient
viewer field, or post-action state.

Rules-sensitive state includes turn phase, Supporter/manual-Energy/Retreat/Stadium usage, attack as
a turn-ending commitment, `appearThisTurn`, evolution timing, special conditions, HP/damage,
attachments, and distinct Active/Bench/hand/discard/deck/Prize zones. Unknown is never encoded as
zero. An identity count is exact only when a verified full visible deck view and 60-card
conservation prove it; unidentified transitions degrade the affected knowledge instead of guessing.

Opponent hand/deck/Prize expose public counts. Opponent hand identity exists only after an actual
actor-visible reveal and remains paired with unknown-slot count; an unseen draw creates one unknown
slot, not a guessed card.

## Complete model-ready feature contract

### Faithful 0010 base — always stored

- `global_cat[4]`: select type/context, first player, per-turn action flags.
- `global_num[12]`: turn/action count, both deck/hand/Prize counts, option count, remaining damage
  and Energy payments, Bench occupancy.
- `entity_cat[N,7]`, `N<=192`: card ID, relative owner, zone, slot, kind, status bits, parent index.
- `entity_num[N,5]` stored as FP32: damage ratio, Energy count, Tool count, evolution depth,
  `appearThisTurn`.
- `option_cat[O,12]`, `O<=128`: action type, source/target areas, relative owner, source/target
  card IDs, number, source/target slots, resolved source/target entity references, occurrence index.
- ordered target sequence, termination kind, min/max count and masks. The primary A0/AC comparison
  constructs the legacy 0010 always-STOP target at batch time; the corrected forced-max target is
  retained as a separate explicit future view rather than silently changing the baseline.

A0 must reproduce these tensors and the 0010 forward path without auxiliary tokens, changed masks,
changed sequence lengths or unused-reader parameters.

### Five additional information families — always stored

1. **Card capability.** Card IDs address a committed `card_ontology.json` sidecar containing the
   deterministic official-card structure: kind/stage,
   HP/type/weakness/resistance/retreat, move Energy cost/damage, audited effect primitives and
   capabilities. Missing, unknown and not-applicable are distinct.
2. **Public-zone inventory.** Both players' observed zone counts and visible board aggregates.
   Opponent hidden zones provide counts only.
3. **Own-resource ledger.** Per registered card identity: initial multiplicity, currently visible
   zone counts, deck/Prize value or bounds, epistemic state, source event and age.
4. **Causal event memory.** Numeric official-engine logs accumulated across decisions with stable
   `(actor_decision_index, local_log_ordinal)`, relative actor/age, card identity and known/unknown
   hand consequences. Redacted opponent-hand departure clears unsupported definite identity.
5. **Registered deck.** The actual own 60-card identity multiset and multiplicity for that episode.

Relations connect owner/location, attachment, evolution, option source/target and audited event
transitions. Serial numbers associate physical instances but are never treated as ordered numeric
strategy features.

## Models

### A0 — faithful 0010 control

A0 uses the original 7,154,562-parameter architecture: `d_model=320`, four pre-norm state
Transformer layers, eight heads, FFN width 960, one option cross-attention block, source/target
entity gathers, and the GRU pointer/STOP decoder. The raw dataset and training runtime change; the
model and input view do not silently gain capacity or information.

### AC — all-feature causal policy

AC retains the A0 board trunk as the primary information path.

- Card semantics enter their corresponding entity/option/resource token through normalized,
  zero-gated residual adapters.
- Public-zone inventory enters the state through a normalized global adapter.
- One resource token per registered card identity combines card capability, registered count,
  visible zones, deck/Prize knowledge and epistemic state; deck and ledger are not duplicated.
- A shallow causal event encoder produces event memory without mixing raw event tokens into the
  board sequence.
- Board-conditioned role queries retrieve from resource memory for setup/board development,
  attack/Prize progress, resource access/recovery, and tempo/survival.
- State and legal options receive retrieved context through separately normalized zero-initialized
  gates. Option queries remain bound to their source/target board entities.

This preserves the Goal-QKV idea while replacing 0013's raw-state query and uncontrolled token
concatenation. The value interface is not active during this BC project stage.

## Primary campaign

The primary sequence is:

1. `A0_0010_faithful_control`
2. `AC_all_features_lightweight_readers`

A1-A5 single-family runs, Goal-off, relation-off and capacity-matched controls are reserved for
diagnosis if AC regresses, becomes numerically unstable, or fails to use the added information.
Competition progress, not exhaustive academic attribution, determines allocation.

## R1: structured scenario conditioning

R1 is the first deliberately stronger feature-reader model. It keeps the same immutable V1 cache,
visibility contract, full-action labels and pointer output contract as A0/AC; it changes only the
model path after AC's compatible board encoder.

- Public zone inventories are two distinct player tokens; registered deck / causal-ledger entries
  remain individual resource tokens; event summary, known/unknown opponent-hand summary and four
  goal tokens are separate typed memory entries.
- A two-layer scenario-memory encoder contextualizes that actor-visible bank. The board state
  retrieves from it through state cross-attention; every legal action retrieves separately through
  option cross-attention.
- Each action produces a soft route over setup, prize, recovery and tempo goals; the route reads a
  role-specific value. Retrieved evidence conditions state and action representations with FiLM
  before the unchanged pointer scorer.
- Unlike AC's initially closed auxiliary residuals, R1's new conditioning starts at strength 0.10:
  it is stable but observable from the first update. The inherited AC gates remain present only as
  the compatible encoder substrate.
- Expected parameter count is 14,306,244. R1 is the model-family label; its first real training
  allocation is V10_r1_structured_causal_policy. Later structural proposals are R2/R3 while
  repository versions continue their strictly increasing V sequence.

### R1 measured intervention result

The fixed V10 epoch-25 checkpoint (`validation exact_action=0.853130`) was evaluated on the complete
validation split under one intervention at a time. This is an offline reliance diagnosis, not an
official-engine strength result. Disabling all new paths reduced exact action to `0.834385`.
Removing the resource ledger, Goal routing, card capability and public-zone families reduced exact
action by 1.658, 1.256, 0.786 and 0.297 percentage points respectively. Removing R1's state and
option conditioning reduced it by only 0.031 and 0.124 points; event and opponent-hand memory were
negligible at this checkpoint. The canonical payload is
`analysis/V10_epoch25_feature_ablation.json`.

This demonstrates that the frozen feature data is used, but R1's low-strength global conditioning
does not materially change the learned policy family. It motivates a strong test rather than
another capacity-only expansion.

Official-engine revision-7 evaluation confirmed the overfitting risk: epoch 7 loss-best won
150/200 (75.0%), while epoch 32 exact-best won 145/200 (72.5%), both with zero errors. Epoch 7 is
the selected R1 self-contained archive. Offline exact-action improvement did not imply stronger
play.

## R2: strong independent Scenario ScaleGate

R2 keeps the dataset, visibility contract, AC-compatible board trunk, Goal roles and pointer action
contract fixed. It changes how actor-visible scenario evidence reaches state and legal actions.

- **Scenario facts** are four separately identifiable families: two public-zone inventory tokens,
  registered-card causal-ledger tokens, causal-event summary, and known/unknown opponent-hand
  summary. Card capability annotates entities/resources but is not itself a scenario family.
- A two-layer scenario encoder contextualizes only those facts. It produces separate zone, ledger,
  event and hand summaries rather than an uncontrolled mean, then a fused scenario summary.
- Goal-QKV remains a separate planning path. Its four board-conditioned role queries retrieve from
  the registered-card resource memory; Goal outputs never enter the Scenario ScaleGate input.
- State and every legal option independently cross-attend the scenario bank. Their evidence is
  converted into FiLM residuals.
- Independent MLP ScaleGates use only actor-visible scenario-derived summaries and produce
  `2 * sigmoid(logit)`, hence positive per-channel scales in `(0, 2)`. Option scales are also
  per-option. Final layers are zero-initialized, so the formal run begins at scale `1.0`, ten times
  R1's initial `0.10`, while gradients can immediately suppress or amplify each channel.
- Expected parameter count is 17,387,842. `V11_r2_strong_scenario_scalegate` was allocated but
  failed after one train batch because its external background PTY closed stdout (`EIO`); local
  metrics and status preserve that launcher failure. No model defect was observed. The real formal
  training allocation is `V12_r2_strong_scenario_scalegate`, launched from epoch zero through a
  persistent session with a local `tee` log.

R2 is deliberately the strong hypothesis. If validation improves, the next version may move the
same scenario mechanism earlier or increase role-specific interaction. If it regresses, the next
version reduces the initial scale or limits strong conditioning to the ledger/Goal-supported option
path; no dataset rebuild is required.

At epoch 10, full-validation ScaleGate diagnosis found state scale mean `0.0327` and median
`0.00092`, versus legal-option scale mean `0.347`, median `0.262`, P90 `0.789` and P99 `1.336`.
The learned R2 path is therefore action-selective; R4 should allocate capacity to option-centric
family routing rather than another global state gate.

V12 stopped at epoch 18 under the patience-5 rule. Epoch 8 was validation-loss best (`0.198270`)
and won 158/200 official-engine games (79.0%); epoch 13 was greedy-exact best (`0.859441`) and won
159/200 (79.5%), both with zero errors. Epoch 13 is the selected R2 self-contained archive and is
the strongest 0014 official-engine result at this point.

## R3: typed early scenario fusion

R3 is an independent hypothesis designed before observing R2's stopping result, so it can begin as
soon as R2 releases the GPU. It tests *where* feature interaction should happen rather than another
post-trunk gate strength.

- A shallow scenario encoder builds typed zone, registered-card ledger, event and known/unknown-hand
  tokens with an explicit padding mask.
- Those tokens join CLS and faithful board entities inside every layer of the original board
  Transformer. Thus board CLS and physical entity representations become scenario-aware before
  option source/target entity gathering.
- Official card capability is always present in entity and option embeddings from the first update;
  R3 does not inherit AC's initially closed semantic gate.
- Every option cross-attends the combined board-plus-scenario sequence. Goal-QKV reads the
  contextualized registered-card token slice, and its option residual is always open.
- The pointer decoder, ordered full-action objective, dataset and visibility contract are unchanged.

R3 has 10,710,722 parameters and passed a concurrent small-batch CUDA BF16 optimizer/validation
smoke. Its reserved formal allocation is `V13_r3_early_scenario_fusion`, with the same 100-epoch
ceiling and patience-5 rule. R2 results will design R4, not retroactively modify R3.

V13 stopped at epoch 20 under the patience-5 rule. Epoch 7 was validation-loss best
(`loss=0.221961`, `exact_action=0.839458`) and won 155/200 official-engine games (77.5%). Epoch 15
was greedy-exact best (`loss=0.255715`, `exact_action=0.847068`) and won 157/200 (78.5%). Both had
zero errors and 100% completion. Epoch 15 is archived, but R3 trails the option-centric R2 by
1.0 percentage point and its rising validation loss shows that early scenario fusion is not the
preferred interaction boundary.

## R4: option-centric evidence-family routing

R4 is derived from R2's measured ScaleGate behavior: global state conditioning collapsed while
legal-option conditioning remained broad and strongly nonconstant. It keeps the faithful/AC board
substrate but freezes duplicated global-zone, state-auxiliary, option-semantic and option-auxiliary
AC gates closed. Six explicit routes condition legal options:

1. option-to-ledger attention over registered-card resource tokens;
2. four-role Goal-QKV read;
3. source/target official card capability;
4. option-to-public-zone attention;
5. option-conditioned causal-event context;
6. option-conditioned known/unknown opponent-hand context.

Each family has an independent projection and `2 * sigmoid(family_mlp(evidence))` per-channel
ScaleGate. Ledger, Goal and card routes initialize at 1.0; zone at 0.5; event and hand at 0.25.
This preserves lower-prior families without averaging them into the high-value ledger route. The
six routed deltas are variance-normalized before the unchanged pointer decoder. R4 has 14,720,642
parameters and passed a concurrent CUDA BF16 optimizer/validation smoke. Its reserved formal
allocation is `V14_r4_option_family_router`; it began immediately after V13 stopped and a
batch-256 BF16 gate passed.

V14 stopped at epoch 18 under the patience-5 rule. Epoch 7 was validation-loss best
(`loss=0.240858`, `exact_action=0.829869`) and won 139/200 official-engine games (69.5%). Epoch 13
was greedy-exact best (`loss=0.271302`, `exact_action=0.841809`) and won 151/200 (75.5%). Both had
zero errors and 100% completion. Epoch 13 is archived. The six always-added deltas substantially
regressed real play, so R4 is evidence against unconditional family summation, not against the
underlying actor-visible features.

## R5: option-query typed scenario memory

R5 is derived from R3, not R4. It keeps the useful typed-memory hypothesis but removes the path
that made CLS and every physical entity scenario-aware inside all board Transformer layers.

- The faithful board/entity trunk never receives scenario tokens. Aggregate global-zone,
  state-auxiliary and duplicate option-auxiliary AC gates are frozen closed; entity/option card
  capability adapters remain trainable.
- A one-layer typed memory encoder contextualizes two public-zone tokens, registered-card causal
  ledger tokens, causal-event summary and known/unknown opponent-hand summary with an explicit
  actor-visible mask.
- Every legal option is the query in cross-attention over this memory. Four-role Goal-QKV remains
  an independent read from the raw registered-card resource bank.
- An option-conditioned MLP produces a delta from `(option, scenario_read, goal_read)`. A separate
  MLP produces `2 * sigmoid(gate(option, scenario_read))`, giving a positive per-option,
  per-channel scale in `(0, 2)` initialized at 1.0.
- State, pointer decoder, ordered-action loss, frozen dataset and visibility contract do not
  change.

R5 has 12,664,002 parameters. CUDA BF16 forward/backward/optimizer and validation smoke passed.
Its reserved formal allocation is `V15_r5_option_query_scenario`; it is ready to start immediately
when V14 stops. It passed a batch-256 BF16 gate and began immediately after V14 stopped. R4 results
design R6 rather than retroactively altering R5.

V15 stopped at epoch 20 under the patience-5 rule. Epoch 9 was validation-loss best
(`loss=0.206352`, `exact_action=0.849913`) and won 156/200 official-engine games (78.0%). Epoch 15
was greedy-exact best (`loss=0.242995`, `exact_action=0.858451`) but won only 145/200 (72.5%). Both
had zero errors and 100% completion. Epoch 9 is archived. R5 confirms that option-query-only typed
memory is viable, but its strong residual still learns a late offline-exact shortcut that harms
real play.

## R6: sparse family mixture with null route

R6 is derived from R4, not R5. It retains specialized evidence readers while replacing the
unconditional sum of six residuals with per-option competition.

- Ledger attention, four-role Goal-QKV, public-zone attention, event context and known/unknown-hand
  context each produce one expert value. Card capability is removed as a duplicate expert and
  restored to the local entity/option identity adapters.
- A scorer for each family observes `(option, family_evidence)`. An explicit null/base scorer
  observes the option alone. Softmax across null plus five families makes route allocation bounded
  and auditable.
- Initial probabilities are null 0.50, ledger 0.20, Goal 0.15, zone 0.075, event 0.0375 and hand
  0.0375. Thus new evidence is available immediately without forcing every family into every
  action.
- A separate `2 * sigmoid` amplitude gate, initialized at 1.0, scales the weighted expert mixture
  before a normalized option residual. Global state is not rewritten.

R6 has 14,826,888 parameters and passed CUDA BF16 forward/backward/optimizer plus validation
smoke. Its reserved allocation is `V16_r6_sparse_family_mixture`; it will start immediately when
V15 stops. It passed a batch-256 BF16 gate and began immediately after V15 stopped. V15 results
design R7 rather than retroactively altering R6.

## R7: regularized low-prior option-query memory

R7 is derived from R5, not R6. It preserves the successful option-query boundary and directly
targets the Epoch 9-to-15 real-play collapse.

- Typed scenario memory remains outside the board/entity trunk and Goal-QKV remains independent.
- Route FFN multiplier is reduced from 2 to 1, lowering capacity from 12,664,002 to 11,946,562
  parameters.
- Scenario-attention and Goal evidence each receive 0.20 training-only dropout before the option
  residual. Inference remains deterministic and uses all evidence.
- The positive per-option, per-channel `2 * sigmoid` scale initializes at 0.35 rather than 1.0.
  This prior comes from R2's measured legal-option scale mean of 0.347, rather than an arbitrary
  weakening.

R7 passed CUDA BF16 forward/backward/optimizer plus validation smoke. Its reserved allocation is
`V17_r7_regularized_option_query`; it started immediately when V16 stopped. V16 results designed
R8 rather than retroactively altering R7.

## R8: faithful R2 option-only conditioning

R8 is derived from R6's result, not from the still-running R7. R6 showed that a more elaborate
family router can improve offline fitting while still creating a real-policy shortcut. R8 therefore
stops adding routing machinery and returns to the information flow of the strongest R2 result.

- It preserves R2's two-layer typed scenario bank, legal-option scenario attention, independent
  Goal-QKV, family-summary FiLM and dynamic positive option ScaleGate initialized at 1.0.
- It deletes the complete global state scenario-attention/FiLM/ScaleGate branch rather than merely
  gating it off. R2 diagnostics justify that removal: state scale mean was 0.0327 and median was
  0.00092, while the option scale mean was 0.347.
- The faithful 0010/AC board state is passed through unchanged; all new causal evidence may affect
  action selection only through the option representation.
- The resulting model has 15,128,322 parameters. CPU teacher-forced and deterministic-decode smoke
  passed with finite logits and legal actions; the batch-256 CUDA BF16 optimizer gate is deferred
  until V17 stops so validation does not perturb the active formal run.

Its reserved allocation is `V18_r8_r2_option_only`. Under the offset pipeline, R7's result will
design R9; it will not retroactively alter the ready R8 implementation.

R8 passed the batch-256 CUDA BF16 optimizer gate and started immediately after R7 stopped. Its
formal run is `V18_r8_r2_option_only`.

## R9: narrow strong option-query conditioning

R9 is derived from the completed R7 official-engine comparison and does not modify R8. R7's
epoch-8 validation-loss best checkpoint scored only 144/200 (72.0%), while epoch 14 recovered to
158/200 (79.0%) after its validation loss had already worsened. The low 0.35 scale and 0.20
evidence dropout therefore delayed useful scenario conditioning rather than protecting real play.

- R9 retains R7's one-layer scenario encoder, FFN multiplier 2 and route FFN multiplier 1.
- Evidence dropout is removed, so training and inference both observe the complete causal evidence.
- The dynamic positive option ScaleGate returns to an initial scale of 1.0.
- The board/entity trunk remains untouched; scenario and Goal evidence remain legal-option query
  inputs only.

R9 has 11,946,562 parameters. CPU teacher-forced and deterministic-decode smoke passed with an
exact initial scale of 1.0, finite logits and legal actions. Its reserved allocation is
`V19_r9_narrow_strong_option_query`; R8 results will design R10 rather than alter R9.

R9 passed the batch-256 CUDA BF16 optimizer gate and started immediately after R8 stopped. Its
formal run is `V19_r9_narrow_strong_option_query`.

## R10: bounded global-state calibration

R10 is derived from the completed R8 official-engine comparison and does not modify R9. Removing
R2's low-amplitude state branch reduced official-engine win rate by 3.5–4.0pp despite comparable
offline exact action. The branch therefore carries useful contextual calibration even though its
measured R2 scale was small.

- R10 restores the complete R2 state reader and unchanged strong option path.
- The state ScaleGate is explicitly bounded to `(0, 0.25)` and initializes at 0.05, close to R2's
  measured mean 0.0327. It can calibrate the board state but cannot dominate the faithful trunk.
- The option ScaleGate remains `(0, 2)`, initialized at 1.0, preserving R2's successful action-level
  evidence flow.
- Scenario depth, Goal-QKV, FiLM and all full-feature inputs remain identical to R2.

R10 has 17,387,842 parameters. CPU teacher-forced and deterministic-decode smoke passed with state
initial scale 0.05, finite logits and legal actions. Its reserved allocation is
`V20_r10_bounded_state_calibration`; R9 results will design R11 rather than alter R10.

R10 passed the batch-256 CUDA BF16 optimizer gate and started immediately after R9 stopped. Its
formal run is `V20_r10_bounded_state_calibration`. It stopped at epoch 18: epoch 8 loss-best
reached 150/200 (75.0%), while epoch 13 exact-best reached 155/200 (77.5%) and was archived.
Bounding the state scale recovered 2.0pp over deleting the state path, but remained 2.0pp below
the unbounded R2 result.

## R11: strong-scale option query with evidence dropout

R11 is derived from the completed R9 official-engine comparison and does not modify R10. R9's
full-evidence, initial-scale-1.0 reader reached only 153/200 (76.5%) at both loss-best and
exact-best, while R7's otherwise identical narrow reader with evidence dropout eventually reached
158/200 (79.0%). Dropout therefore contributed real-policy robustness, although R7's low 0.35
scale delayed its usefulness.

- R11 keeps the one-layer scenario encoder, FFN multiplier 2 and route FFN multiplier 1.
- Training-only scenario/Goal evidence dropout is restored to 0.20.
- The dynamic option ScaleGate remains strong at initialization, with scale 1.0 in `(0, 2)`.
- The board/entity trunk remains untouched and inference remains deterministic with full evidence.

R11 has 11,946,562 parameters. CPU teacher-forced and deterministic-decode smoke passed with exact
initial scale 1.0, dropout 0.20, finite logits and legal actions. Its candidate exporter and package
validation also passed. Its reserved allocation is `V21_r11_strong_dropout_option_query`; R10
results will design R12 rather than alter R11.

R11 passed its batch-256 CUDA BF16 optimizer gate and started immediately after R10 stopped. Its
formal run is `V21_r11_strong_dropout_option_query`. It stopped at epoch 19: epoch 10 loss-best
reached 143/200 (71.5%), while epoch 14 exact-best reached 151/200 (75.5%) and was archived.
Restoring dropout while retaining the unit option-scale prior did not reproduce R7's robustness.

## R12: state-preserving option evidence dropout

R12 is derived from the completed R10 official-engine comparison and does not modify R11. The
ordered state-path evidence is now consistent: deleting it produced 75.5% at exact-best, bounding
it to `(0, 0.25)` produced 77.5%, and the original unbounded R2 produced 79.5%. The next model
therefore restores R2's complete state calibration rather than treating its small final scale as
proof that it can be deleted or permanently suppressed.

- The state reader, FiLM residual and `(0, 2)` ScaleGate are exactly the R2 structure, initialized
  at 1.0 and trained without evidence dropout.
- The complete R2 option reader and independent Goal-QKV path remain intact.
- Training-only dropout 0.20 is applied only to option scenario attention and option Goal evidence,
  where R7 versus R9 provides evidence of shortcut regularization.
- Evaluation and exported inference are deterministic and consume the complete actor-visible
  evidence; no feature is removed from the dataset or runtime encoder.

R12 has 17,387,842 parameters. Compile, CPU teacher-forced logits, greedy decode, finite-output,
temporary checkpoint export and candidate package validation passed. Its reserved allocation is
`V22_r12_state_preserving_option_dropout`; R11 results will design R13 rather than alter R12.

R12 passed its batch-256 CUDA BF16 optimizer gate and started immediately after R11 stopped. Its
formal run is `V22_r12_state_preserving_option_dropout`. It stopped at epoch 26: epoch 12
loss-best reached 154/200 (77.0%), while epoch 22 exact-best reached 150/200 (75.0%). The loss-best
checkpoint was archived. Late exact-action optimization again reduced official-engine strength.

## R13: strong state with gradual option scale

R13 is derived from the completed R11 official-engine comparison and does not modify R12. Within
the narrow option-only family, R11's dropout 0.20 plus initial scale 1.0 reached 75.5%, whereas
R7's dropout 0.20 plus initial scale 0.35 reached 79.0%. This isolates an interaction between
evidence dropout and an overly strong initial option residual.

- The complete R2 state reader remains unbounded in `(0, 2)` and initializes at 1.0.
- The option scenario/Goal evidence keeps training-only dropout 0.20.
- Only the option ScaleGate initialization changes from 1.0 to 0.35; it retains the full `(0, 2)`
  learnable range and full actor-visible evidence at inference.
- Dataset, online encoder, Goal-QKV, FiLM, action decoder and every feature contract remain fixed.

R13 has 17,387,842 parameters. Compile, CPU teacher-forced and greedy paths, exact initial option
scale 0.35, finite outputs, temporary checkpoint export and candidate validation passed. Its
reserved allocation is `V23_r13_gradual_option_scale`; R12 results will design R14 rather than
alter R13.

R13 passed its batch-256 CUDA BF16 optimizer gate and started immediately after R12 stopped. Its
formal run is `V23_r13_gradual_option_scale`. It stopped at epoch 18: epoch 9 loss-best reached
145/200 (72.5%), while epoch 13 exact-best reached 153/200 (76.5%) and was archived. Lowering the
option prior improved validation loss but did not rescue elementwise evidence corruption.

## R14: structured scenario-only modality dropout

R14 is derived from the completed R12 official-engine comparison and does not modify R13. R12
applied ordinary elementwise dropout to both option scenario attention and Goal-QKV role evidence.
Its 77.0% peak remained 2.5pp below R2, and the exact-best checkpoint regressed to 75.0%. For
actor-visible game state, randomly corrupting individual channels of a precise resource/planning
vector is not equivalent to teaching robustness to missing evidence.

- The complete R2 state path and Goal-QKV path remain deterministic during training and inference.
- Only the option scenario-attention route is regularized.
- One Bernoulli mask is shared by every option and channel in a sample. The whole scenario route
  is either present or absent, preserving option consistency and within-vector geometry.
- Kept routes use inverted-dropout scaling; evaluation always consumes complete scenario evidence.
- Option ScaleGate remains in `(0, 2)` and initializes at 1.0. Dataset, encoder, FiLM and action
  contract remain unchanged.

R14 has 17,387,842 parameters. Compile, CPU teacher-forced and greedy paths, structured-mask
semantics, finite outputs, temporary checkpoint export and candidate validation passed. Its
reserved allocation is `V24_r14_scenario_modality_dropout`; R13 results will design R15 rather
than alter R14.

R14 passed its batch-256 CUDA BF16 optimizer gate and started immediately after R13 stopped. Its
formal run `V24_r14_scenario_modality_dropout` stopped at epoch 15. Epoch 8 loss-best reached
147/200 (73.5%); epoch 10 exact-best reached 148/200 (74.0%) and was archived. Sharing one mask
across all options and channels did not fix the core problem: actor-visible scenario evidence is
needed consistently for exact turn-budget and relay decisions. Further evidence-dropout variants
are therefore out of scope.

## R15: deterministic full evidence with gradual option scale

R15 is derived from the completed R13 official-engine comparison and does not modify R14. R13's
initial option scale 0.35 reduced validation loss to 0.19976, but channelwise dropout still left
the official-engine peak at 76.5%. R15 keeps the gradual prior while removing the remaining
evidence corruption, isolating whether the initial option residual strength itself is useful.

- State ScaleGate is the full R2 `(0, 2)` path initialized at 1.0.
- Option ScaleGate remains learnable in `(0, 2)` but initializes at 0.35.
- Scenario attention, Goal-QKV, family summary and every feature remain deterministic and fully
  visible throughout training and inference.
- Dataset, online encoder, FiLM, model capacity and action contract remain fixed.

R15 has 17,387,842 parameters. Compile, CPU teacher-forced and greedy paths, exact initial option
scale 0.35, finite outputs, temporary checkpoint export and candidate validation passed. Its
reserved allocation is `V25_r15_deterministic_gradual_option`; R14 results will design R16 rather
than alter R15.

R15 passed its batch-256 CUDA BF16 optimizer gate and started immediately after R14 stopped. Its
formal run `V25_r15_deterministic_gradual_option` stopped at epoch 19 with W&B online mirroring
fully synced. Epoch 8 loss-best reached 148/200 (74.0%); epoch 14 exact-best reached 165/200
(82.5%), was archived, and replaced R2's 79.5% as the observed project SOTA. The 8.5pp gap between
the two R15 checkpoints again shows that validation loss is not a sufficient policy selector.

## R16: deterministic typed rule-contract option reader

R16 is derived from the completed R14 official-engine comparison and does not modify R15. R14's
structured scenario-only dropout peaked at 74.0%, 5.5pp below R2. R16 therefore preserves every
deterministic R2 evidence route and adds a shorter, explicitly typed path for rule-relevant facts
that otherwise must be rediscovered from a single aggregate board/scenario representation.

- Four tokens are built only from actor-visible current-observation tensors: `turn_budget`
  (selection phase/context, used-once flags and remaining selection counters), `prize_race`
  (Prize, deck, hand and board totals), `library_pressure` (deck/hand/discard and known/unknown
  hand-zone counts), and `board_relay` (Active/Bench, Energy, Tool, evolution, damage and capacity).
- A one-layer Transformer lets these four contracts interact without mixing them into the faithful
  0010 entity trunk or replacing the existing R2 scenario and Goal-QKV routes.
- Each legal option independently queries the rule-token bank. An independent positive per-channel
  ScaleGate in `(0, 2)`, initialized at 1.0, controls the resulting option residual.
- The four token names are representation roles, not hard-coded action scores. The BC target still
  determines whether an attack, Supporter, Energy attachment, evolution, retreat or END is preferred.
- No opponent hidden card identity, future state, reward, matchup prior or post-action information
  enters the tokens. General official rules motivate the grouping; current card text and engine
  runtime still define legality; learned action value remains a project policy hypothesis.

R16 has 20,699,202 parameters. Compile, CPU teacher-forced and deterministic decode, finite logits,
legal actions, exact initial rule scale 1.0, temporary checkpoint export and self-contained candidate
validation passed. Its reserved allocation is `V26_r16_rule_contract_option_reader`; R15 results
will design R17 rather than alter R16.

The user ended further model exploration while V26 was in epoch 1. The process was interrupted,
its failed/interrupted runtime record and W&B identity are retained for audit, and no R16 checkpoint
is eligible for SOTA comparison. Selection is restricted to completed, officially evaluated runs.

## Training and efficiency contract

0014 retains the accepted training-runtime optimizations through local 0014 code:

- exactly one train update pass per epoch;
- online teacher-forced optimization loss/token/exact diagnostics from the backward logits;
- no static full-train evaluation and no train-time greedy decode;
- complete epoch-end teacher-forced plus greedy validation;
- one shared validation encoding, on-device accumulation, inference mode and BF16 CUDA AMP;
- pinned host batches, non-blocking transfers and bounded prefetch;
- no option permutation for A0 because 0010 has an option-position embedding; any AC permutation
  is a separately declared augmentation with ordered-target remapping;
- canonical `training_metrics.jsonl`, then TensorBoard, then failure-isolated W&B online mirror.
- terminal TQDM reports train batches, running loss and decisions/s at a bounded refresh interval;
  batch progress remains in canonical JSONL/TensorBoard and appears in W&B console logs, while
  W&B scalar history receives one complete record per epoch. Thus both W&B `_step` and
  `trainer/epoch` are 0, 1, 2, ... and greedy exact-action is emitted exactly once per complete
  validation epoch, never as an extra train pass.
- Formal exploratory runs declare a 100-epoch ceiling but stop after five consecutive complete
  validation epochs without at least `0.001` improvement in greedy exact action. This is a compute
  allocation rule, not evidence that offline exact action equals policy strength.

Materialization groups rows by episode-player. Chronology is serial within a group and groups are
processed in parallel. Worker count, shard size, queue depth, batch size and prefetch are benchmarked
on this machine before the immutable build; the fastest deterministic configuration wins. GPU time
is reserved for tensor/model throughput unless a measured preprocessing kernel proves beneficial.

## A0 gates and measured runtime

- Full visibility scan passed on 162,128 rows. Logs are incremental; 22,652/22,652 nonempty full
  deck views match deckCount and permit conservation-based own Prize recovery with complete visible
  zones (`looking`, `contextCard`, and `effect` included).
- Faithful codec/collate parity is exact; legacy FP32 numerics and dynamic E/O/target widths are
  preserved. Original and A0 control teacher logits are bitwise equal under the same state dict;
  batched greedy equals original scalar greedy.
- A0 parameter count is exactly 7,154,562 and the A0 loader returns only the ten legacy model keys.
- CPU compiler benchmark selected 8 workers (1/4/8: 1443.5/1728.0/1859.8 decisions/s on 9,532
  decisions). BF16 optimizer benchmark selected batch 256; observed peak allocation was 1.70 GiB.
- A 20-train-batch + 4-validation-batch noncanonical BF16 smoke completed with finite loss,
  100% greedy legality, shared validation encoding and no GradScaler.
- W&B credentials were verified for `dragon_bra`; formal metrics remain local-first and mirrored
  online to private project `pokemon-tcg-policy-learning`. The run name contract is
  `0014 · faithful_board_causal_features · <version_name>`, grouped by the full project ID.

Offline imitation metrics do not establish playing strength. Candidate export and official-engine
evaluation happen only after training and package validation; promotion remains user-controlled.
