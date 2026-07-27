# 0016 Alakazam Multideck BC

**Project ID:** `0016_alakazam_multideck_bc`  
**Current phase:** formal V1 complete; formal V2 R15 comparison training  
**Deployment target:** the frozen 60-card Battle Cage + Wondrous Patch deck in
`train/0016_alakazam_multideck_bc/deck.csv`

## Evidence boundary

- Official rules: attacks end the turn; Supporter, manual Energy, Retreat and Stadium use have
  their official once-per-turn constraints. These rules are summarized in the repository rules
  evidence and enforced for legality by the official runtime.
- Current card/runtime facts: exact Card IDs, legal full-action options, registered decks and card
  effects come from current official card data and replay/runtime payloads.
- Project hypothesis: winning demonstrations from several Alakazam experts will improve a policy
  deployed with a new exact construction. Offline imitation does not prove playing strength.

## Data contract

The primary dataset keeps only complete official Episodes where the selected Top-100 source player
actually registered an Alakazam deck and won. Every trajectory records exact Unicode team identity,
source ID, Episode/player index, payload hash, outcome, exact 60-card multiset and whole-Episode
split. Different teams are never mixed without the explicit source-persona input.

`goonew` submission `54960905` is the targeted Wondrous Patch source. Its frozen collection has
166 validated winning replays; every own deck contains exactly one `Wondrous Patch` (`1146`).
Battle Cage coverage is collected naturally from the frozen Top-100 Alakazam cohort and remains an
audited dataset statistic rather than a per-Episode inclusion filter.

The frozen cache contains 760,106 train decisions and 85,299 validation decisions; all 845,405
validated decisions are eligible for training. Its content commitment is
`59dddebe35e3217e4d1a508d23c1720644f908bfa1b611eafad17594ad4301b0`. Across the full raw pool,
Battle Cage is legal in 2,500 decisions and selected in 257. Wondrous Patch is legal in 283,
selected in 76, and produces 114 audited follow-up target decisions. These are action-level
coverage diagnostics, not strength metrics.

## Model

The backbone is a physical, self-contained freeze of 0014
`V12_r2_strong_scenario_scalegate`, trained from a fresh seed:

- width 320, 8 heads, four board Transformer layers and base FFN multiplier 3;
- full card/entity/option encoding plus card capabilities, zone inventory, causal resource ledger,
  event memory, known/unknown opponent-hand boundary and exact registered-deck tokens;
- four independent Goal-QKV roles;
- two scenario Transformer layers with FFN multiplier 3;
- state and option ScaleGates in `(0, 2)`, initialized to neutral scale `1.0`;
- 17,387,842 parameters in the unchanged R2 backbone.

Because the dataset contains several team policies, 0016 adds one declared source-persona residual
after the R2 state/option encoder. Source ID 0 is a zero-neutral deployment token; known sources use
a small embedding projected independently into state and options with initial scale `0.05`. This is
the only intentional structural delta from V12 R2.

Formal V2 adds a project-local adaptation of 0014 R15 without importing 0014 at runtime. It keeps
the complete R2 state and option evidence paths, all 17,601,282 source-conditioned parameters and
the state ScaleGate initial value `1.0`. Its only model variable relative to V1 is the option
ScaleGate initial value, reduced from `1.0` to `0.35`; the gate remains learnable and bounded in
`(0, 2)`. Inputs, tensor shapes, source vocabulary, Goal-QKV roles and output heads are unchanged.

The action head remains the autoregressive legal-option pointer over ordered full actions. The raw
contract, cache, teacher-forced objective and online greedy decoder all support action sequences up
to length 64. This changes the V12 action horizon without changing its parameterized backbone;
validated 1-64 step decisions are retained and trained rather than filtered by the old 16-step cap.
The frozen 10,872-Episode audit contains 845,405 decisions: observed maxima are action length 26,
`minCount=26`, `maxCount=26` and 50 legal options. Exactly 360 decisions (0.043%) exceed the old
16-step horizon and none exceeds 32. The official API declares only
`maxCount <= len(option)`, not a numeric 64 cap; therefore inference also has a capacity-independent
count-valid fallback for future observations outside the trained option/action envelope.

## Objective and training

The objective is ordinary full-action behavioral-cloning cross entropy. Each epoch performs exactly
one update pass over the train split and one fixed-model teacher-forced plus greedy pass over the
complete validation split. Defaults remain AdamW, learning rate `3e-4`, weight decay `0.02`, batch
256, validation batch 512, BF16, seed `20260723`, and early stopping patience 5/min delta `0.001`.
Training metrics are canonical JSONL, then TensorBoard, then W&B online under the private repository
project.

Formal runs are resumable only inside the same immutable repository version and stable W&B run. Resume
requires matching checkpoint SHA, version, dataset commitment, model/feature config, optimizer
state, batch sizes, learning rate, seed and early-stopping contract. After the host power loss during
partial epoch 9, the last complete boundary is epoch 8/global step 23,760; partial epoch 9 progress
is discarded and epoch 9 is replayed from its beginning. The original epoch-8 checkpoint predates
RNG persistence, so this recovery preserves weights, AdamW moments and deterministic epoch data
order but not the exact dropout stream. All checkpoints written after recovery include Python,
NumPy, Torch and CUDA RNG state.

Early stopping follows validation token cross-entropy rather than greedy exact-action accuracy.
Training preserves distinct `best_validation_loss`, `best_teacher_exact`, `best_greedy_exact` and
`latest` checkpoint pointers when their epochs differ. Exact-action metrics diagnose imitation; they
do not select the deployment policy by themselves. Frozen candidates from this declared set use the
same target deck, official engine, opponent catalog, seeds, seat schedule and metric profile, and
official-engine playing strength selects the final candidate.

Wins-only selection intentionally favors successful state distributions and is not an unbiased
value label. Outcome never becomes an inference feature or a per-action reward.

## Package boundary

Training/runtime code may share `rl_environment/`, `evaluation/`, official card data and the official
engine runtime. It must not import executable code from another numbered training project. Candidate
export defaults to the frozen 0016 target deck and copies portable policy code, ontology, model and a
physical `cg/` into a self-contained candidate. Controlled deck-comparison exports may provide an
explicit exact 60-card deck; the candidate manifest records that deck's source path and hash so the
checkpoint remains fixed and the deployment-deck variable is auditable. No candidate is admitted to
the opponent pool or submitted to Kaggle without explicit user authorization.

## Next gate

V1 completed at epoch 17 after early stopping; epoch 12 is its validation-loss best at
`0.2552687146`, with exact action `81.9283%` and legal action `100%`. Train
`V2_win_multideck_r15_gradual_option` from the same seed with the same cache, optimizer, batch,
validation and early-stopping contract. Preserve its loss-best, teacher-exact-best,
greedy-exact-best and latest checkpoints, then compare frozen V1/V2 candidates through controlled
official-engine evaluation. Offline loss alone does not select the stronger model.
