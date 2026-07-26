# 0014 Faithful Board Causal Features

**Project ID:** `0014_faithful_board_causal_features`  
**Status:** feature/visibility audits passed; the immutable full-feature cache and A0 parity/BF16
smoke gates passed. V1 six-epoch control completed; V8 is the active from-zero 100-epoch A0 run
with validation greedy-exact early stopping. No policy-strength result exists yet.

## Purpose

0014 asks two practical competition questions on one frozen single-expert corpus:

1. How well does the faithful 0010 ID-only pointer model learn on the newer Yushin Ito raw dataset
   and split?
2. After restoring every 0010 board/option feature, does adding complete actor-visible card,
   public-zone, registered-deck, causal resource, and event information produce a stronger policy?

The project does not treat the incomplete 0013 M5 representation as a base. It reuses 0013's raw
data, causal/audit lessons, materialization architecture, BF16 training optimizations and W&B
contract, while rebuilding the feature and model-reader contracts.

## Frozen source and labels

- Raw dataset: `../../rl_runs/0013_semantic_goal_policy/dataset/V1_causal_semantic_v1`
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

## Training and efficiency contract

0014 reuses the accepted 0013 optimized runtime:

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
