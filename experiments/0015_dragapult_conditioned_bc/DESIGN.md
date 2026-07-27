# 0015 Dragapult Conditioned BC

**Project ID:** `0015_dragapult_conditioned_bc`
**Status:** 0726 ingestion/materialization and V1–V5 are complete. All five formal runs are
W&B-synced and have official-engine reports; V5 also has an independent confirmation run.
**Current phase:** V2 remains the strongest supported candidate. Preserve the genuine-source gate
for T2/T3 and do not continue tuning terminal-outcome weights. Value/RL remains deferred.

**Open T2 data gate:** the 0726 registration-frame audit found no genuine Starmie + Dusknoir deck.
The previous environment label was a hard-coded Mega Starmie display heuristic even when the exact
deck contained Froslass instead. T2/T3 formal allocation waits for real card-list evidence; see
[`decisions/003_starmie_dusknoir_source_gap.md`](decisions/003_starmie_dusknoir_source_gap.md).

## 1. Research question

High-quality Dragapult demonstrations are scarce. The target build, Dragapult + Dusknoir
(“自爆多龙”), needs two reusable forms of knowledge:

```text
Pure Dragapult ────── teaches Dragapult attack/relay module ──┐
                                                             ├─> Dragapult + Dusknoir target
Starmie + Dusknoir ── teaches Dusknoir setup/blast module ───┘
```

The experiment asks whether demonstrations from adjacent builds improve imitation of the target
without washing incompatible labels together. The intended policy is

\[
\pi(a \mid s, D, E),
\]

where `s` is the actor-visible action-time state, `D` is the exact registered own-deck multiset and
`E` is an auditable expert/source persona. At target deployment `E` is fixed to the target teacher
persona. This is deck/source-conditioned play, not automatic deck construction.

## 2. 0014 readiness audit

The selected 0014 baseline is **R15**, specifically
`rl_runs/0014_faithful_board_causal_features/versions/V25_r15_deterministic_gradual_option`.
The earlier architectural verdict was **yes for the registered-deck AC/R1 family, no for A0**. The
full evidence is recorded in
[`decisions/001_0014_registered_deck_audit.md`](decisions/001_0014_registered_deck_audit.md).

| Layer | Finding | Meaning for 0015 |
|---|---|---|
| Cache/compiler | `registered_card_ids` + aligned `registered_multiplicity` losslessly represent the exact 60-card multiset. | Pure, self-sacrificing and Starmie builds can be distinguished. |
| AC/R1 model | Registered cards and ledger state are consumed as resource/scenario memory and condition state/options. | Reuse this family and preserve the channel. |
| Online/export | The candidate's `deck.csv` is validated as 60 cards and encoded through the same fields. | Training and deployed package can share the contract. |
| A0 | Does not consume registered-deck features. | Excluded from the transfer experiment. |
| Existing evidence | V9 logits respond to a one-card count swap. | Wiring and learned sensitivity are proven, strategic generalization is not. |

The source-conditioned R15 is `R15DeterministicGradualOptionPolicy` plus a source-persona
embedding, with 17,416,642 parameters in the frozen 84-source configuration. It preserves the
complete
R2 state path and deterministic scenario/Goal-QKV option evidence, initializes state scale at
`1.0` and option scale at `0.35`, and enables card capability, event memory, resource ledger,
public inventory, registered deck and relation graph features. The 0015 optimizer defaults inherit
R15's batch 256, validation batch 512, learning rate `3e-4`, weight decay `0.02`, BF16, seed
`20260723`, and patience 5/min-delta `0.001` logic. All matrix arms start from the same fresh seeded
initialization; the Alakazam-trained R15 checkpoint is architectural evidence, not a Dragapult
warm-start.

The 0014 cache contains 162,128 decisions but only two signatures: 138,898 decisions (85.7%) use
Enhanced Hammer ×4 / Nighttime Mine ×2, while 23,230 (14.3%) use Hammer ×3 / Mine ×3. The V9
epoch-11 checkpoint changed its finite logits when only this multiplicity was switched
(max absolute delta `0.1878586`, mean `0.0823456`). This is reassuring, but a one-card imbalanced
variation inside one archetype cannot prove that the model understands three materially different
builds. That is precisely what 0015 must test.

### R15 typed rule-contract reader

V7–V13 preserve the complete source-conditioned R15 policy and add four actor-visible option-level
rule tokens: `turn_budget[10]`, `prize_race[16]`, `library_pressure[12]` and
`board_relay[19]`. One Transformer layer lets those roles interact; each legal option queries the
four-token bank, and a positive per-channel ScaleGate applies one residual before the unchanged
full-action decoder. No rule token contains future frames, terminal outcome, hidden opponent cards
or an opponent identity.

The full-width reader operates at width 320/eight heads, has 20,728,002 total parameters and adds
3,311,360 trainable parameters to a frozen V2 base. Warm-start is strict: all 271 inherited V2
tensors must load exactly. `--train-rule-only` freezes 17,416,642 parameters and permits updates to
only the 51 audited new tensors. V9 proved that updating the whole warm-started model even at LR
`1e-4` rapidly destroys V2; V10–V13 therefore use the frozen-base contract.

The implementation also supports a prepared low-rank reader without changing old checkpoints:
missing metadata reconstructs the historical width 320/eight-head topology exactly. Width 80/four
heads projects option, state and categorical evidence into the bottleneck and projects its residual
back to width 320. It has 18,054,162 total and 637,520 trainable parameters (80.7% fewer adapter
parameters). Its five-batch CUDA BF16 smoke, full validation, portable export and candidate
validation passed, and all 271 inherited tensors remained bitwise unchanged. This is a prepared
research checkpoint only. Its gate required V11 or V12 to beat V2 in official-engine games; that
gate failed, so the low-rank reader received no formal version. V14 was allocated independently to
the label-smoothing experiment.

## 3. Why Starmie + Dusknoir is useful

The common object is not “the whole policy”; it is a tactical subproblem. Dusclops/Dusknoir's
Cursed Blast places damage counters and KOs itself. Good use depends on evolution timing, target
value, Prize race, Bench space, the attack that follows and the post-KO relay. Starmie games can
provide far more examples of those decisions even though their primary attacker differs.

This transfer is plausible, not automatic. Starmie's damage geometry, energy plan and desired
targets differ from Dragapult's Phantom Dive. Therefore auxiliary labels are useful only when the
model sees the exact deck and evaluation isolates the shared Dusknoir decisions from attacker-
specific decisions. The official rules remain binding: attacking ends the turn; evolution and
once-per-turn budgets constrain whether a blast line can be completed.

## 4. Data and input contract

Every episode must bind:

- the exact 60-card multiset and stable deck hash;
- build family (`dragapult_dusknoir`, `pure_dragapult`, `starmie_dusknoir`);
- normalized expert/team policy source and replay provenance;
- seat, outcome, completeness, extraction version and audit status.

The policy receives only actor-visible state, legal full-action options, its complete own deck and
an explicit normalized expert/source ID. It never receives future frames, terminal outcome,
opponent hidden cards or preset opponent ID/deck. Source conditioning is required because the
training matrix combines labels from different teams; it is a declared persona control, not hidden
matchup information. The deployed target package fixes it to `THIRD PTCG Club`.

```text
actor-visible board/history + legal full-action options
                         │
                   state/option encoder
                         │
exact own 60-card multiset ──> registered-card + resource-ledger memory
expert/source ID ────────────> small source-persona embedding
                         │
              conditioned state and legal options
                         │
              full-action BC policy π(a|s,D,E)
```

Use the selected 0014 R15 registered-deck path. Preserve card identity, initial multiplicity, ontology,
known zones, remaining known/bounded resources and unknown boundaries. Unknown must not become zero.
Add only the source-persona path required for multi-team labels. It must be recorded in model and
dataset manifests and fixed to the target token in exported target candidates. An optional larger
adapter is a later fallback if shared training demonstrably harms a build; it must not replace
exact-deck conditioning.

## 5. Controlled T0–T3 matrix

All arms train the same target-deck policy contract and preserve the same complete target episode
pool/split. This is a **competition-oriented natural-data addition study**, not a compute-matched
causal study. Every arm uses every trajectory that passes the same audit. T1–T3 add all qualified
auxiliary trajectories without downsampling the target or enforcing artificial 50/50 quotas.

| Arm | Training demonstrations | Question |
|---|---|---|
| T0 | Dragapult + Dusknoir only | What can scarce target data learn by itself? |
| T1 | T0 + pure Dragapult | Does the Dragapult module transfer? |
| T2 | T0 + Starmie + Dusknoir | Does Dusknoir setup/blast knowledge transfer? |
| T3 | All three | Are the two modules compositional, complementary and non-destructive? |
| T3-no-deck | Same rows/order/epochs as T3; only exact initial-deck composition removed | Is any gain actually attributable to construct conditioning? |

Because the registration-frame audit found no genuine Starmie + Dusknoir source, T2/T3 remain
blocked rather than being populated with mislabeled Starmie + Froslass data. While that gate is
open, `T1-no-deck` is an explicitly exploratory ablation: it uses the exact T1 rows, ordering,
optimizer contract and target validation, but applies the same safe initial-deck removal specified
for T3-no-deck. It tests conditioning at the largest currently materialized natural union and must
not be reported as a substitute for T3-no-deck.

`T3-no-deck` keeps live board/resource-ledger tensors and the source-persona token unchanged. The
registered-card sequence uses a valid learned null/sentinel token with a nonempty mask and zero
multiplicity, so the ablation cannot create all-masked attention or NaNs. Tests must prove row,
label and order identity with T3 and finite forward/backward behavior.

Each epoch processes every selected decision once. T1–T3 will naturally contain more updates and
take longer than T0; this is accepted because the practical question is whether adding the complete
available source improves the competition policy. Report natural build/source counts, updates and
wall time, and describe any gain as a combined data-plus-compute result rather than pure causal
transfer. Train/validation/test still split whole episodes and prevent replay, expert/session or
duplicated-state leakage.

## 6. Using losing games without a value head

Scarcity may require complete losing episodes. Losing does not make every action wrong, so outcome
must not be copied onto individual actions as a reward or value label in 0015. Keep all audited
episodes initially and expose outcome as dataset metadata.

If loss-heavy noisy data overwhelms cleaner demonstrations, compare predeclared static BC weights
such as source quality and a bounded episode-outcome coefficient. Apply one episode-level weight to
the imitation loss; do not infer the critical mistake and do not add a value head. Always report
unweighted validation metrics and effective sample weight by build/outcome so weighting cannot hide
a collapsed stratum.

### Bounded outcome-weighted extension

V4/V5 test this BC-only weighting while leaving selected rows, model and unweighted validation
unchanged. For train logits `z ∈ R[B,T,C]`, targets `y ∈ Z[B,T]`, outcome IDs
`o ∈ {-1,0,1}^B`, and positive decision weights `w ∈ R[B]`, only optimization changes:

```text
w_i = 1.0                    if o_i = win
w_i = α                      if o_i = loss, α ∈ {0.75, 0.50}
w_i = 1.0                    if o_i = draw/unknown
L = Σ_i,t w_i · mask_i,t · CE(z_i,t, y_i,t) / Σ_i,t w_i · mask_i,t
```

No trajectory is dropped. The cache field `outcome_id: [B]` is consumed only by the training-loss
callback; it is not an inference input. Teacher/greedy validation, legality, checkpoint selection
and official-engine evaluation remain unweighted. Canonical metrics record weighted optimization
loss, mean decision weight and win/loss/draw decision counts each epoch.

### Legal-class label smoothing

V14 returns to the plain 17,416,642-parameter R15 T1 model and changes only train-time token CE to
epsilon `0.05`. The pointer decoder represents illegal/padded classes with the dtype minimum value,
so smoothing over the entire class vocabulary is mathematically invalid and produced a nonfinite
diagnostic smoke. The accepted objective distributes the smoothing mass only across the current
token's legal logits (finite and above half the dtype-minimum mask sentinel):

```text
L_i,t = (1 - ε) · NLL(y_i,t) + ε · mean_{c in legal(i,t)}[-log p(c)]
ε = 0.05
```

Padding is excluded, the labeled class must be legal, and epsilon must satisfy `0 <= ε < 1`.
Validation remains ordinary unsmoothed CE plus greedy full-action decoding. Model structure,
features, registered deck/source conditioning and exported runtime are unchanged from V2. The two
failed `.tmp` smokes are retained as diagnostics; the legal-only smoke completed five CUDA BF16
updates with finite loss and gradients before formal V14 started.

## 7. Target-only evaluation principle

The experiment has exactly one deployment target: Dragapult + Dusknoir. There is no deployable
teacher policy available for local engine evaluation, so no “student versus teacher win rate” can
be measured. **All success
claims are decided on that target build only**. Pure Dragapult and Starmie + Dusknoir are auxiliary
training sources. Their holdouts may diagnose whether the source modules were learned, but they do
not enter checkpoint ranking, the headline score or the go/no-go decision.

BC agreement and playing strength answer different questions:

- target validation agreement asks whether the policy reproduces the target teacher on logged
  teacher states;
- target official-engine wins ask whether the policy succeeds on the state distribution created by
  its own actions.

It is therefore valid for T1/T2/T3 to have slightly lower target exact-action agreement than T0 but
higher target real-game strength. That is potentially the most interesting result: auxiliary data
may have taught a more robust choice where several legal actions are strategically equivalent, or
may improve recovery in states the teacher holdout rarely contains. It must still survive a fresh
confirmation run; a noisy point estimate is not enough. The frozen contract is recorded in
[`decisions/002_target_only_evaluation_contract.md`](decisions/002_target_only_evaluation_contract.md).

## 8. Required offline diagnostics

Every epoch keeps the repository contract: one training pass, then fixed-model full validation with
teacher-forced and greedy full-action metrics. The target split is the sole checkpoint-selection
split and must report loss, token accuracy, greedy full-action exact, legality, phase and action
type. Use the same predeclared target-only selector for every arm. Aggregate exact match is a
diagnostic, not the primary success criterion.

The Dusknoir transfer audit must include slices for:

- Duskull setup and Bench-space timing;
- Duskull → Dusclops → Dusknoir evolution timing;
- Cursed Blast available: use versus intentional non-use;
- Dusclops 50-counter versus Dusknoir 130-counter choice;
- blast target selection and Prize/KO consequence;
- blast followed by the intended attack or KO line;
- post-self-KO promotion/relay and subsequent Prize race.

The Dragapult audit separately covers setup, energy/retreat budgets, front/back rotation, Phantom
Dive commitment and counter placement. For each slice compare T0–T3 on the same target-build
holdout. Auxiliary-build accuracy is optional diagnosis only: it cannot compensate for a target
regression or contribute to the headline result.

Implementation acceptance also requires tests that two valid 60-card decks encode differently,
materialized and online tensors agree, a fixed model responds when only `D` changes, and grouped
metrics cannot silently disappear.

## 9. Two-axis success and failure interpretation

- Target imitation up and target engine strength up: clean positive transfer.
- Target imitation down and target engine strength up: potentially stronger-than-literal imitation;
  accept only if a fresh frozen-package confirmation reproduces the gain.
- Target imitation up but target engine strength flat/down: better teacher reproduction, no evidence
  of stronger play.
- Both down: negative transfer / learned off-target.
- `T3 ≈ T3-no-deck` on target outcomes: extra data may help, but exact-deck conditioning was not
  shown to matter.
- One expert per build: build and style remain confounded; describe the result as build-plus-source
  transfer, not pure deck causality.

A later, stronger transfer test can fine-tune on a fourth held-out construction with a fixed small
sample budget and compare T3 initialization with scratch. It is not needed to answer the first
T0–T3 component question.

## 10. Target-only official-engine primary endpoint

Do not run a large tournament every epoch or select epochs from engine wins. Apply the same
predeclared target-validation checkpoint rule to every arm and freeze one candidate per arm. Every
candidate exports the same target Dragapult+Dusknoir `deck.csv`; auxiliary builds are not formally
evaluated for success.

The primary endpoint is self-relative target-deck official-engine strength: T1/T2/T3/T3-no-deck and
their predeclared milestone checkpoints are compared with our own T0 under the same opponent
catalog snapshot, paired seat/seed schedule, game count and metric profile. Report W/L/error,
completion, paired win-rate difference with uncertainty, both seats and per-opponent results. Use a
development batch for runtime/preliminary evidence, then one previously unused confirmation batch
for the claimed winner versus T0. Freeze packages before confirmation; do not tune from it. A
point-estimate gain without enough games or confirmation remains exploratory.

This contract deliberately permits “target exact-action agreement lower, confirmed target win rate
higher” to count as success. It does not permit auxiliary accuracy or one lucky engine run to do so.
No result automatically promotes an opponent.

The detailed data-overlay, matrix execution, W&B recording, overfitting and optional-extension
procedure is authoritative in [`RUNBOOK.md`](RUNBOOK.md).

Formal runs follow `V<n>_<tag>`, immutable artifacts, per-epoch validation and
`training_metrics.jsonl → TensorBoard → W&B`. The user authorized data preparation, training and
official-engine evaluation for this experiment window. Candidate admission, Kaggle submission,
dataset upload, commit and push remain unauthorized.

## 11. Results through V17

All candidates use the same THIRD PTCG Club target deck and persona, the predeclared
best-target-greedy-exact checkpoint selector, and the same 20-opponent official-engine catalog at
10 games per opponent. Every formal run completed 200/200 games with zero errors.

| Version | Training arm | Best target exact | Best epoch | Official-engine W/L | Win rate |
|---|---|---:|---:|---:|---:|
| V1 | T0 target only | 54.3% | 11 | 15 / 185 | 7.5% |
| V2 | T1 + all audited pure Dragapult | 58.7% | 10 | 39 / 161 | 19.5% |
| V3 | exploratory T1-no-deck | 60.1% | 11 | 30 / 170 | 15.0% |
| V4 | T1, loss weight 0.75 | 57.5% | 11 | 28 / 172 | 14.0% |
| V5 | T1, loss weight 0.50 | 59.0% | 11 | 41 / 159 | 20.5% |
| V6 | T1 independent seed 20260727 | 56.2% | 7 | 20 / 180 | 10.0% |
| V7 | fresh full-width rule reader, scale 0.10 | 60.2% | 11 | 25 / 175 | 12.5% |
| V8 | fresh full-width rule reader, scale 0.03 | 56.2% | 4 | 18 / 182 | 9.0% |
| V9 | V2 warm-start, full-model LR 1e-4 | 57.9% | 4 | 30 / 170 | 15.0% |
| V10 | V2 frozen base, rule-only LR 3e-4 | 61.3% | 4 | 29 / 171 | 14.5% |
| V11 | V2 frozen base, rule-only LR 1e-4 | 60.7% | 4 | 38 / 162 | 19.0% |
| V12 | V2 frozen base, scale 0.10 | 61.6% | 4 | 29 / 171 | 14.5% |
| V13 | V12 independent seed 20260727 | 61.5% | 7 | 40 / 160 | 20.0% |
| V14 | plain R15, legal-class smoothing 0.05 | 59.0% | 10 | 30 / 170 | 15.0% |
| V15 | T4 + 535 capped Marnie/Munkidori trajectories | 57.5% | 4 | 40 / 160 | 20.0% |
| V16 | T4 + 267 capped Marnie/Munkidori trajectories | 60.3% | 7 | 37 / 163 | 18.5% |
| V17 | V16 data, source initial scale 0.03 | 59.9% | 9 | 37 / 163 | 18.5% |

T1 improves both axes over T0: +4.4 percentage points target exact and +12.0 percentage points
official-engine win rate. The independent temporary engine runs were directionally consistent
(7.0% versus 20.0%). V3 is intentionally not T3-no-deck. Its higher offline agreement but lower
engine strength than V2 is evidence that initial-deck conditioning is useful to deployed strength
even though it slightly burdens exact teacher matching. Offline exact action therefore cannot rank
policy strength by itself.

Outcome weighting does not establish a new best policy. V4 is worse than V2 on both target exact
and engine strength. V5's first formal run is only +1.0 percentage point over V2, then its frozen
confirmation falls to 32/200 (16.0%). Combining the two independent batches gives V5 73/400
(18.25%) versus V2 79/400 (19.75%); the approximate V5−V2 interval spans `[-6.9pp, +3.9pp]`.
Therefore V2 remains the strongest supported candidate and the outcome-weight search stops.

The R15 seed/rule-reader sequence sharpens that conclusion. V6 shows substantial initialization
variance. V7–V10 all improve or match offline exact while losing materially more official-engine
games; V11 nearly recovers V2 at 38/200. Full-width rule adapters also overfit within a few epochs.
V12/V13 reproduce 61.6%/61.5% offline exact yet split to 29/200 and 40/200 in the engine. V13 is
only one win above V2's single 39/200 batch, while V12 is ten wins below it; this is variance, not
supported rule-reader uplift. V2 remains the selected policy, the prepared low-rank V14 gate is
rejected, and rule scale/LR/seed exploration stops.

V14's legal-class label smoothing slightly raises offline exact from 58.7% to 59.0% but falls to
30/200 (15.0%) in the official engine, so epsilon exploration stops. A read-only diagnostic average
of V2 epochs 9/10/11 also reaches only 59.0% exact (loss 0.59732, legal 100%); that 0.3pp offline
change is insufficient to allocate another formal version after V14 demonstrated the same offline
level can still regress badly in rollout.

V15 resolves the Unicode source-identity gate and adds 535 outcome-stratified Marnie/Munkidori
trajectories (48,793 decisions) to every T1 row. Its target exact falls to 57.5%, while its
official-engine result is 40/200 (20.0%), only one win above V2 and identical to the unstable V13
point estimate. V16 cuts the same deterministic cap order to 267 trajectories (24,555 decisions),
raising target exact to 60.3% but falling to 37/200 (18.5%) in the official engine. Neither dosage
establishes a supported improvement, and the cap sweep stops.

V17 is the one permitted architecture-conditioning follow-up. It keeps the V16 data membership,
source vocabulary, seed, optimizer and validation contract unchanged and changes only the source
embedding initial scale from 0.10 to 0.03. Its five-batch CUDA BF16 smoke and formal training both
complete with finite loss and 100% validation legality. Target exact reaches 59.9%, but official
engine strength remains exactly 37/200 (18.5%), identical to V16. Source-scale exploration stops.

A separate read-only V2 loss-best diagnostic scored 26/200 (13.0%), rejecting loss-best checkpoint
selection for 0015 despite its success in some 0014 runs.

Authoritative reports and their immutable run IDs are indexed at
[`evaluation/index.html`](evaluation/index.html). V1–V5 W&B sync and checkpoint provenance are
recorded in their corresponding version artifacts.

## 12. Current execution gates

1. The 0726 official ZIP and targeted overlays are frozen, hashed and deduplicated.
2. The per-build/source/outcome/seat decision audit and source-persona vocabulary are frozen.
3. The target validation split contains 12 whole episodes and 1,000 decisions, stratified to six
   wins and six losses; no trajectory crosses splits.
4. The shared R15 cache contains 55,764 eligible decisions and passed offline/online and safe-mask
   smoke tests.
5. T0, T1, exploratory T1-no-deck, the bounded V4/V5 outcome-weight sweep, V6–V13 R15/rule
   sequence, V14 label smoothing, V15/V16 T4 dosage comparison and V17 source-scale check are
   complete. Their gates did not displace V2, and no further cap/scale sweep is authorized.
   T2/T3/T3-no-deck remain pending a genuine Starmie + Dusknoir source and must not use Starmie +
   Froslass as a substitute.

The go/no-go criterion is not a minimum global row count. It is enough independent target episodes
to reserve a credible target holdout plus auxiliary coverage of the named component decisions. If
that coverage is absent, collect/audit data before increasing model complexity. Value calibration,
reward design and RL remain a later project.

## 13. Preferred post-matrix generalization arm

After T0–T3 and T3-no-deck are complete, the first related-deck extension is not an open-ended deck
search. It is a capped Marnie's Grimmsnarl dataset containing Munkidori (`card_id=112`, TWM 95).
Adrena-Brain moves up to three damage counters from one own Pokémon to one opponent Pokémon when
Munkidori has Darkness Energy attached. That shares damage-counter target selection, KO arithmetic
and pre-attack sequencing with Dragapult, and some Dragapult builds also carry Munkidori.

`T4-marnie-munkidori` adds this source to the complete T3 union. Because Marnie episodes are much
more numerous, T4 is the deliberate exception to the natural-use-all rule: freeze a deterministic,
source/deck/outcome/seat-stratified episode cap after audit and before training, no larger than the
number of T3 training episodes. Compare only on the target Dragapult+Dusknoir validation and engine
contract. If the exact target deck lacks Munkidori, describe any gain as indirect damage-counter
reasoning transfer, not direct Munkidori skill.

The T4 source-identity gate is resolved in dataset generation V2. All frozen source IDs 1–84 remain
unchanged and the seven exact Unicode identities receive stable hashed IDs 85–91. The complete pool
contains 4,672 candidates; 191 trajectories were in the former Unicode collision bucket. V15 uses
the declared 535-trajectory cap and V16 uses its 267-trajectory subset. Evidence is in
[`data_audit/t4_marnie_v2.md`](data_audit/t4_marnie_v2.md) and
[`decisions/005_t4_non_ascii_source_identity_gate.md`](decisions/005_t4_non_ascii_source_identity_gate.md).
