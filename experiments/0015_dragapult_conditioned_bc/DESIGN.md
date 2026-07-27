# 0015 Dragapult Conditioned BC

**Project ID:** `0015_dragapult_conditioned_bc`
**Status:** prospective design only; no 0015 dataset, run, candidate or strength result exists.
**Current phase:** maximize BC quality and test component transfer. Value/RL remains deferred.

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
\pi(a \mid s, D),
\]

where `s` is the actor-visible action-time state and `D` is the exact registered own-deck multiset.
This is deck-conditioned play, not automatic deck construction.

## 2. 0014 readiness audit

The architectural verdict is **yes for AC/R1, no for A0**. The full evidence is recorded in
[`decisions/001_0014_registered_deck_audit.md`](decisions/001_0014_registered_deck_audit.md).

| Layer | Finding | Meaning for 0015 |
|---|---|---|
| Cache/compiler | `registered_card_ids` + aligned `registered_multiplicity` losslessly represent the exact 60-card multiset. | Pure, self-sacrificing and Starmie builds can be distinguished. |
| AC/R1 model | Registered cards and ledger state are consumed as resource/scenario memory and condition state/options. | Reuse this family and preserve the channel. |
| Online/export | The candidate's `deck.csv` is validated as 60 cards and encoded through the same fields. | Training and deployed package can share the contract. |
| A0 | Does not consume registered-deck features. | Excluded from the transfer experiment. |
| Existing evidence | V9 logits respond to a one-card count swap. | Wiring and learned sensitivity are proven, strategic generalization is not. |

The 0014 cache contains 162,128 decisions but only two signatures: 138,898 decisions (85.7%) use
Enhanced Hammer ×4 / Nighttime Mine ×2, while 23,230 (14.3%) use Hammer ×3 / Mine ×3. The V9
epoch-11 checkpoint changed its finite logits when only this multiplicity was switched
(max absolute delta `0.1878586`, mean `0.0823456`). This is reassuring, but a one-card imbalanced
variation inside one archetype cannot prove that the model understands three materially different
builds. That is precisely what 0015 must test.

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

The policy receives only actor-visible state, legal full-action options and its complete own deck.
It never receives future frames, terminal outcome, opponent hidden cards, preset opponent ID/deck,
or expert identity. `source_id` remains audit/sampling metadata unless a separate deployable expert-
persona contract is explicitly approved.

```text
actor-visible board/history + legal full-action options
                         │
                   state/option encoder
                         │
exact own 60-card multiset ──> registered-card + resource-ledger memory
                         │
              conditioned state and legal options
                         │
               full-action BC policy π(a|s,D)
```

Use the 0014 AC or R1 registered-deck path. Preserve card identity, initial multiplicity, ontology,
known zones, remaining known/bounded resources and unknown boundaries. Unknown must not become zero.
An optional small adapter is a later fallback if shared training demonstrably harms a build; it
must not replace exact-deck conditioning.

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
| T3-no-deck | Same data/updates as T3, exact deck channel masked | Is any gain actually attributable to construct conditioning? |

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
`training_metrics.jsonl → TensorBoard → W&B`. No training, evaluation, candidate admission or
external submission is authorized by this prospective design.

## 11. Tomorrow's minimum work package

1. Freeze and hash the three exact deck lists and enumerate expert/team sources.
2. Audit complete episode counts, win/loss mix, decision counts and duplicate/session leakage.
3. Decide whether target data is sufficient for a held-out target validation/test set.
4. Materialize a dry-run manifest and verify the 0014 offline/online exact-deck tensors agree.
5. Freeze natural T0–T3 episode membership/counts, epoch contract, one target-only checkpoint
   selector, Dusknoir slices, paired engine schedule and fresh confirmation seeds before launching
   any formal version.

The go/no-go criterion is not a minimum global row count. It is enough independent target episodes
to reserve a credible target holdout plus auxiliary coverage of the named component decisions. If
that coverage is absent, collect/audit data before increasing model complexity. Value calibration,
reward design and RL remain a later project.

## 12. Preferred post-matrix generalization arm

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
