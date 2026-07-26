# 0015 Outcome-Aware Value Foundations

**Project ID:** `0015_outcome_aware_value_foundations`  
**Status:** prospective design; no dataset has been frozen, no training version exists, and no
policy-strength claim is made.  
**Scope:** improve the first supervised stage by retaining outcome information from complete,
same-source trajectories while establishing value-calibration and later causal-attribution
foundations. This is not yet online RL, offline-RL policy optimization, or an opponent promotion.

## 1. First-principles question

In Pokémon TCG, a loss does not label every earlier action as incorrect. A loss may follow an
unlucky draw, an opponent response, hidden information, or one decision that irreversibly spends a
turn budget. Conversely, a win may contain actions that were merely not punished.

0015 therefore does **not** ask the policy head to classify every action in a loss as bad. It asks:

1. Can an action-time, actor-visible state representation estimate the probability that the acting
   player will eventually win?
2. Does adding that representation-learning task improve or at least preserve BC quality under a
   controlled comparison?
3. Which decisions deserve later counterfactual investigation, without pretending that a single
   observed trajectory proves causation?

The relevant game semantics are part of the state, not optional reward heuristics: attack commits
and ends the turn; Supporter, manual Energy and Retreat are turn budgets; evolution has a timing
clock; Prize, board survival and future draw-at-turn-start determine victory. The official engine
handles legality, while the model must learn the value consequences among legal choices.

## 2. Hard boundaries

### 2.1 Data provenance and isolation

- The first 0015 dataset must contain complete win **and** loss trajectories from one explicitly
  named expert/team policy and one deck. It must record upstream Episode ID, date, deck, source
  policy, player seat, terminal result and extraction/audit status.
- Different decks, teams, or implementation policies must remain separate datasets and separate
  policies. A future shared model requires explicit deck/source conditioning, conflict handling,
  grouped evaluation and a separately approved design.
- Train/validation/test assignment is grouped by complete episode, never by decision. No state from
  one episode may appear in more than one split. Time/source/deck counts and outcome balance must
  be reported before any training starts.
- Incomplete, corrupted, ambiguous-winner or provenance-invalid episodes are excluded rather than
  silently assigned a target.

### 2.2 Visibility and target contract

For a decision state \(s_t\), the encoder may consume only information visible to the acting player
before that action: current observation, legal options, audited action-time history and known/unknown
resource facts. It may not consume a post-action frame, terminal reason, terminal result, future
log event, opponent hidden card identity or viewer-omniscient data.

The terminal label is from the **acting player perspective**:

\[
z_t = \begin{cases}1 & \text{if the actor at }s_t\text{ eventually wins}\\
0 & \text{if that actor eventually loses.}\end{cases}
\]

`z_t` is a training target only. It is never serialized as a feature, supplied to the policy at
inference, or used to select an action in a live official-engine game.

## 3. Candidate first-stage architecture

0015 begins with one actor-visible board/option encoder and two explicitly separate output heads:

```text
action-time observation + legal options + causal state
                         │
                  shared encoder h(s)
                    ┌────┴────┐
                    │         │
          policy π(a | s)   value V(s) = P(win | s)
          BC supervision    all-outcome terminal supervision
```

The initial objective is a controlled multi-task supervised objective:

\[
L = L_{\mathrm{BC}} + \lambda_v L_{\mathrm{value}},
\qquad
L_{\mathrm{value}} = \operatorname{BCE}(V(s_t), z_t).
\]

The first policy target policy is deliberately conservative: `L_BC` is the existing valid
full-action teacher-forced BC objective on the chosen expert-label subset, initially winner-only
if that is the frozen baseline. Loss trajectories expand value supervision first; they are not
automatically converted to negative action labels. A declared, separately evaluated all-outcome BC
control may be added only to measure whether label quality or state coverage changes the result.

`λ_v` is an experimental variable, not a presumed improvement. The shared encoder, policy head,
value head, optimizer, train/validation cadence and checkpoint-selection rule must all be recorded.
A detached-value-head control is permitted if joint gradients damage BC.

## 4. What value can and cannot say

The immediate diagnostic is a value trajectory:

\[
\Delta V_t = V(s_{t+1}) - V(s_t).
\]

A sharp drop marks a **candidate decision window** for replay and state audit. It is not proof that
the action caused the loss: `s_{t+1}` may include randomness, opponent consequences or unobserved
counterfactual choices. A `V(s)` model estimates state outcome, not the values of all legal actions.

Stronger action attribution later requires action value:

\[
Q(s,a) = \mathbb{E}[z \mid s,a],\qquad
A(s,a) = Q(s,a) - \max_{a'\in\operatorname{legal}(s)} Q(s,a').
\]

Historical data normally provides one selected action per state. Therefore it cannot alone identify
the counterfactual value of every legal alternative. Candidate errors from 0015 must eventually be
tested through reproducible official-engine rollouts with documented policy, opponent catalog,
seeds/seat protocol and uncertainty; they cannot be promoted from a value dip to a causal claim.

## 5. Data and feature research questions

The value task is only meaningful if the state tells it why a game is strategically fragile. Its
feature audit must explicitly cover, while maintaining actor visibility:

| Family | Why it matters to value | Required boundary |
|---|---|---|
| Turn phase and action budget | Attack commitment and the once-per-turn Supporter/Energy/Retreat choices can make a decision irreversible. | State before the labeled action only. |
| Board and evolution clock | A Basic played this turn and an older Basic have different future routes. | `appearThisTurn` and audited causal history; do not infer from future evolution. |
| Resources and zones | Prize race, recovery, attachments and future draw risk depend on where known cards are. | Unknown remains unknown; opponent hidden identities remain hidden. |
| Legal options and action structure | Value must distinguish a constrained state from a state with multiple live routes. | Options at the decision only, with the established full-action contract. |
| Opponent public threat | Damage, Energy, board and Prize state affect immediate survival and race. | Public observation only; no preset opponent ID/deck. |

The starting feature candidate is the 0014 actor-visible, faithful-board causal representation,
but 0015 must audit its availability over the newly selected all-outcome corpus before reusing it.
No future terminal field may enter the feature cache.

## 6. Staged experiment matrix

No row below creates a `V<n>_<tag>` directory until its data audit, config and unused-path checks
are complete. Each actual training must receive a strictly increasing version and its own local
metrics, TensorBoard, W&B staging/status and, if evaluated, a non-overwritable formal report.

| Stage | Hypothesis | Policy supervision | Value supervision | Decision gate |
|---|---|---|---|---|
| D0 | Data can support an unbiased audit. | None | None | Manifest, split, outcome balance, chronology and visibility audits pass. |
| P0 | Existing BC remains reproducible on frozen 0015 source/split. | Frozen winner-only BC control. | None | Offline contract parity; no strength claim. |
| P1 | Outcome supervision learns calibrated state value without harming imitation. | Same as P0. | All valid win/loss decision states, `z`. | Compare `λ_v=0`, small, and larger declared values. |
| P2 | Broader action labels improve coverage rather than copy failures. | Explicit all-outcome BC control. | Same all-outcome value target. | Only proceed if grouped offline and engine evidence justify it. |
| P3 | Value identifies useful review windows. | Best frozen P0/P1/P2 policy. | Best calibrated value head. | Replay audit plus repeated engine rollouts; no automatic policy update. |
| Later | An action-value/advantage signal can support policy improvement. | Not specified by 0015. | Requires a new Q/rollout/RL design. | Separate project decision; never inferred from P1 alone. |

The P1 primary comparison is not “does terminal-loss weighting make the policy look good?” It is
whether a calibrated auxiliary value target improves a frozen, otherwise identical BC contract.
P2 exists precisely to test the tempting but risky alternative of treating all observed losing
actions as imitation labels.

## 7. Measurement and gates

### 7.1 Value evidence

- Report binary cross-entropy, Brier score, reliability/calibration curve and calibration intercept/
  slope, not accuracy alone.
- Report each metric by terminal outcome, turn/decision bucket, seat, first/second player, game
  length, deck/source and strategically relevant bins such as Prize differential and attack-ready
  state. Sparse bins are reported with counts and not overinterpreted.
- Holdout evaluation must use the same actor-visible feature pipeline. A result that depends on
  outcome leakage is invalid even if its calibration is numerically strong.
- Inspect value trajectories on preselected and randomly sampled games; do not select only dramatic
  losses after seeing the value curve.

### 7.2 Policy evidence

- Keep online training diagnostics distinct from fixed-model validation teacher-forced and greedy
  full-action metrics. `training_metrics.jsonl` remains canonical, followed by TensorBoard and then
  failure-isolated W&B scalar mirroring.
- A value improvement is not a policy-strength result. A policy can be exported only as a
  self-contained candidate package, validated, and tested in real games with the official engine.
- Compare policies only under the same frozen opponent catalog/pool snapshot, seeds, seats, metric
  profile and report contract. No single noisy run authorizes promotion.

### 7.3 Stop conditions

Stop or record a failed version when the data audit finds outcome/future leakage, source/deck mixing,
invalid chronology, poor calibration relative to a trivial baseline, policy regression beyond the
declared tolerance, or an unexplainable divergence between replay audit and engine behavior. Keep
the artifacts and failure reason; do not overwrite or delete the iteration.

## 8. Design Q&A for the next discussion

| Question | Current answer / hypothesis | Evidence needed before committing |
|---|---|---|
| Should loss actions train the policy? | Not by default. Use them first for value supervision; make all-outcome policy BC a named control. | P0/P1/P2 comparison and official-engine evaluation. |
| Is win/loss enough as the first reward? | Yes for initial value calibration: terminal win/loss is unambiguous and avoids inventing dense reward. It is sparse and cannot alone assign blame. | Outcome audit, calibration metrics and replay review. |
| Should we add dense Prize/KO/turn rewards now? | No. They may later be auxiliary analyses, but they can change the objective and reward-hack the policy. | Separate reward-design proposal and ablation plan. |
| How can we identify the wrong action? | `ΔV` gives a review candidate, not proof. Need Q-values or controlled engine alternatives for action-level claims. | Legal-action counterfactual rollout design. |
| Should value share the policy encoder? | Test it. Sharing may improve state representation but can also interfere with exact imitation. | `λ_v=0` and multiple nonzero λ controls; optionally detached-head control. |
| Can multiple decks be pooled for more data? | Not in 0015's first stage. Keep deck-specific policies/data; pooling needs explicit conditioning and grouped contracts. | A separate cross-deck design. |
| When does this become RL? | Only when the current policy is improved from rewards/returns through an explicitly specified offline or online policy-optimization procedure. | New rollout/reward/Q/PPO design and official-engine validation. |

## 9. Tomorrow's first audit checklist

1. Select one candidate deck and one normalized expert/team source; enumerate all official complete
   episodes, including both outcomes, without modifying raw archives.
2. Produce an immutable manifest with episode-level provenance, acting player, winner, dates,
   duplicate handling, exclusion reasons, source/deck counts and grouped split assignment.
3. Verify that the terminal result can be attached as a label to each decision from actor perspective
   without becoming an input feature, and verify decision chronology.
4. Run the 0014-style actor-visible feature audit on this corpus; identify any state fields that are
   unavailable, stale or future-derived.
5. Establish P0 before P1. Define in advance the candidate `λ_v` values, checkpoint selection rule,
   calibration metrics and P1 success/failure criteria.
6. Do not start RL, change the engine, promote an opponent or submit externally as part of this
   design/audit work.

## 10. Current decision record

0015 adopts the following working proposition: **use complete loss trajectories to learn and
calibrate state value, not to declare every demonstrated action wrong.** The project will seek
better representation and data coverage first, preserve BC as an explicit control, and treat
action-level causal attribution and RL optimization as subsequent evidence-gated work packages.

This document is authoritative for 0015's intended model inputs, heads, targets and phase boundary.
It must be updated with the actual dataset manifest, feature schema, architecture shapes, training
objective, run versions and results before any implementation or training changes are considered
complete.
