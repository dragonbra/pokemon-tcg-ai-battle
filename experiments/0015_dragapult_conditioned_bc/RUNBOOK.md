# 0015 Dragapult Conditioned BC Execution Runbook

**Purpose:** operational handoff for the first authorized 0015 experiment day.
**Status:** execution authorized for data preparation, BC training and official-engine evaluation.
Kaggle submission/upload, candidate admission, commit and push remain unauthorized.
**Primary target:** Dragapult + Dusknoir only.
**Expected window:** up to 12 hours of continuous, evidence-driven work from the 2026-07-27 handoff.

## 1. Completed handoff

The required handoff is complete. Do not copy secrets into manifests, logs or W&B.

One source gap was discovered after handoff: the environment's Mega Starmie/Dusknoir display label
was hard-coded, while the exact decks contain Froslass and no Duskull line. T2 must stay pending
until registration-frame evidence identifies a genuine Starmie + Dusknoir source. This does not
block shared infrastructure or T0/T1 smoke; details are in
`decisions/003_starmie_dusknoir_source_gap.md`.

| Input | Required content | Acceptance check |
|---|---|---|
| Official episode bundle | 0726 immutable ZIP; 4,554 Episode JSON; SHA-256 `9394d9c4c18dc476cc1fffd5a2dae29c4f2a7afbfa9b2d435876568e2c22fe95` | ZIP CRC and member count passed. |
| Dragapult user list | THIRD PTCG Club (target), LumenLiquidity and Oshbocker (Dragapult+Blaziken) | Submission IDs, player indices and exact decks frozen. |
| Temporary JSON patch | 229 downloaded JSON plus 81 official-base references | 310 unique Episode IDs/content hashes; no double count. |
| Exact named decks | Four audited 60-card manifests including `flg` Marnie/Munkidori | Row counts and sorted-multiset hashes passed. |
| 0014 successor model | `0014/V25_r15_deterministic_gradual_option` | R15 code, feature switches, hyperparameters and SOTA evidence identified. |
| Compute envelope | RTX 5080 16 GB; 834 GB free at preflight | Matrix smoke must still establish safe effective batch/runtime. |
| Execution authority | User explicitly requested continuous execution for up to 12 hours | Data/training/evaluation authorized; external publication actions are not. |

R15 is frozen as the architectural base. The 0015-only source-persona condition required by the
multi-team BC contract must be implemented and smoke-tested before T0. Do not train T0 on one
architecture and later arms on another. Any architecture correction after T0 creates a new complete
matrix generation; it cannot be silently mixed with earlier arms.

## 2. Immutable research question

The matrix tests whether auxiliary demonstrations improve one deployed target policy:

| Arm | Training data | Inclusion rule |
|---|---|---|
| T0 | Target Dragapult+Dusknoir | Every usable audited target trajectory. |
| T1 | T0 + pure Dragapult | T0 unchanged, plus every usable audited pure-Dragapult trajectory. |
| T2 | T0 + Starmie+Dusknoir | T0 unchanged, plus every usable audited Starmie+Dusknoir trajectory. |
| T3 | All three | Exact union of all usable T0, pure-Dragapult and Starmie+Dusknoir trajectories. |
| T3-no-deck | Same rows/order/epochs as T3 | Only exact initial-deck composition is removed; source persona and live ledger remain. |

If T2 remains blocked by the genuine-source gate after T0/T1 complete, run one explicitly
exploratory `T1-no-deck` arm with the exact T1 rows and training contract. This is an early
conditioning diagnostic, not a replacement for T2, T3 or the paired T3-no-deck result.

There are no artificial 50/50 build quotas in the primary matrix. “Usable” means the trajectory
passes the predeclared provenance, completeness, deck-identification, causal-feature and label-
contract checks; it does not mean selecting only enough rows to balance another source. Record the
natural episode/decision proportions before training, then keep the selected episode set unchanged
through T0–T3.

### Executed through V3 (2026-07-27)

T0 and T1 completed under the frozen contract. T2 remained blocked because the full 0726 audit
found no genuine Starmie + Dusknoir registration. The blocked-gate contingency was used to run an
exploratory T1-no-deck diagnostic; it is not T3-no-deck.

| Version | Target exact | Official-engine result | Interpretation |
|---|---:|---:|---|
| V1 T0 | 54.3% | 15-185 (7.5%) | scarce-target baseline |
| V2 T1 | 58.7% | 39-161 (19.5%) | positive pure-Dragapult transfer |
| V3 T1-no-deck | 60.1% | 30-170 (15.0%) | better label match, weaker play than conditioned T1 |
| V4 loss weight 0.75 | 57.5% | 28-172 (14.0%) | negative; do not select |
| V5 loss weight 0.50 | 59.0% | 41-159 (20.5%) | first batch inconclusive; confirmation 32-168 |

All formal evaluations ran 200 games, completed 100%, and had zero errors. The authoritative
reports and run IDs are in [`evaluation/index.html`](evaluation/index.html). Continue to preserve
the T2 source gate; do not rename V3 as a base-matrix arm.

V5's frozen confirmation reversed its first-run +1pp margin. Combined V5 is 73/400 (18.25%) versus
V2 at 79/400 (19.75%). Keep V2 as the current best and stop the outcome-weight sweep; do not search
intermediate weights or additional seeds from this evidence.

All arms use the same:

- target train/validation/test episode split;
- model architecture, initialization policy, optimizer and random-seed contract;
- maximum epoch/early-selection contract, optimizer and effective batch size;
- target validation checkpoint selector;
- milestone checkpoint schedule;
- target `deck.csv` at inference and official-engine evaluation.

Each epoch processes every selected training decision once, as required by the repository training
contract. T1–T3 therefore naturally have more decisions, optimizer updates and wall time than T0.
That is intentional: this is a competition-oriented “use all available qualified data” experiment,
not an academic isolation of data from compute. Report decisions, updates and wall time so the gain
is described honestly as the result of adding the complete data source, not as pure causal transfer.

## 3. Data overlay without duplicating the baseline

The official daily bundle is the base source. The manually downloaded user episodes are a temporary
overlay, not another copy of the complete dataset.

```text
official daily bundle (read-only) ──┐
                                    ├─> episode index / manifest ─> shared 0015 feature cache
targeted JSON patch (small) ────────┘             │
                                                  ├─> T0 sampler view
                                                  ├─> T1 sampler view
                                                  ├─> T2 sampler view
                                                  ├─> T3 sampler view
                                                  └─> T3-no-deck view
```

Required storage behavior:

1. Keep the base bundle read-only and reference its absolute resolved path plus checksum.
2. Store only the additionally downloaded episode JSON in one versioned overlay directory under
   `rl_runs/0015_dragapult_conditioned_bc/dataset/sources/`; do not copy the base bundle there.
3. Build one canonical episode index keyed by official `episode_id`. If the same episode appears in
   base and overlay with identical content, retain the base record and mark the overlay duplicate.
   If payloads differ, quarantine it and do not train until explained.
4. Bind each selected player-side trajectory to exact deck hash, build, source and outcome. Reject
   inferred/ambiguous builds rather than guessing.
5. Compile shared feature shards once. Arm-specific datasets are lightweight manifests/sampler
   views over the shared cache, not physical copies of feature tensors.
6. When the next official bundle absorbs the patch, rebuild only the source index/reference. Verify
   that selected episode IDs and content hashes are unchanged before deleting no overlay material.
   Deletion is a separate user-approved cleanup action.

### Required data audit

Produce `experiments/0015_dragapult_conditioned_bc/data_audit/<dataset_version>.md` and a machine-
readable manifest containing, per build/source/split/outcome:

- independent episodes, player trajectories and decisions;
- exact deck hashes and number of distinct registered decks;
- win/loss/unknown counts, first/second seat and episode completeness;
- action-type and key Dragapult/Dusknoir slice coverage;
- duplicate episode IDs, duplicate content hashes and near-duplicate sessions;
- base-versus-overlay provenance and all excluded/quarantined reasons.

Split entire episode-player groups. Group the same session/user-day conservatively when consecutive
episodes can leak repeated states or expert habits. Only the target build is required to have a
credible fixed validation/test holdout; auxiliary splits exist for diagnosis and leakage control.

## 4. Model readiness gate

The selected successor is 0014 R15 from
`rl_runs/0014_faithful_board_causal_features/versions/V25_r15_deterministic_gradual_option`.
It is accepted for a formal 0015 version only if the implementation passes all of these:

- its feature schema is frozen and includes the exact registered 60-card channel;
- offline compilation and online encoding agree on registered IDs, multiplicity, masks and ledger;
- changing only a valid deck multiset changes the intended conditioning path;
- source IDs use a frozen auditable vocabulary and target export fixes the target persona;
- T3-no-deck replaces initial-deck memory with one valid null token/nonempty mask and zero
  multiplicity without changing live ledger, source persona, rows, labels, order or epochs;
- the candidate export reads the target package `deck.csv` and validates exactly 60 cards;
- a tiny train/validation smoke produces finite metrics and one loadable checkpoint;
- a candidate smoke returns only official-engine legal actions;
- the training interpreter imports W&B and the online project is reachable, or the formal status
  records why online mirroring is unavailable.

Reuse R15's complete deterministic R2 evidence paths, option scale `0.35`, batch 256, validation
batch 512, LR `3e-4`, weight decay `0.02`, BF16, seed `20260723` and early-stop logic. Start every
arm from the same fresh seeded initialization; do not warm-start Dragapult from the Alakazam R15
checkpoint. Record the R15 source version, source state and model-contract hash as the 0015 base.

## 5. Version allocation and W&B contract

Interpret the requested “OneDB” record as the repository's W&B contract unless the user supplies a
different system tomorrow. Before every run, inspect existing 0015 version directories and allocate
the next unused strict version. Never overwrite or append new semantics to an existing version.

Recommended tags, with actual `V<n>` assigned at execution time:

```text
V<n>_t0_target_only
V<n>_t1_plus_pure_dragapult
V<n>_t2_plus_starmie_dusknoir
V<n>_t3_component_mix
V<n>_t3_no_deck_ablation
```

Every formal arm writes:

```text
rl_runs/0015_dragapult_conditioned_bc/versions/<version>/
├── artifact/
│   ├── training_config.json
│   ├── dataset_reference.json
│   ├── model_contract.json
│   ├── training_metrics.jsonl
│   ├── training_summary.json
│   ├── status.json
│   └── evaluation.json
├── checkpoint/
├── tensorboard/
└── wandb/
```

W&B uses private project `dragon_bra/pokemon-tcg-policy-learning`, one stable run ID per repository
version, `trainer/epoch` as the axis and `bc/*` metrics. The canonical write order is
`training_metrics.jsonl → TensorBoard → W&B`. At minimum log:

- `bc/optimization/loss`, token accuracy, teacher exact and throughput;
- target validation loss, token accuracy, greedy full-action exact and legality every epoch;
- target Dragapult/Dusknoir slice metrics with sample counts;
- auxiliary metrics under an explicit diagnostic namespace, never the headline namespace;
- natural rows/episodes per build/source/outcome and any repeated-sampling rate;
- learning rate, gradient norm, epoch/update, wall time and checkpoint identity;
- W&B sync state/failure in `status.json` without losing local canonical metrics.

Do not upload datasets, episode JSON, replays, checkpoints, observations or secrets to W&B.

### Start-of-window preflight

Run these read-only checks and paste their outputs or summaries into the day's command record under
`experiments/0015_dragapult_conditioned_bc/commands/` before allocating T0:

```bash
git status --short
python3 --version
python3 -c "import torch, wandb; print(torch.__version__, torch.cuda.is_available(), wandb.__version__)"
nvidia-smi
python3 -m wandb status
python3 -m evaluation list-opponents
```

Then run the exact model/data validation and smoke commands delivered with the 0014 successor. Do
not invent a compatibility wrapper in `scripts/`. Record the resolved dataset/model paths, hashes,
CLI arguments, environment overrides and exit status. If the selected model has no reproducible
entrypoint, that is a model-readiness blocker rather than permission to improvise a formal run.

## 6. Training order and checkpoint curve

### Gate A: matrix smoke

Before any full run, execute a tiny smoke for all five arms. Verify row identity for T3 versus
T3-no-deck, nonzero target/auxiliary counts, finite forward/backward loss, validation completion,
checkpoint reload and W&B namespace. A failure blocks the affected arm and triggers root-cause
diagnosis; it does not justify skipping silently to later experiments.

### Gate B: primary matrix

Run T0, T1, T2, T3 and T3-no-deck with the frozen contract. T0 estimates runtime and memory but
must not be used to change only later arms. If a safety correction is needed, create new versions
and restart the comparable matrix.

To observe self-relative strength during training without choosing epochs from wins, predeclare the
same small set of epoch-relative milestones for every arm:

- first fully trained epoch;
- 25%, 50%, 75% and 100% of planned epochs;
- the target-validation-selected checkpoint, if it is not already one of those milestones.

Export those checkpoint candidates with the same target deck. Run a small, identical official-
engine schedule for the milestone curve and store reports under
`.tmp/evaluation/0015_milestone_curve/`. These are real games but exploratory curve evidence. Never
add or remove a milestone after seeing its win rate, and never use the curve to change the selected
checkpoint inside the same matrix.

The formal frozen checkpoint for each arm is selected by the same predeclared target-only validation
rule, not by auxiliary validation or engine wins. Formal evaluation writes the corresponding
`experiments/0015_dragapult_conditioned_bc/evaluation/<version>.html` and refreshes `index.html`.

### Gate C: confirmation

Rank the frozen arms against T0 using the same official-engine opponent catalog snapshot, paired
seats/seeds, game count and metric profile. Report W/L/error, completion, both seats, per-opponent
results and uncertainty. Keep the packages frozen. The leading arm then receives one confirmation
batch against T0 with previously unused seeds. If the confirmation reverses the gain, report the
result as inconclusive; do not search additional seeds.

There is no teacher candidate and no teacher local win rate. All curves and deltas are comparisons
among our own T0–T3 checkpoints using the same target deck.

## 7. Reading the curves and responding to overfitting

Use target-only validation for model-selection diagnostics and target-only engine results for
strength. The four main outcomes are:

| Observation | Meaning | Next action after the base matrix |
|---|---|---|
| Train and target validation improve; engine improves | Clean progress | Confirm the best arm. |
| Train improves; target validation worsens | Logged-state overfitting | Keep the best predeclared checkpoint; start one new regularization/data-balance version. |
| Target exact worsens; engine improves | Possible robust policy transfer | Confirm with frozen package/new seeds; inspect legality and key slices. |
| Target exact improves; engine is flat/worse | Better imitation, not stronger play | Inspect exposure bias and failure cases; do not claim strength. |

If overfitting is observed, change one declared variable per new version, in this priority order:

1. early-stop/checkpoint choice already available from the original run;
2. remove accidental duplicate/repeated sampling while retaining every unique usable episode;
3. adjust regularization or epoch selection before considering source reweighting;
4. regularization already supported by the delivered SOTA model (weight decay/dropout), one change;
5. only then a model-capacity change, which requires a new model contract and DESIGN sync.

Do not retroactively rewrite an arm, pool failed versions, or tune on formal engine results. Record
the observed evidence, single changed variable and expected curve change in a new decision/version.

## 8. Optional extensions after the complete base matrix

No extension starts until all five base arms have complete local metrics, checkpoint packages and
comparable target evaluation, unless a genuine blocker makes an arm impossible and is recorded.

### E1: capped Marnie's Grimmsnarl + Munkidori extension

The first and preferred post-matrix deck source is Marnie's Grimmsnarl containing Munkidori
(`card_id=112`, TWM 95). Its Adrena-Brain Ability, when Darkness Energy is attached, moves up to
three damage counters from one own Pokémon to one opponent Pokémon. This shares damage-counter
allocation, target selection, KO arithmetic and pre-attack sequencing with Dragapult play; some
Dragapult constructions also include Munkidori directly.

Create one explicit arm after T0–T3/T3-no-deck:

| Arm | Training data | Comparison |
|---|---|---|
| T4-marnie-munkidori | Complete T3 union + a capped Marnie/Munkidori episode set | Compare target-only validation and engine strength with T3 and the best base arm. |

Marnie data is the explicit exception to the “use all qualified data” default because its natural
volume is much larger. After the data audit but before training T4, freeze an episode cap no larger
than the number of independent training episodes in the complete T3 union. Select episodes
deterministically by episode hash, stratified across exact Marnie deck hash, source, outcome and
seat. Never keep only wins or choose episodes from their validation performance. Record eligible,
selected and excluded-by-cap counts.

Before any T4 materialization, satisfy
[`decisions/005_t4_non_ascii_source_identity_gate.md`](decisions/005_t4_non_ascii_source_identity_gate.md).
The current candidate index collapses seven non-ASCII team names into one `non_ascii_source` token;
T4 requires a new versioned index with collision-safe Unicode identity. Never regenerate or
renumber the frozen V1–V5 source vocabulary in place.

If the target Dragapult+Dusknoir deck itself contains Munkidori, T4 tests direct component transfer.
If it does not, T4 tests only indirect damage-counter reasoning transfer; state that weaker claim
before training. Add diagnostic slices for:

- Munkidori in play with Darkness Energy versus without it;
- own damaged Pokémon and legal opponent counter targets;
- Adrena-Brain available: use/non-use, source Pokémon, target Pokémon and counters moved;
- whether the counter move creates a KO or improves the following attack/Prize line;
- sequencing of Adrena-Brain relative to other Abilities, Supporter actions and the final attack.

If T4 improves the target and time remains, one second dose arm may increase the frozen cap up to
twice the T3 training-episode count. Do not automatically consume the full Marnie corpus. If T4 is
flat or negative, stop this source rather than tuning many caps against engine results.

### E2: outcome-aware BC weight sweep

Keep the best base data mixture and model fixed. Test a small predeclared set of bounded episode-
level weights for losing demonstrations. Suggested first sweep:

| Variant | Winning episode weight | Losing episode weight | Purpose |
|---|---:|---:|---|
| E2-control | 1.0 | 1.0 | Reproduce the chosen base arm. |
| E2-soft | 1.0 | 0.75 | Retain losing-state coverage with modest trust discount. |
| E2-strong | 1.0 | 0.50 | Test whether noisy losing labels are limiting BC. |

Weights multiply imitation loss uniformly within the episode. They are not per-action credit
assignment, reward-to-go or a value target. Compare unweighted target validation and the same target
engine contract. A true reward/value head is a separate semantic experiment and new version; start
it only after the static sweep and only if the user explicitly confirms that extension at execution
time.

### E3: deeper reward/value work

This is last priority and outside the BC-first base claim. If explicitly authorized, first publish a
separate loss/head/reward contract and synchronize DESIGN.md/HTML before implementation. Do not call
episode-weighted BC a value function or RL.

## 9. Twelve-hour persistence protocol

Once explicitly launched, continue through the highest safe incomplete item without asking for
routine permission or stopping merely because one arm underperforms:

1. ingest and audit base + overlay;
2. freeze model/data/matrix contracts;
3. smoke all arms;
4. complete T0–T3 and T3-no-deck;
5. export/evaluate milestone and frozen target candidates;
6. run the one fresh confirmation comparison;
7. if time remains, execute the capped T4 Marnie/Munkidori arm, then the E2 outcome-weight sweep;
8. E3 only with explicit authorization.

Send concise progress updates at least every 60 minutes during active work and at every gate. A
training process may run longer, but it must be polled and its metrics/status checked. If a run
fails, preserve its version, diagnose the root cause, and continue with a new version when safe.

Stop and request direction only for a real scope/authority blocker, ambiguous target/source mapping,
conflicting episode payload, missing model contract, unsafe resource condition, or a decision that
would change the frozen comparison. Lack of improvement is a result, not a reason to stop.

## 10. End-of-window handoff

Deliver one concise index containing:

- exact base bundle and overlay provenance, selected episode/deck/source counts and hashes;
- frozen model/feature contract and initialization source;
- T0–T3/T3-no-deck version IDs, W&B URLs/sync state and completion status;
- target-only validation curves and milestone official-engine win-rate curves;
- formal frozen target evaluations and confirmation result;
- imitation-versus-strength interpretation, including inconclusive/negative results;
- every deviation, failed version, overfitting response and optional extension;
- clickable DESIGN, RUNBOOK, data audit, evaluation index and authoritative reports;
- the next highest-value experiment, without automatically starting external submission, opponent
  admission, commit or push.

The canonical truth remains local `training_metrics.jsonl` plus immutable version artifacts and
official-engine reports. W&B is the online mirror and comparison surface, not a substitute for
those records.
