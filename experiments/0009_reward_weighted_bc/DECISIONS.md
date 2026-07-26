# 0009 Reward-weighted BC Decisions

## 2026-07-24 correction: two-epoch screens are not final experiments

The earlier workflow incorrectly generalized V1's epoch-2 peak to every category and reward
setting. V2 through V6 were stopped after only two epochs. Those runs remain valid short-screen
records, but they do not establish each setting's own optimum or overfitting point. The final
selection and completed status are therefore retracted.

Long-horizon versions must record full train and validation evaluation every epoch, run long
enough to expose their own peak and decline, and only then enter official-engine gates. V1
remains a provisional control candidate while this correction is in progress.

### V7 long-horizon early setup

- Version: `V7_early_setup_long_curve`
- Purpose: rerun V3 from the frozen 0008 source for 10 epochs instead of stopping at epoch 2.
- Every epoch performs a second full pass over all 221,289 train records to record
  `train_eval/exact_action_rate`, alongside the full validation metrics.
- All optimizer, reward, dataset, source checkpoint, and seed settings match V3; only the
  horizon and complete train-evaluation logging change.

### V7 result

- Status: the user stopped the run during epoch 9 after eight complete epochs. The partial
  epoch 9 pass is not recorded as an epoch result and is not used for selection.
- Validation exact by epoch: `0.8064100`, `0.8085780`, `0.8074647`, `0.8065858`,
  `0.8053554`, `0.8052968`, `0.8058827`, `0.8048866`.
- Full-train exact by epoch: `0.8859365`, `0.8937995`, `0.8979434`, `0.9025573`,
  `0.9059149`, `0.9079936`, `0.9108406`, `0.9133667`.
- Validation loss rises from `0.5965859` at epoch 1 to `0.6019086` at epoch 2 and
  `0.6476776` at epoch 8, while full-train loss falls from `0.3022220` to `0.2288965`.
- Best: epoch 2, validation exact `0.8085779575`. Checkpoint SHA-256:
  `93b7795070f7bcc0ffedd061b66def609d5825e6f9bcc95e201d6dfcca5ca17f`.
- Decision: epoch 2 is selected for the next official-engine gate. The eight-point curve
  confirms this setting's own early peak and subsequent overfitting; it does not retroactively
  prove that the other two-epoch category/reward screens reached their own optima.

## Frozen inputs

- Base dataset: `/mnt/d/pokemon-tcg-ai-battle-data/rl_runs_dataset_0008/dataset.jsonl`
- Dataset SHA-256: `36415a61b93fdcdb2e1b17c0fbaebb9432f8d94ed0c203365b36cd3011b09bdf`
- Records: 221,289 train / 17,067 validation / 0 test
- Episodes: 2,875 train / 229 validation
- Expert policy: one team only, `Yushin Ito`
- Selection: winner-only daily trajectories, 2026-07-13 through 2026-07-22
- Source checkpoint: `0008-rank_010_alakazam_bc/V1_shared_config_streaming`, best epoch 12
- Source checkpoint SHA-256: `8141faf9ad65bd3fd75e53b489bb794a03e9636b79b222fc537daba8e0e6cd7c`
- Source validation exact action: `0.8008437335`

## Capacity decision

0008 used `d_model=192`, two Transformer layers, hidden size 384, and 3,539,395
parameters. Train exact action continued to `0.8951100145` by epoch 20 while validation
peaked at epoch 12 and ended at `0.7991445480`. This is already an overfitting signal, not
an under-capacity signal. 0009 therefore keeps the same model capacity unless a controlled
continuation cannot preserve the source validation score.

## Reward boundary

The frozen corpus is winner-only, so `terminal_outcome=+1` for every training record.
Terminal-only reward weighting cannot create relative sample weights with a dataset-mean
baseline and is not a useful 0009 ablation. The experiment budget is reserved for card
category, opening/second-turn setup, strict Dunsparce bridge, attack quality, and Post-KO
relay signals.

## V1 control

- Version: `V1_control_finetune`
- Hypothesis: a low-LR continuation from the 0008 best checkpoint can preserve or slightly
  improve validation exact action without reopening the high-LR overfitting phase.
- Config: same capacity and data, no category embedding, no reward, uniform weights,
  `lr=2e-5`, six epochs, batch size 256, seed 7.
- Stop/decision rule: keep the best validation checkpoint. If validation never reaches the
  0008 source score, use the source checkpoint as the policy baseline and treat V1 only as
  a trainer/control calibration; do not increase capacity from that result alone.

### V1 result

- Status: completed in 1,810.34 seconds; all six epochs recorded.
- Validation exact by epoch: `0.8066444`, `0.8081678`, `0.8072303`, `0.8063514`,
  `0.8050624`, `0.8050038`.
- Best: epoch 2, `0.8081678092`, which is `+0.0073240757` absolute over the 0008 source
  checkpoint.
- Decision: keep the existing 3,539,395-parameter capacity and `2e-5` continuation LR.
  Reward/category ablations should use a two-epoch budget for first-pass screening; longer
  runs already show a consistent validation decline. V1 is the offline control checkpoint,
  not yet an official-engine strength claim.

## V2 category only

- Version: `V2_category_only`
- Hypothesis: the fixed 9-class official card category lookup improves imitation beyond the
  V1 control without changing the dataset or action contract.
- Fairness: restart from the same frozen 0008 source checkpoint, seed, LR, batch size, and
  capacity; the only model-input variable is category embedding enabled. Screen for two
  epochs, matching the V1-selected continuation horizon.

### V2 result

- Validation exact: `0.8064686` at epoch 1 and `0.8079334` at epoch 2.
- Delta versus V1 at epoch 2: `-0.0002343704` absolute (`-0.0234` percentage points).
- Decision: retain the feature implementation but do not carry it into the reward ablation
  spine. The one-seed result is effectively neutral and provides no positive evidence yet;
  it can be revisited after a reward profile has passed stronger gates.

## V3 early setup reward

- Version: `V3_early_setup_rwb`
- Hypothesis: upweighting trajectories that choose Active Abra when available and complete
  second-turn Powerful Hand improves the desired early-game policy without changing input
  features.
- Fairness: restart from the same 0008 source with category disabled, two epochs, and the
  V1 optimizer settings. Only `opening_active_abra_credit` and `second_turn_setup_credit`
  affect weights.
- Weighting: exponential reward-weighted BC around the frozen dataset mean, component
  weights `0.5` and `0.25`, clamped to `[0.25, 3.0]`; value loss remains disabled so this
  version tests sample weighting rather than a second learned objective.

### V3 result

- Validation exact: `0.8064100` at epoch 1 and `0.8085780` at epoch 2.
- Delta versus V1 at epoch 2: `+0.0004101482` absolute (`+0.0410` percentage points).
- Train ESS ratio: `0.9496592`; observed weights `0.5708` to `2.0647` in the final epoch.
- Decision: modestly positive offline screen with healthy weights. Carry early setup into
  the progressive reward spine, while withholding any official strength claim until engine
  evaluation.

## V4 early setup plus strict bridge

- Version: `V4_setup_bridge_rwb`
- Hypothesis: adding credit for the rare Dunsparce first-turn preparation and second-turn
  Ability handoff improves the coherent bridge without destabilizing the broader setup gain.
- Added weights: `0.25` for first-turn Bench Abra preparation and `0.5` for strict
  second-turn execution. Category remains disabled and value loss remains zero.
- Preflight: 4,607 positive / 445 negative first-turn records and 2,522 positive / 8,902
  negative execution records in train; projected ESS ratio `0.9318`, so the profile remains
  conservative enough for a two-epoch screen.

### V4 result

- Validation exact: `0.8065272` at epoch 1 and `0.8082850` at epoch 2.
- Delta versus V3 at epoch 2: `-0.0002929630` absolute (`-0.0293` percentage points).
- Slice evidence versus V3: bridge first-turn positive unchanged at `405/458`; bridge
  execution positive fell from `287/325` to `286/325`; execution negative fell from
  `667/791` to `665/791`.
- Decision: reject the bridge weights from the progressive spine. They neither improve the
  target slices nor preserve the V3 overall score. Keep the annotation component available
  for future reweighting or larger-data experiments.

## V5 early setup plus attack quality

- Version: `V5_setup_attack_quality_rwb`
- Hypothesis: reward decisions leading to Prize gain and downweight attacks that take no
  Prize, while preserving the V3 early setup improvement.
- Added weights: `+0.25 * attack_prize_delta` and `-0.25 * non_prize_attack`. The separate
  Powerful Hand penalty stays off to avoid double-penalizing the same decision.
- Preflight: 10,101 positive Prize records and 6,678 non-Prize attack records in train;
  projected weights `0.3564` to `2.6336`, ESS ratio `0.9434`.

### V5 result

- Validation exact: `0.8071717` at epoch 1 and `0.8085780` at epoch 2; epoch 2 ties V3
  overall and remains `+0.0410` percentage points over V1.
- Target slice versus V3: Prize-positive attack exact rose from `585/768` (`76.17%`) to
  `605/768` (`78.78%`); non-Prize attack stayed `401/498` and did not gain imitation.
- Decision: carry attack quality. The unchanged overall score hides a material improvement
  on successful attack decisions without increasing imitation of the explicitly downweighted
  non-Prize slice.

## V6 setup, attack quality, and Post-KO relay

- Version: `V6_setup_attack_relay_rwb`
- Hypothesis: add Post-KO continuation credit and downweight recoverable discard-route misses
  without erasing the V5 attack-quality slice gain.
- Added weights: `+0.5 * post_ko_relay_credit` and
  `-0.25 * recoverable_discard_miss`.
- Preflight: 199 positive relay, 4,570 negative relay, and 591 recoverable discard-miss
  records in train; projected weights `0.3602` to `2.6616`, ESS ratio `0.9388`. The positive
  validation slice has only 18 records, so this screen is exploratory and cannot support a
  strong relay claim by itself.

### V6 result

- Validation exact: `0.8072889` at epoch 1 and `0.8084608` at epoch 2, two exact records
  below V5 at the selected epoch.
- Slice evidence versus V5: Post-KO positive stayed `11/18`, Post-KO negative stayed
  `343/371`, and recoverable discard miss stayed `42/51`. The V5 Prize-positive gain also
  remained unchanged at `605/768`.
- Decision: do not carry relay into the final candidate. The component is data-limited and
  produced no measurable target-slice change; retain it as a switch for a larger corpus.

## Candidate gate

- Offline control: `V1_control_finetune`.
- Offline selected reward profile: `V5_setup_attack_quality_rwb`.
- Reason: V5 ties the best overall validation exact while improving Prize-positive attack
  exact by 20 records over V3 and V1-era behavior; bridge and relay additions failed their
  target-slice gates.
- Next evidence: export both as category-free inference packages and compare them using the
  same official-engine opponent catalog and metric profile. Offline accuracy alone does not
  choose the final engine candidate.

## Earlier official engine gate and superseded decision

All promoted packages were validated as self-contained 60-card candidates and evaluated
against the same enabled 23-opponent catalog, 10 games per opponent, with 8 isolated workers,
one CPU thread per worker, and `auto_iteration_v8_setup_relay` revision 3.

| Candidate | Outcome | G0 | 2T Powerful Hand | Post-KO | Non-Prize attacks |
| --- | ---: | --- | ---: | ---: | ---: |
| V1 control | 185/230 | 0 errors, 0 unfinished | 79/230 | 215/562 | 219/956 |
| V3 early setup | 180/230 | 1 game error | 73/230 | 223/549 | 203/921 |
| V5 attack quality | 172/230 | 0 errors, 1 unfinished | 83/230 | 206/566 | 218/909 |

The runs are independent stochastic official-engine reports, so their win differences are not
treated as paired causal estimates. G0 is nevertheless a hard gate: V3 and V5 cannot be
promoted from these reports. V5 also fails its intended engine-side attack-quality direction:
the non-Prize rate is `23.98%` versus V1's `22.91%`, and the Powerful Hand subset is `15.71%`
versus `13.75%`.

The earlier selection was `V1_control_finetune`, checkpoint SHA-256
`b9522b808d547a6080c2bd11d6920603b44c8f0a3567c30134258199bb0a244b`. It is the only new
candidate with 230/230 completed, zero errors, and zero unfinished games. Reward experiments
remain useful negative/diagnostic evidence from that gate.

That final label is superseded by the V7 long-horizon correction. At the user's direction,
V7 epoch 2 is now the selected checkpoint for a fresh package validation and official-engine
run. No new final candidate is declared until that report completes.

## V7 official gate and submission

V7 epoch 2 was exported as a category-free full-action package and evaluated against the same
enabled 23-opponent catalog, 10 games per opponent, with 8 isolated workers, one CPU thread per
worker, and `auto_iteration_v8_setup_relay` revision 3.

- G0: 230/230 completed, zero errors, zero unfinished games.
- Outcome: 174 wins / 56 losses (`75.65%`); first 91/115, second 83/115.
- Second-turn Powerful Hand: 80/230; Dunsparce bridge: 0/84.
- Post-KO immediate relay: 214/570 (`37.54%`).
- Non-Prize attacks: 219/940 (`23.30%`); Powerful Hand subset: 113/826 (`13.68%`).
- Report: `rl_runs/evaluation/0009-reward_weighted_bc/V7_early_setup_long_curve/`
  `run-426d4bd49123409baa22610471657d91/report.html`.

The independent V7 run has fewer wins than the earlier V1 run (174 versus 185), and its
process metrics do not establish a broad strategy improvement. The reports are stochastic
and unpaired, but there is still no evidence for calling V7 stronger. The user explicitly
selected V7 epoch 2 for the next deliverable; because it passes G0, it is the submitted 0009
artifact without a SOTA claim.

The validated archive is `submission/dist/alakazam_reward_weighted_bc_v7.tar.gz`, SHA-256
`f78872528213c79df5b3da5cff01ec3db36608aca1d918f69dbf0a4a04f55776`. One Kaggle
submission was created as ref `54954628`; its status was `PENDING` immediately after upload.
No automatic retry is permitted.
