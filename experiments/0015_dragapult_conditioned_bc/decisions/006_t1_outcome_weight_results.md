# Decision 006: terminal-outcome weighting does not replace V2

Date: 2026-07-27

## Decision

Keep V2 `T1_plus_pure_dragapult` as the strongest supported 0015 candidate. Do not continue tuning
loss weights after V4/V5. Terminal outcome may remain episode metadata for later value/RL work, but
the tested static BC weights do not provide a credible policy-strength improvement.

## Contract

Both weighted arms retain every one of the 54,764 T1 training decisions and the same 1,000 target
validation decisions. Each epoch contains 33,601 decisions from won games and 21,163 from lost
games. Win weight is 1.0; loss weight is the only changed variable. Outcome affects train token CE
only. Validation, checkpoint selection, candidate inputs and official-engine evaluation are
unweighted.

| Version | Loss weight | Mean decision weight | Best target exact | Formal W/L | Win rate |
|---|---:|---:|---:|---:|---:|
| V2 | 1.00 | 1.00000 | 58.7% | 39 / 161 | 19.5% |
| V4 | 0.75 | 0.90339 | 57.5% | 28 / 172 | 14.0% |
| V5 | 0.50 | 0.80678 | 59.0% | 41 / 159 | 20.5% |

All formal runs completed 200/200 games with zero errors. V4/V5 were selected at epoch 11 by the
same best target greedy exact rule used for V2. Both W&B runs synced successfully.

## Confirmation and interpretation

V5's first-run lead over V2 is only two wins in 200 games. Its frozen independent confirmation is
32-168 (16.0%), split evenly as 16 first-seat and 16 second-seat wins. Combining available
independent batches gives:

- V2: 79/400 = 19.75%, Wilson 95% `[16.14%, 23.93%]`;
- V5: 73/400 = 18.25%, Wilson 95% `[14.77%, 22.33%]`;
- V5−V2: −1.5pp, approximate interval `[-6.94pp, +3.94pp]`.

The evidence is compatible with no effect and does not justify selecting V5. V4's larger weight is
directionally harmful. Searching 0.6/0.4 or additional random runs after seeing these outcomes
would be post-hoc tuning on engine noise, so the sweep stops here.

## Provenance

- V4 formal: `evaluation/V4_t1_loss_weight_075.html`, run
  `run-bc6c5895675d4c34bd59bfd6df36bf80`.
- V5 formal: `evaluation/V5_t1_loss_weight_050.html`, run
  `run-c7a50c874a5a4b429bf12726de2103ff`.
- V5 confirmation: `.tmp/evaluation/0015_v5_loss_weight_050_confirmation/`
  `run-1361e457f7754afe83c68f995283b4c4/report.html`.
- Stable W&B URLs are stored in each version's `artifact/status.json`.
