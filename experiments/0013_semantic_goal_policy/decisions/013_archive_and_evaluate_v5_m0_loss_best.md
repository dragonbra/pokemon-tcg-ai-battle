# Decision 013: Archive and evaluate the V5 M0 loss-best snapshot

**Date:** 2026-07-26
**State:** accepted with evaluation correctness caveat

## Decision

Freeze epoch 14 of `V5_m0_100epoch_monitor` as the validation-loss-best M0 snapshot. The original
training checkpoint remains immutable in its version directory. Export an inference-only state-dict
checkpoint, self-contained CPU runtime, 60-card deck, and physical `cg/` runtime to:

`archive/submission/0013_v5_m0_epoch14_loss_best/`

The matching upload-form archive belongs in `archive/submission/dist/`. The repository-root
`submission/` path is retired for all new assets.

## Selection evidence

- Epoch 14 validation loss: `0.6907608200963037`
- Epoch 14 validation exact action: `0.7247479433413744`
- Epoch 14 validation legality: `1.0`
- Source checkpoint SHA-256:
  `e1eeed8bccbe04e4937bc6489038503928358df4c15b7e571f710922cc906cfa`
- Exported model-state SHA-256:
  `c068be385d4161366b01d655fd80aa6e393dc84559a13b974b06887c323c7a26`
- Deterministic tar.gz SHA-256:
  `352dda470a330df3c811ef974e01eb5831a48f67849590464a64dc1c52835fd3`

V5 completed epoch 16. Epochs 15 and 16 regressed validation loss while train metrics improved.
The training process was absent after writing only 38.58% of epoch 17 optimization progress, so
partial epoch 17 is excluded and V5 is recorded as interrupted.

## Official-engine evaluation

The authoritative report is `evaluation/V5_m0_100epoch_monitor.html`, run ID
`run-004c32a6a3414ca292f934b47af49c96`. It used the complete 20-opponent catalog, 10 games per
opponent, eight workers, one CPU thread per worker, and metric profile
`auto_iteration_v8_setup_relay`.

- 200 scheduled games: 92 wins, 98 losses, 10 errors, 0 unfinished.
- Fixed-denominator win rate: `46.0%`; completion rate: `95.0%`.
- Completed-game win rate: `92/190 = 48.42%`.
- Completed first-seat games: `47/95 = 49.47%`.
- Completed second-seat games: `45/95 = 47.37%`.

All ten errors are early engine errors against `alakazam_dudunsparce_03`; the other 19 opponents
produced 190 completed games without candidate runtime errors. The immutable report is retained as
observed. This run is official-engine evidence, but it does not pass a clean zero-error correctness
gate and cannot authorize automatic promotion to the formal opponent pool.
